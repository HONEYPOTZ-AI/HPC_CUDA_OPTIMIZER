// =============================================================================
// HPC CUDA Optimizer Library — Combined ZeRO-3 + Tensor Parallel Training Demo
// examples/train_zero3_tp.cu
//
// Demonstrates a realistic multi-GPU training loop that combines:
//   • ZeRO-3 parameter + optimizer state sharding  (hpc_zero3.cuh)
//   • Megatron-LM Tensor Parallelism for FFN/Attention (hpc_tensor_parallel.cuh)
//   • Mixed-precision BF16 forward, FP32 master weights
//   • WarmupCosine learning-rate schedule
//   • Gradient clipping on sharded gradients
//   • Checkpoint save every N steps
//
// Model:  GPT-style transformer (simplified, compute-focused)
//   Config:  hidden=2048, ffn=8192, heads=16, layers=24
//   Params:  ~1.3B (before TP sharding across T ranks)
//   Sequence length: 2048 tokens
//   Batch: 4 sequences per DP rank
//
// Parallelism layout:
//   Total GPU count = DP × TP
//   e.g. 32 GPUs: DP=4, TP=8  →  each TP group of 8 shares one layer
//                                  ZeRO-3 shards across 4 DP replicas
//
// Launch:
//   torchrun --nproc_per_node=8 train_zero3_tp.cu   [not Python — see note]
//   mpirun -np 8 ./train_zero3_tp
//
// Single-GPU stub (no NCCL): runs on rank 0 / world 1, stubs comms
//   nvcc -std=c++17 -O2 -I../include \
//        -gencode arch=compute_80,code=sm_80 \
//        train_zero3_tp.cu -o train_zero3_tp && ./train_zero3_tp
//
// Multi-GPU build:
//   nvcc -std=c++17 -O2 -DHPC_HAVE_NCCL -DHPC_HAVE_MPI -I../include \
//        -gencode arch=compute_80,code=sm_80 \
//        train_zero3_tp.cu -lnccl -lmpi -lcublas -o train_zero3_tp
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cassert>

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_comm.cuh"
#include "hpc_lr_scheduler.cuh"
#include "hpc_optimizer.cuh"
#include "hpc_profiler.cuh"
#include "hpc_zero2.cuh"
#include "hpc_zero2_optimizer.cuh"
#include "hpc_zero3.cuh"
#include "hpc_tensor_parallel.cuh"

using namespace hpc_opt;
using namespace hpc_opt::zero2;
using namespace hpc_opt::zero3;
using namespace hpc_opt::tp;

// ============================================================================
// Configuration
// ============================================================================
struct ModelConfig {
    int hidden_dim  = 2048;
    int ffn_dim     = 8192;
    int n_heads     = 16;
    int n_layers    = 24;
    int vocab_size  = 32000;
    int max_seq_len = 2048;
};

struct TrainConfig {
    int    batch_size      = 4;       // sequences per DP rank
    int    total_steps     = 1000;
    int    warmup_steps    = 100;
    float  lr_max          = 3e-4f;
    float  lr_min          = 1e-5f;
    float  beta1           = 0.9f;
    float  beta2           = 0.95f;
    float  eps             = 1e-8f;
    float  weight_decay    = 0.1f;
    float  max_grad_norm   = 1.0f;
    int    ckpt_interval   = 200;
    int    log_interval    = 10;
    bool   fp16            = false;
    bool   bf16            = true;    // default: BF16 activations, FP32 master
    int    tp_size         = 1;       // tensor parallelism degree
    const char* ckpt_dir   = "/tmp/hpc_zero3_ckpts";
};

// ============================================================================
// Print banner
// ============================================================================
static void print_banner(const ModelConfig& mc, const TrainConfig& tc,
                         int rank, int world, int dp_size)
{
    if (rank != 0) return;

    auto param_count = [&]() -> size_t {
        // Embedding:        vocab * H
        // LayerNorm (each): 2H
        // Attention QKV:    H * 3H
        // Attention out:    H * H
        // FFN fc1:          H * ffn
        // FFN fc2:          ffn * H
        // Final LN + LM head: H * vocab
        size_t emb  = (size_t)mc.vocab_size * mc.hidden_dim;
        size_t attn = (size_t)mc.n_layers * (3ULL*mc.hidden_dim*mc.hidden_dim
                                              + mc.hidden_dim*mc.hidden_dim);
        size_t ffn  = (size_t)mc.n_layers * 2ULL * mc.hidden_dim * mc.ffn_dim;
        size_t ln   = (size_t)(mc.n_layers * 2 + 1) * 2 * mc.hidden_dim;
        size_t head = (size_t)mc.hidden_dim * mc.vocab_size;
        return emb + attn + ffn + ln + head;
    };

    size_t total_params = param_count();
    size_t param_bytes  = total_params * 16;  // ZeRO-0: 16 bytes/param
    size_t z3_bytes     = param_bytes / world; // ZeRO-3: /world_size

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║        HPC Optimizer — ZeRO-3 + Tensor Parallel Demo        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  Model\n");
    printf("    Hidden dim     : %d\n",  mc.hidden_dim);
    printf("    FFN dim        : %d\n",  mc.ffn_dim);
    printf("    Attention heads: %d\n",  mc.n_heads);
    printf("    Layers         : %d\n",  mc.n_layers);
    printf("    Vocab size     : %d\n",  mc.vocab_size);
    printf("    Seq length     : %d\n",  mc.max_seq_len);
    printf("    ~Parameters    : %.2fB\n", total_params / 1e9);
    printf("  Training\n");
    printf("    Batch / DP rank: %d seqs\n", tc.batch_size);
    printf("    Steps          : %d  (warmup %d)\n",
           tc.total_steps, tc.warmup_steps);
    printf("    LR             : %.2e → %.2e (WarmupCosine)\n",
           tc.lr_max, tc.lr_min);
    printf("    Grad clip norm : %.1f\n",     tc.max_grad_norm);
    printf("    Precision      : %s\n",
           tc.bf16 ? "BF16 fwd/bwd + FP32 master" : "FP32");
    printf("  Parallelism\n");
    printf("    World size     : %d GPUs\n", world);
    printf("    DP size        : %d\n", dp_size);
    printf("    TP size        : %d\n", tc.tp_size);
    printf("  Memory (estimated, per DP rank)\n");
    printf("    ZeRO-0         : %.1f GB\n", param_bytes / 1e9);
    printf("    ZeRO-3         : %.1f GB   (%.1f× saving vs ZeRO-0)\n",
           z3_bytes / 1e9, (float)world);
    printf("──────────────────────────────────────────────────────────────\n\n");
}

// ============================================================================
// Synthetic data generator
//   Produces random token IDs [batch, seq_len] on device
// ============================================================================
__global__ void k_random_tokens(int32_t* __restrict__ tokens,
                                 size_t n, int vocab_size, uint64_t seed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // LCG random
    uint64_t v = seed ^ (i * 6364136223846793005ULL + 1442695040888963407ULL);
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    tokens[i] = (int32_t)((uint32_t)v % (uint32_t)vocab_size);
}

// ============================================================================
// Minimal transformer layer (compute-placeholder)
//
// In a full implementation each layer would be:
//   LN → ColParallel(QKV) → Attention → RowParallel(Out)
//              ↓
//   LN → ColParallel(FC1) → GELU → RowParallel(FC2)
//
// Here we demonstrate the full wiring with the library structs.
// Actual backward passes would use autograd or hand-written CUDA;
// for this demo we synthesize gradients proportional to activations.
// ============================================================================
struct TransformerLayer {
    SequenceParallelLayerNorm ln1, ln2;
    ColumnParallelLinear       qkv_proj;   // ColParallel: H → 3H/T
    RowParallelLinear          out_proj;   // RowParallel: H/T → H
    ColumnParallelLinear       fc1;        // ColParallel: H → 4H/T
    RowParallelLinear          fc2;        // RowParallel: 4H/T → H

    void init(const ModelConfig& mc, int tp_rank, int tp_sz, Dtype dt,
              size_t max_tok)
    {
        int H   = mc.hidden_dim;
        int ffn = mc.ffn_dim;

        ln1.init(H, dt, 1e-5f, max_tok);
        ln2.init(H, dt, 1e-5f, max_tok);

        // QKV: input H → output 3H (col parallel → each rank gets 3H/T)
        qkv_proj.init(H, 3*H, tp_rank, tp_sz,
                      /*gather=*/false, /*bias=*/true, dt, max_tok);
        // Out: input H/T (from QKV shard) → output H (row parallel)
        out_proj.init(H, H, tp_rank, tp_sz,
                      /*bias=*/true, dt, max_tok);
        // FFN
        fc1.init(H, ffn, tp_rank, tp_sz, false, true, dt, max_tok);
        fc2.init(ffn, H, tp_rank, tp_sz, true,  dt, max_tok);
    }

    // Returns: [tokens, H]  (same shape as input)
    void* forward(const TPContext& tpc, const void* x, int tokens,
                  cudaStream_t cs)
    {
        // Attention sub-layer (simplified: skip actual softmax attention)
        void* ln1_out = ln1.forward(x, tokens, cs);
        // QKV projection → [tokens, 3H/T]
        void* qkv     = qkv_proj.forward(tpc, ln1_out, tokens, cs);
        // In a full impl: reshape QKV, compute scaled dot-product attention...
        // Out projection: take first H/T slice as mock "attention output"
        void* attn_out = out_proj.forward(tpc, qkv, tokens, cs);
        // (Residual add would go here — omitted for brevity)

        // FFN sub-layer
        void* ln2_out  = ln2.forward(attn_out, tokens, cs);
        void* fc1_out  = fc1.forward(tpc, ln2_out, tokens, cs);
        // GELU inplace on fc1_out [tokens, ffn/T]
        size_t ffn_local = fc1.local_out;
        k_gelu_inplace<__nv_bfloat16><<<
            hpc_blocks((size_t)tokens * ffn_local), HPC_BLOCK, 0, cs>>>(
            static_cast<__nv_bfloat16*>(fc1_out),
            (size_t)tokens * ffn_local);
        void* fc2_out  = fc2.forward(tpc, fc1_out, tokens, cs);
        return fc2_out;
    }

    size_t total_params_per_layer(const ModelConfig& mc, int tp_sz) const {
        int H = mc.hidden_dim, ffn = mc.ffn_dim;
        // QKV weight: H * 3H/T, Out weight: H/T * H, FC1: H * ffn/T, FC2: ffn/T * H
        // Plus biases (small)
        return (size_t)(H*(3*H/tp_sz) + (H/tp_sz)*H
                      + H*(ffn/tp_sz) + (ffn/tp_sz)*H);
    }

    void destroy() {
        ln1.destroy(); ln2.destroy();
        qkv_proj.destroy(); out_proj.destroy();
        fc1.destroy(); fc2.destroy();
    }
};

// ============================================================================
// Synthetic "loss" and gradient computation
//   loss = mean(||output - target||^2) / 2
//   dloss/doutput = (output - target) / N
// ============================================================================
__global__ void k_synthetic_loss(
        const __nv_bfloat16* __restrict__ output,
        const __nv_bfloat16* __restrict__ target,
        float* __restrict__ loss_out,
        size_t N)
{
    __shared__ float sdata[HPC_BLOCK];
    float sum = 0.f;
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < N;
         i += gridDim.x*blockDim.x) {
        float diff = __bfloat162float(output[i]) - __bfloat162float(target[i]);
        sum += diff * diff;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < (size_t)s) sdata[threadIdx.x] += sdata[threadIdx.x+s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_out, sdata[0] / (2.0f * N));
}

__global__ void k_compute_grad(
        const __nv_bfloat16* __restrict__ output,
        const __nv_bfloat16* __restrict__ target,
        float* __restrict__ grad,
        size_t N)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < N;
         i += gridDim.x*blockDim.x)
        grad[i] = (__bfloat162float(output[i])
                 - __bfloat162float(target[i])) / (float)N;
}

// ============================================================================
// ZeRO-3 + TP trainer
// ============================================================================
struct Zero3TPTrainer {
    // ZeRO-3 engine (DP sharding)
    ZeRO3Trainer zero3;

    // TP layers (replicated per DP rank, sharded across TP)
    std::vector<TransformerLayer> layers;
    VocabParallelEmbedding        embed;

    TPContext tp_ctx;

    ModelConfig mc;
    TrainConfig tc;

    int world, rank, dp_rank, dp_size;
    int tp_rank, tp_size_val;

    cudaStream_t compute_stream, comm_stream;

    // LR scheduler
    WarmupCosineScheduler* lr_sched = nullptr;

    // Profiler
    IterationProfiler* profiler = nullptr;

    void init(const ModelConfig& m, const TrainConfig& t,
              CommContext& comm)
    {
        mc   = m;
        tc   = t;
        rank  = comm.rank();
        world = comm.world_size();

        // Layout: TP contiguous, DP outer
        // rank = dp_rank * tp_size + tp_rank
        tp_size_val = tc.tp_size;
        tp_rank     = rank % tp_size_val;
        dp_rank     = rank / tp_size_val;
        dp_size     = world / tp_size_val;

        HPC_CUDA_CHECK(cudaStreamCreate(&compute_stream));
        HPC_CUDA_CHECK(cudaStreamCreate(&comm_stream));

        // Build TP context for this rank's TP group
        tp_ctx = TPContext::from_comm(comm, compute_stream, comm_stream);
        tp_ctx.tp_rank  = tp_rank;
        tp_ctx.tp_size  = tp_size_val;

        Dtype dt = tc.bf16 ? Dtype::BF16 :
                   tc.fp16 ? Dtype::FP16 : Dtype::FP32;

        size_t max_tok = (size_t)tc.batch_size * mc.max_seq_len;

        // Vocab embedding (TP-sharded)
        embed.init(mc.vocab_size, mc.hidden_dim,
                   tp_rank, tp_size_val, dt, max_tok);

        // Transformer layers
        layers.resize(mc.n_layers);
        for (int l = 0; l < mc.n_layers; ++l)
            layers[l].init(mc, tp_rank, tp_size_val, dt, max_tok);

        // ZeRO-3: shard embedding + layer weights across DP group
        // Compute total parameter count for this TP rank
        size_t local_params = (size_t)mc.hidden_dim * (mc.vocab_size / tp_size_val);  // embed
        for (int l = 0; l < mc.n_layers; ++l)
            local_params += layers[l].total_params_per_layer(mc, tp_size_val);

        ZeRO3TrainerConfig z3c;
        z3c.lr           = tc.lr_max;
        z3c.beta1        = tc.beta1;
        z3c.beta2        = tc.beta2;
        z3c.eps          = tc.eps;
        z3c.weight_decay = tc.weight_decay;
        z3c.max_grad_norm = tc.max_grad_norm;
        z3c.release_full_params_immediately = true;

        zero3.init(local_params, dp_rank, dp_size, z3c, comm,
                   compute_stream, comm_stream);

        // LR scheduler
        lr_sched = new WarmupCosineScheduler(
            tc.lr_max, tc.lr_min, tc.warmup_steps, tc.total_steps);

        // Profiler
        profiler = new IterationProfiler(rank == 0);

        if (rank == 0) {
            printf("[ZeRO-3+TP] Initialized:\n");
            printf("  TP rank %d/%d, DP rank %d/%d\n",
                   tp_rank, tp_size_val, dp_rank, dp_size);
            printf("  Local params (TP-sharded): %.2fM\n",
                   local_params / 1e6);
            zero3.engine.print_memory_stats(rank, dp_size);
        }
    }

    // -------------------------------------------------------------------------
    // One training step
    // -------------------------------------------------------------------------
    float step(int step_idx, int32_t* d_tokens)
    {
        float lr = lr_sched->get_lr(step_idx);
        zero3.trainer_config.lr = lr;

        int tokens = tc.batch_size * mc.max_seq_len;

        // ---- Forward pass ----
        profiler->start_forward();

        // Prefetch params from shards
        zero3.prefetch_params();

        // Embedding lookup
        void* hidden = embed.forward(tp_ctx, d_tokens, tokens, compute_stream);

        // Transformer layers
        for (int l = 0; l < mc.n_layers; ++l)
            hidden = layers[l].forward(tp_ctx, hidden, tokens, compute_stream);

        // Synthetic target = zeros (cross-entropy would go here)
        Dtype dt = tc.bf16 ? Dtype::BF16 : Dtype::FP32;
        size_t out_bytes = (size_t)tokens * mc.hidden_dim *
                           (dt == Dtype::FP32 ? 4 : 2);

        __nv_bfloat16 *d_target, *d_hidden_bf16;
        cudaMalloc(&d_target, out_bytes);
        cudaMemset(d_target, 0, out_bytes);
        d_hidden_bf16 = static_cast<__nv_bfloat16*>(hidden);

        // Compute loss
        float *d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));
        k_synthetic_loss<<<hpc_blocks((size_t)tokens*mc.hidden_dim), HPC_BLOCK,
                           0, compute_stream>>>(
            d_hidden_bf16, d_target, d_loss,
            (size_t)tokens * mc.hidden_dim);

        float h_loss;
        cudaMemcpyAsync(&h_loss, d_loss, sizeof(float),
                        cudaMemcpyDeviceToHost, compute_stream);

        profiler->end_forward();

        // ---- Backward pass ----
        profiler->start_backward();

        // Release full params (ZeRO-3 memory saving)
        zero3.release_params();

        // Synthesize gradients (in a real system: autograd / custom bwd kernels)
        float *d_grad;
        cudaMalloc(&d_grad, (size_t)tokens * mc.hidden_dim * sizeof(float));
        // grad = (output - target) / N
        k_compute_grad<<<hpc_blocks((size_t)tokens*mc.hidden_dim), HPC_BLOCK,
                         0, compute_stream>>>(
            d_hidden_bf16, d_target, d_grad,
            (size_t)tokens * mc.hidden_dim);

        // ZeRO-3 backward: pack grads → ReduceScatter → clip → optimizer step
        // For this demo, we use the engine's grad bucket directly
        float *d_grad_bucket = zero3.engine.grad_flat;
        size_t shard_numel   = zero3.engine.layout.shard_size;

        // Copy our synthetic grad into the shard grad (scaled to shard size)
        size_t copy_n = std::min((size_t)tokens * mc.hidden_dim, shard_numel);
        cudaMemcpyAsync(d_grad_bucket, d_grad,
                        copy_n * sizeof(float),
                        cudaMemcpyDeviceToDevice, compute_stream);
        if (copy_n < shard_numel)
            cudaMemsetAsync(d_grad_bucket + copy_n, 0,
                            (shard_numel - copy_n) * sizeof(float),
                            compute_stream);

        zero3.backward_step(step_idx + 1);
        profiler->end_backward();

        // Synchronize and read loss
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(comm_stream);

        cudaFree(d_target);
        cudaFree(d_loss);
        cudaFree(d_grad);

        return h_loss;
    }

    void save_checkpoint(int step_idx) {
        char path[512];
        snprintf(path, sizeof(path), "%s/step_%06d_dp%d_tp%d.bin",
                 tc.ckpt_dir, step_idx, dp_rank, tp_rank);
        zero3.save_checkpoint(path);
        if (rank == 0)
            printf("[CKPT] Saved checkpoint: %s\n", path);
    }

    void destroy() {
        for (auto& l : layers) l.destroy();
        embed.destroy();
        tp_ctx.destroy();
        delete lr_sched;
        delete profiler;
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(comm_stream);
    }
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    // Parse optional CLI overrides
    ModelConfig mc;
    TrainConfig tc;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layers")  == 0 && i+1 < argc) mc.n_layers    = atoi(argv[++i]);
        if (strcmp(argv[i], "--hidden")  == 0 && i+1 < argc) mc.hidden_dim  = atoi(argv[++i]);
        if (strcmp(argv[i], "--ffn")     == 0 && i+1 < argc) mc.ffn_dim     = atoi(argv[++i]);
        if (strcmp(argv[i], "--steps")   == 0 && i+1 < argc) tc.total_steps = atoi(argv[++i]);
        if (strcmp(argv[i], "--tp")      == 0 && i+1 < argc) tc.tp_size     = atoi(argv[++i]);
        if (strcmp(argv[i], "--batch")   == 0 && i+1 < argc) tc.batch_size  = atoi(argv[++i]);
        if (strcmp(argv[i], "--fp32")    == 0)  { tc.bf16 = false; tc.fp16 = false; }
        if (strcmp(argv[i], "--fp16")    == 0)  { tc.bf16 = false; tc.fp16 = true;  }
    }

#ifdef HPC_HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    // Initialize communicator
    CommContext comm;
    comm.init();
    int rank  = comm.rank();
    int world = comm.world_size();
    cudaSetDevice(rank % 8);  // up to 8 GPUs per node

    if (rank == 0) print_hpc_banner();

    int dp_size = world / tc.tp_size;
    print_banner(mc, tc, rank, world, dp_size);

    // ---- Initialize trainer ----
    Zero3TPTrainer trainer;
    trainer.init(mc, tc, comm);

    // ---- Synthetic data ----
    int total_tokens = tc.batch_size * mc.max_seq_len;
    int32_t *d_tokens;
    cudaMalloc(&d_tokens, total_tokens * sizeof(int32_t));

    // ---- Training loop ----
    auto t_start = std::chrono::high_resolution_clock::now();
    float running_loss = 0.f;

    if (rank == 0) {
        printf("%-8s %-10s %-10s %-12s %-14s\n",
               "Step", "Loss", "LR", "Tokens/s", "Time(ms)");
        printf("─────────────────────────────────────────────────────\n");
    }

    for (int step = 0; step < tc.total_steps; ++step) {
        // Generate synthetic batch
        k_random_tokens<<<hpc_blocks(total_tokens), HPC_BLOCK, 0, 0>>>(
            d_tokens, total_tokens, mc.vocab_size,
            (uint64_t)step * 1234567891ULL + rank);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        float loss = trainer.step(step, d_tokens);
        auto t1 = std::chrono::high_resolution_clock::now();

        running_loss = (step == 0) ? loss :
                       0.9f * running_loss + 0.1f * loss;

        if (rank == 0 && (step % tc.log_interval == 0 || step == tc.total_steps - 1)) {
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double total_s = std::chrono::duration<double>(t1 - t_start).count();
            double toks_s = total_tokens * world / (ms / 1e3);
            float  lr_now = trainer.lr_sched->get_lr(step);

            printf("%-8d %-10.4f %-10.2e %-12.0f %-14.1f\n",
                   step, running_loss, lr_now, toks_s, ms);
        }

        // Checkpoint
        if (tc.ckpt_interval > 0 && (step + 1) % tc.ckpt_interval == 0)
            trainer.save_checkpoint(step + 1);
    }

    if (rank == 0) {
        printf("─────────────────────────────────────────────────────\n");
        printf("Training complete. Final loss: %.4f\n\n", running_loss);

        // Memory summary
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("GPU memory: %.1f GB used / %.1f GB total\n",
               (total_mem - free_mem) / 1e9, total_mem / 1e9);
    }

    // ---- Memory comparison table ----
    if (rank == 0) {
        size_t psi       = 1'300'000'000ULL;  // approx model params
        size_t bytes_fp  = tc.bf16 ? 2 : 4;
        size_t z0_bytes  = 16ULL * psi;
        size_t z1_bytes  = (8 + 8/world) * psi;  // rough
        size_t z2_bytes  = (4 + 12/world) * psi;
        size_t z3_bytes  = 16ULL * psi / world;
        printf("\nMemory savings summary (1.3B params, %d GPUs):\n", world);
        printf("  %-12s  %6.1f GB  (baseline)\n", "ZeRO-0", z0_bytes / 1e9);
        printf("  %-12s  %6.1f GB  (%.1f×)\n",
               "ZeRO-1", z1_bytes / 1e9, (float)z0_bytes/z1_bytes);
        printf("  %-12s  %6.1f GB  (%.1f×)\n",
               "ZeRO-2", z2_bytes / 1e9, (float)z0_bytes/z2_bytes);
        printf("  %-12s  %6.1f GB  (%.1f×)  ← active\n",
               "ZeRO-3", z3_bytes / 1e9, (float)z0_bytes/z3_bytes);
    }

    // ---- Cleanup ----
    cudaFree(d_tokens);
    trainer.destroy();
    comm.destroy();

#ifdef HPC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
