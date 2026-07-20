// =============================================================================
// HPC CUDA Optimizer Library
// examples/train_multigpu.cu  –  Multi-GPU distributed training demo
//
// Demonstrates:
//   • NCCL all-reduce gradient synchronisation across N GPUs
//   • AdamW + WarmupCosine LR on a simulated transformer-scale model
//   • BF16 params + FP32 master weights (Ampere+ path)
//   • Gradient clipping with FP16-compressed all-reduce
//   • NVTX profiling ranges for Nsight Systems
//   • Periodic checkpoint saving (root rank only)
//   • IterationProfiler throughput logging
//
// Launch (torchrun style – sets RANK/WORLD_SIZE/LOCAL_RANK env vars):
//   torchrun --nproc_per_node=8 train_multigpu   (single node, 8× A100)
//   mpirun -n 16 ./train_multigpu                 (multi-node, 2× DGX)
//
// Single-GPU fallback (WORLD_SIZE not set):
//   ./train_multigpu
//
// Build:
//   cmake .. -DCUDA_ARCHS="80;90" -DHPC_ENABLE_NCCL=ON -DHPC_ENABLE_MPI=ON
//   make -j$(nproc) train_multigpu
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/hpc_optimizer.cuh"
#include "../include/hpc_profiler.cuh"

using namespace hpc_opt;

// ---------------------------------------------------------------------------
// Simulated model: 7 parameter tensors (GPT-3 175B shard proportions scaled)
// ---------------------------------------------------------------------------
static constexpr int N_TENSORS = 7;
static const size_t TENSOR_SIZES[N_TENSORS] = {
    32768ULL * 4096,   // ~134M  – embedding table
    4096ULL  * 4096,   // ~16M   – attention QKV
    4096ULL  * 4096,   // ~16M   – attention projection
    4096ULL  * 16384,  // ~67M   – FFN up-projection
    16384ULL * 4096,   // ~67M   – FFN down-projection
    4096ULL  * 4096,   // ~16M   – layer norm
    4096ULL  * 50257,  // ~205M  – output lm-head
};

static constexpr int    TOTAL_STEPS    = 500;
static constexpr int    WARMUP_STEPS   = 50;
static constexpr int    LOG_EVERY      = 25;
static constexpr int    CKPT_EVERY     = 100;
static constexpr float  BASE_LR        = 3e-4f;
static constexpr float  WEIGHT_DECAY   = 0.1f;
static constexpr float  MAX_GRAD_NORM  = 1.0f;
static constexpr int    BATCH_SIZE     = 2048;   // global batch tokens
static constexpr int    SEQ_LEN        = 2048;

// ---------------------------------------------------------------------------
// Synthetic gradient kernel (simulates backward pass output)
// ---------------------------------------------------------------------------
__global__ void synthetic_backward_bf16(
        __nv_bfloat16* __restrict__ grads,
        size_t numel, float scale, uint32_t seed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < numel; i += gridDim.x * blockDim.x) {
        uint32_t s = (uint32_t)(i * 1664525u + seed * 1013904223u);
        float r    = (float)((int)s) * (1.0f / 2147483648.0f);
        grads[i]   = __float2bfloat16(r * scale);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // ---- Distributed context ----
    CommContext comm;
    comm.init_from_env();
    const int rank  = comm.rank();
    const int world = comm.world_size();

    if (rank == 0) {
        print_hpc_banner(comm.dist_config());
        printf("  Model: %d tensors  |  batch=%d tokens  |  seq=%d\n",
               N_TENSORS, BATCH_SIZE * world, SEQ_LEN);

        size_t total_params = 0;
        for (int i = 0; i < N_TENSORS; ++i) total_params += TENSOR_SIZES[i];
        printf("  Total params per rank: %.1f M\n\n",
               static_cast<double>(total_params) / 1e6);
    }

    cudaStream_t stream = comm.compute_stream();

    // ---- Allocate BF16 params + FP32 master weights + BF16 grads ----
    __nv_bfloat16* d_params_bf16[N_TENSORS];
    float*         d_master_fp32[N_TENSORS];
    __nv_bfloat16* d_grads_bf16[N_TENSORS];

    TensorView params[N_TENSORS];
    TensorView grads[N_TENSORS];

    for (int i = 0; i < N_TENSORS; ++i) {
        size_t n     = TENSOR_SIZES[i];
        size_t bytes_bf16 = n * sizeof(__nv_bfloat16);
        size_t bytes_fp32 = n * sizeof(float);

        HPC_CUDA_CHECK(cudaMalloc(&d_params_bf16[i], bytes_bf16));
        HPC_CUDA_CHECK(cudaMalloc(&d_master_fp32[i], bytes_fp32));
        HPC_CUDA_CHECK(cudaMalloc(&d_grads_bf16[i],  bytes_bf16));

        // Xavier-like init: fill master FP32 then cast to BF16
        HPC_CUDA_CHECK(cudaMemsetAsync(d_master_fp32[i], 0, bytes_fp32, stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_params_bf16[i], 0, bytes_bf16, stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_grads_bf16[i],  0, bytes_bf16, stream));

        params[i] = TensorView(d_params_bf16[i], n);
        grads[i]  = TensorView(d_grads_bf16[i],  n);
    }

    // ---- Build AdamW optimizer (BF16 path with FP32 master) ----
    AdamConfig cfg = make_adamw_config(BASE_LR, WEIGHT_DECAY);
    cfg.beta1 = 0.9f; cfg.beta2 = 0.95f; cfg.eps = 1e-8f;  // GPT-style

    HPCOptimizer<AdamOptimizer, AdamConfig> opt(cfg, stream, &comm, d_master_fp32);
    opt.enable_clipping(MAX_GRAD_NORM, /*fp16_comm=*/true);
    opt.init(params, N_TENSORS);

    // Attach WarmupCosine scheduler
    opt.set_scheduler(std::make_unique<WarmupCosineLR>(
        opt.config().lr, WARMUP_STEPS, TOTAL_STEPS, /*eta_min=*/1e-5f));

    // Periodic checkpoint
    opt.set_checkpoint_dir("/tmp/hpc_opt_ckpt", CKPT_EVERY);

    // ---- Profiling ----
    StepTimer       timer;
    StepTimer       ar_timer;
    IterationProfiler profiler(/*window=*/50);
    ThroughputLogger  tput(world, BATCH_SIZE, SEQ_LEN);

    // ---- Barrier: all ranks ready ----
    comm.barrier(stream);
    HPC_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (rank == 0)
        printf("%-8s  %-10s  %-12s  %-12s  %-10s\n",
               "Step", "LR", "GradNorm(pre)", "GradNorm(post)", "ms/step");

    // ================================================================
    // Training loop
    // ================================================================
    for (int step = 1; step <= TOTAL_STEPS; ++step) {
        HPC_RANGE_PUSH("train_step", 0xFF2196F3);
        timer.start(stream);

        // ---- Simulate backward pass (fill BF16 gradients) ----
        {
            HPC_RANGE_PUSH("backward", 0xFFE91E63);
            // Spike gradients at step 100 to verify clipping
            float grad_scale = (step == 100) ? 20.0f : 0.3f;
            for (int t = 0; t < N_TENSORS; ++t) {
                size_t n  = TENSOR_SIZES[t];
                int    blk = hpc_blocks(n, 256);
                synthetic_backward_bf16<<<blk, 256, 0, stream>>>(
                    d_grads_bf16[t], n, grad_scale,
                    (uint32_t)(step * 7919 + t * 31));
            }
            HPC_RANGE_POP();
        }

        // ---- NCCL all-reduce + clip + optimizer step ----
        // (opt.step() handles all-reduce then clip then kernel)
        opt.step(params, grads, N_TENSORS);

        float step_ms = timer.stop(stream);
        profiler.record(step_ms);

        HPC_RANGE_POP();  // train_step

        // ---- Logging ----
        if (rank == 0 && (step % LOG_EVERY == 0 || step == 1)) {
            const auto& st = opt.stats();
            printf("%-8d  %-10.2e  %-12.4f  %-12.4f  %-10.2f\n",
                   step, opt.current_lr(),
                   st.grad_norm_before,
                   st.grad_norm_after,
                   step_ms);
            tput.log(step_ms, step);
        }
    }

    // ---- Final summary ----
    HPC_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (rank == 0) {
        profiler.print_summary(TOTAL_STEPS, world);
        printf("\nTraining complete. Saving final checkpoint...\n");
        opt.save_checkpoint("/tmp/hpc_opt_ckpt/final.bin");
    }

    // ---- Cleanup ----
    for (int i = 0; i < N_TENSORS; ++i) {
        cudaFree(d_params_bf16[i]);
        cudaFree(d_master_fp32[i]);
        cudaFree(d_grads_bf16[i]);
    }

    return 0;
}
