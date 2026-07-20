// =============================================================================
// HPC CUDA Optimizer Library — Tensor Parallel Extension
// hpc_tensor_parallel.cuh  —  Megatron-LM-style Column/Row Parallel Linear
//
// Reference:
//   Shoeybi et al. "Megatron-LM: Training Multi-Billion Parameter Language
//   Models Using Model Parallelism." arXiv:1909.08053 (2019)
//
// Architecture:
//   TP group: a set of T ranks that cooperate on the same layer.
//   Each rank holds a weight SHARD; NCCL handles inter-rank communication.
//
//   ColumnParallelLinear  —  split output dimension:
//     Weight shape: [in_features, out_features / T]   (local)
//     Forward:      Y_local = X @ W_col               (no comm needed if chained)
//                   Y_full  = AllGather(Y_local)       (if output is not RowParallel)
//     Backward:     dX      = ReduceScatter(dY @ W_col^T)
//                   dW_col  = X^T @ dY
//
//   RowParallelLinear  —  split input dimension:
//     Weight shape: [in_features / T, out_features]   (local)
//     Forward:      Y_local = X_shard @ W_row
//                   Y       = AllReduce(Y_local)       (sum partial products)
//     Backward:     dX_shard = dY @ W_row^T            (no comm, each rank has own)
//                   dW_row   = X_shard^T @ dY
//
//   Typical transformer FFN wiring (T=4 ranks):
//     x [B,S,H]  →  ColumnParallel(H, 4H/T, gather=false)
//               →  [B,S,4H/T]  GELU  →  RowParallel(4H/T, H, allreduce=true)
//               →  [B,S,H]  (fully reduced)
//
//   Typical transformer Attention wiring:
//     QKV proj:   ColumnParallel(H, 3*H/T, gather=false)
//     Out proj:   RowParallel(H/T, H, allreduce=true)
//
// GEMM back-end: cuBLAS
//   FP32: cublasSgemm
//   FP16: cublasHgemm
//   BF16: cublasGemmEx with CUDA_R_16BF
//
// Build guards:
//   HPC_HAVE_NCCL  — enables all-reduce / all-gather
//   HPC_HAVE_CUBLAS — enables cuBLAS GEMM (falls back to custom GEMM otherwise)
//
// Design notes:
//   • Weight tensors are kept on-device in column-major cuBLAS layout
//   • Activations are assumed row-major [batch*seq, features]
//   • Both FP16 and BF16 routes use CUBLAS_COMPUTE_32F_FAST_16F for stability
//   • Bias (optional): held only by rank-0 for column-parallel;
//     all ranks hold equal bias for row-parallel (added post-allreduce)
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_comm.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <stdexcept>

#ifdef HPC_HAVE_CUBLAS
#  include <cublas_v2.h>
#  define CUBLAS_CHECK(x) do {                                          \
       cublasStatus_t _s = (x);                                         \
       if (_s != CUBLAS_STATUS_SUCCESS) {                               \
           fprintf(stderr, "cuBLAS error %d at %s:%d\n",               \
                   (int)_s, __FILE__, __LINE__);                        \
           std::abort();                                                 \
       } } while(0)
#endif

#ifdef HPC_HAVE_NCCL
#  include <nccl.h>
#endif

namespace hpc_opt {
namespace tp {

// ===========================================================================
// Utility: warp-level reduction (used in custom GEMM fallback bias add)
// ===========================================================================
__inline__ __device__ float warp_reduce_sum_tp(float v) {
    for (int d = 16; d >= 1; d >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
}

// ===========================================================================
// k_bias_add  —  add bias vector b[out_features] to matrix Y[rows, out_features]
// ===========================================================================
template<typename T>
__global__ void k_bias_add(T* __restrict__ Y, const T* __restrict__ b,
                           size_t rows, size_t cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y;
    if (col < cols && row < rows)
        Y[row * cols + col] += b[col];
}

// ===========================================================================
// k_allreduce_inplace  —  single-block Kahan-safe float4 in-place scale
//   Used post-allreduce to apply 1/N if using sum semantics.
// ===========================================================================
__global__ void k_inplace_scale_tp(float* __restrict__ x, float scale, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        x[i] *= scale;
}

// ===========================================================================
// TPContext  —  cuBLAS handle + NCCL communicator for a TP group
// ===========================================================================
struct TPContext {
    int  tp_rank   = 0;
    int  tp_size   = 1;

#ifdef HPC_HAVE_CUBLAS
    cublasHandle_t cublas = nullptr;
#endif

#ifdef HPC_HAVE_NCCL
    ncclComm_t nccl_tp = nullptr;
#endif

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream    = nullptr;

    cudaEvent_t ev_comm_done    = nullptr;
    cudaEvent_t ev_compute_done = nullptr;

    static TPContext from_comm(CommContext& comm,
                               cudaStream_t cs, cudaStream_t ws = nullptr)
    {
        TPContext ctx;
        ctx.tp_rank         = comm.rank();
        ctx.tp_size         = comm.world_size();
        ctx.compute_stream  = cs;
        ctx.comm_stream     = ws ? ws : cs;

#ifdef HPC_HAVE_CUBLAS
        CUBLAS_CHECK(cublasCreate(&ctx.cublas));
        CUBLAS_CHECK(cublasSetStream(ctx.cublas, cs));
        CUBLAS_CHECK(cublasSetMathMode(ctx.cublas,
                                       CUBLAS_DEFAULT_MATH));
#endif

        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ev_comm_done,
                                                cudaEventDisableTiming));
        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ev_compute_done,
                                                cudaEventDisableTiming));
        return ctx;
    }

    void destroy() {
#ifdef HPC_HAVE_CUBLAS
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
#endif
        if (ev_comm_done)    cudaEventDestroy(ev_comm_done);
        if (ev_compute_done) cudaEventDestroy(ev_compute_done);
    }

    bool is_root() const { return tp_rank == 0; }
};

// ===========================================================================
// Internal GEMM dispatch: Y = alpha * A * B + beta * C
//   A: [M, K]   B: [K, N]   Y: [M, N]  (row-major)
//
// cuBLAS is column-major, so we pass transposed dimensions:
//   C^T = (A*B)^T = B^T * A^T
//   i.e., cublas sees: Y^T[N,M] = B^T[N,K] * A^T[K,M]
// ===========================================================================
inline void gemm_dispatch(
        const TPContext& ctx,
        Dtype dtype,
        int M, int K, int N,
        const void* A,    // [M, K] row-major
        const void* B,    // [K, N] row-major
        void*       C,    // [M, N] row-major
        float alpha = 1.0f, float beta = 0.0f,
        cudaStream_t stream = nullptr)
{
#ifdef HPC_HAVE_CUBLAS
    cublasHandle_t h = ctx.cublas;
    if (stream) CUBLAS_CHECK(cublasSetStream(h, stream));

    if (dtype == Dtype::FP32) {
        CUBLAS_CHECK(cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            static_cast<const float*>(B), N,
            static_cast<const float*>(A), K,
            &beta,
            static_cast<float*>(C), N));
    } else {
        // FP16 and BF16 use cublasGemmEx with COMPUTE_32F for stability
        cudaDataType_t dt = (dtype == Dtype::FP16) ? CUDA_R_16F : CUDA_R_16BF;
        CUBLAS_CHECK(cublasGemmEx(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, dt, N,
            A, dt, K,
            &beta,
            C, dt, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#else
    // Fallback: naive reference kernel (not for production — install cuBLAS!)
    // This kernel is intentionally left as a compile-time stub.
    (void)ctx; (void)dtype; (void)M; (void)K; (void)N;
    (void)A; (void)B; (void)C; (void)alpha; (void)beta; (void)stream;
    fprintf(stderr, "[TP] cuBLAS not available — link with -lcublas\n");
    std::abort();
#endif
}

// ===========================================================================
// All-reduce helper (sum)
// ===========================================================================
inline void allreduce_sum(const TPContext& ctx, void* buf, size_t numel,
                          Dtype dtype, cudaStream_t stream)
{
#ifdef HPC_HAVE_NCCL
    if (ctx.tp_size <= 1) return;
    ncclDataType_t ndt = (dtype == Dtype::FP32)  ? ncclFloat :
                         (dtype == Dtype::FP16)  ? ncclHalf  :
                                                   ncclBfloat16;
    NCCL_CHECK(ncclAllReduce(buf, buf, numel, ndt, ncclSum,
                             ctx.nccl_tp, stream));
#else
    (void)ctx; (void)buf; (void)numel; (void)dtype; (void)stream;
#endif
}

// ===========================================================================
// All-gather helper: each rank sends numel_per_rank elements, receives full
// ===========================================================================
inline void allgather_tp(const TPContext& ctx,
                         const void* send, void* recv,
                         size_t numel_per_rank,
                         Dtype dtype, cudaStream_t stream)
{
#ifdef HPC_HAVE_NCCL
    if (ctx.tp_size <= 1) {
        if (send != recv)
            HPC_CUDA_CHECK(cudaMemcpyAsync(recv, send,
                           numel_per_rank * (dtype==Dtype::FP32?4:2),
                           cudaMemcpyDeviceToDevice, stream));
        return;
    }
    ncclDataType_t ndt = (dtype == Dtype::FP32)  ? ncclFloat :
                         (dtype == Dtype::FP16)  ? ncclHalf  :
                                                   ncclBfloat16;
    NCCL_CHECK(ncclAllGather(send, recv, numel_per_rank, ndt,
                             ctx.nccl_tp, stream));
#else
    if (send != recv)
        HPC_CUDA_CHECK(cudaMemcpyAsync(recv, send,
                       numel_per_rank * (dtype==Dtype::FP32?4:2),
                       cudaMemcpyDeviceToDevice, stream));
    (void)ctx; (void)dtype; (void)stream;
#endif
}

// ===========================================================================
// ColumnParallelLinear
//
//   Weight is sharded along the OUTPUT dimension:
//     Full weight: [in_features, out_features]
//     Local shard: [in_features, out_features / T]  stored column-major for cuBLAS
//
//   Forward (Y = X W + b):
//     Y_local = X [batch, in_features] @ W_col [in_features, local_out_features]
//     if gather_output:
//         Y_full [batch, out_features] = AllGather(Y_local) along T-dim
//     else:
//         return Y_local  (feed directly into RowParallelLinear)
//
//   Bias:
//     If bias: only rank 0 adds bias AFTER AllGather (or into Y_local if no gather)
//     Alternatively: each rank holds bias_shard and adds before gather.
//     We follow Megatron convention: no-gather path → each rank adds bias_shard.
//     Gather path → rank 0 adds full bias after AllGather.
// ===========================================================================
struct ColumnParallelLinear {
    int     in_features      = 0;
    int     out_features      = 0;   // TOTAL (across all TP ranks)
    int     local_out         = 0;   // out_features / T
    int     rank              = 0;
    int     tp_size           = 1;
    bool    gather_output     = true; // AllGather output to full width?
    bool    has_bias          = true;
    Dtype   dtype             = Dtype::FP16;

    // Device pointers
    void*   d_weight          = nullptr;  // [in_features, local_out] — this rank's shard
    void*   d_bias_shard      = nullptr;  // [local_out] — shard (or null)
    void*   d_output_scratch  = nullptr;  // [batch*seq, local_out] temp output
    void*   d_full_output     = nullptr;  // [batch*seq, out_features] gathered output

    size_t  max_tokens        = 0;        // max batch*seq seen — for scratch sizing

    // -----------------------------------------------------------------------
    // init: allocate weight shard
    //   weight_init_fn: user callback to fill d_weight with initial values
    // -----------------------------------------------------------------------
    void init(int in_f, int out_f, int tp_rank, int tp_sz,
              bool gather, bool bias, Dtype dt, size_t max_tok = 4096)
    {
        in_features   = in_f;
        out_features  = out_f;
        local_out     = out_f / tp_sz;
        assert(out_f % tp_sz == 0 && "out_features must be divisible by tp_size");
        rank          = tp_rank;
        tp_size       = tp_sz;
        gather_output = gather;
        has_bias      = bias;
        dtype         = dt;
        max_tokens    = max_tok;

        size_t elem_bytes = (dt == Dtype::FP32) ? 4 : 2;
        size_t W_bytes = (size_t)in_features * local_out * elem_bytes;
        size_t B_bytes = (size_t)local_out * elem_bytes;

        HPC_CUDA_CHECK(cudaMalloc(&d_weight, W_bytes));
        HPC_CUDA_CHECK(cudaMemset(d_weight, 0, W_bytes));

        if (has_bias) {
            HPC_CUDA_CHECK(cudaMalloc(&d_bias_shard, B_bytes));
            HPC_CUDA_CHECK(cudaMemset(d_bias_shard, 0, B_bytes));
        }

        // Scratch: local output
        HPC_CUDA_CHECK(cudaMalloc(&d_output_scratch,
                       max_tok * (size_t)local_out * elem_bytes));
        // Full output (only if gathering)
        if (gather_output)
            HPC_CUDA_CHECK(cudaMalloc(&d_full_output,
                           max_tok * (size_t)out_features * elem_bytes));
    }

    // -----------------------------------------------------------------------
    // forward: Y = X W_col, optionally AllGather
    //   X:         [tokens, in_features]
    //   output:    [tokens, out_features]  (if gather_output)
    //              [tokens, local_out]     (if !gather_output)
    //   Returns pointer to output buffer.
    // -----------------------------------------------------------------------
    void* forward(const TPContext& ctx,
                  const void* X, int tokens,
                  cudaStream_t stream)
    {
        assert(tokens <= (int)max_tokens && "increase max_tokens");

        // Y_local = X @ W_col
        // Shapes: X[tokens, in_f] @ W_col[in_f, local_out] = Y[tokens, local_out]
        gemm_dispatch(ctx, dtype,
                      tokens, in_features, local_out,
                      X, d_weight, d_output_scratch,
                      1.0f, 0.0f, stream);

        // Add bias shard
        if (has_bias && d_bias_shard) {
            dim3 grid((local_out + HPC_BLOCK - 1) / HPC_BLOCK, tokens);
            if (dtype == Dtype::FP32) {
                k_bias_add<float><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<float*>(d_output_scratch),
                    static_cast<const float*>(d_bias_shard),
                    tokens, local_out);
            } else if (dtype == Dtype::FP16) {
                k_bias_add<half><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<half*>(d_output_scratch),
                    static_cast<const half*>(d_bias_shard),
                    tokens, local_out);
            } else {
                k_bias_add<__nv_bfloat16><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat16*>(d_output_scratch),
                    static_cast<const __nv_bfloat16*>(d_bias_shard),
                    tokens, local_out);
            }
        }

        if (!gather_output) return d_output_scratch;

        // AllGather Y_local → Y_full along TP dimension
        // Each rank's Y_local is tokens*local_out elems;
        // AllGather produces tokens*out_features = tokens*local_out*T
        allgather_tp(ctx, d_output_scratch, d_full_output,
                     (size_t)tokens * local_out, dtype, stream);

        return d_full_output;
    }

    void destroy() {
        if (d_weight)         { cudaFree(d_weight);         d_weight         = nullptr; }
        if (d_bias_shard)     { cudaFree(d_bias_shard);     d_bias_shard     = nullptr; }
        if (d_output_scratch) { cudaFree(d_output_scratch); d_output_scratch = nullptr; }
        if (d_full_output)    { cudaFree(d_full_output);    d_full_output    = nullptr; }
    }

    ~ColumnParallelLinear() { destroy(); }
    ColumnParallelLinear()  = default;
    ColumnParallelLinear(const ColumnParallelLinear&) = delete;
    ColumnParallelLinear& operator=(const ColumnParallelLinear&) = delete;

    // Weight size in bytes (shard only)
    size_t weight_bytes() const {
        return (size_t)in_features * local_out *
               (dtype == Dtype::FP32 ? 4 : 2);
    }

    // Total weight size across all ranks
    size_t full_weight_bytes() const {
        return (size_t)in_features * out_features *
               (dtype == Dtype::FP32 ? 4 : 2);
    }
};

// ===========================================================================
// RowParallelLinear
//
//   Weight is sharded along the INPUT dimension:
//     Full weight: [in_features, out_features]
//     Local shard: [in_features / T, out_features]
//
//   Forward:
//     Input X is assumed to be pre-sharded along input dim (from ColParallel)
//     Y_local = X_shard [batch, in_features/T] @ W_row [in_features/T, out_features]
//     Y_full  = AllReduce(Y_local)   — sum across T ranks
//     if has_bias: Y_full += bias    (all ranks hold identical bias, add on one rank
//                                     and allreduce takes care of broadcast; OR
//                                     add AFTER allreduce on each rank for zero overhead)
//     Megatron-LM choice: add bias AFTER AllReduce on ALL ranks.
// ===========================================================================
struct RowParallelLinear {
    int     in_features   = 0;    // TOTAL in_features
    int     local_in      = 0;    // in_features / T
    int     out_features  = 0;
    int     rank          = 0;
    int     tp_size       = 1;
    bool    has_bias      = true;
    Dtype   dtype         = Dtype::FP16;

    void*   d_weight      = nullptr;  // [local_in, out_features]
    void*   d_bias        = nullptr;  // [out_features] — ALL ranks identical
    void*   d_output      = nullptr;  // [tokens, out_features] — partial + reduced

    size_t  max_tokens    = 0;

    void init(int in_f, int out_f, int tp_rank, int tp_sz,
              bool bias, Dtype dt, size_t max_tok = 4096)
    {
        in_features  = in_f;
        local_in     = in_f / tp_sz;
        assert(in_f % tp_sz == 0 && "in_features must be divisible by tp_size");
        out_features = out_f;
        rank         = tp_rank;
        tp_size      = tp_sz;
        has_bias     = bias;
        dtype        = dt;
        max_tokens   = max_tok;

        size_t elem_bytes = (dt == Dtype::FP32) ? 4 : 2;
        size_t W_bytes = (size_t)local_in * out_features * elem_bytes;
        size_t B_bytes = (size_t)out_features * elem_bytes;

        HPC_CUDA_CHECK(cudaMalloc(&d_weight, W_bytes));
        HPC_CUDA_CHECK(cudaMemset(d_weight, 0, W_bytes));

        if (has_bias) {
            HPC_CUDA_CHECK(cudaMalloc(&d_bias, B_bytes));
            HPC_CUDA_CHECK(cudaMemset(d_bias, 0, B_bytes));
        }

        HPC_CUDA_CHECK(cudaMalloc(&d_output,
                       max_tok * (size_t)out_features * elem_bytes));
    }

    // -----------------------------------------------------------------------
    // forward: Y = AllReduce(X_shard @ W_row) [+ bias]
    //   X_shard: [tokens, local_in]  — already sharded
    //   Returns pointer to d_output [tokens, out_features] fully reduced
    // -----------------------------------------------------------------------
    void* forward(const TPContext& ctx,
                  const void* X_shard, int tokens,
                  cudaStream_t stream)
    {
        assert(tokens <= (int)max_tokens);

        // Y_local = X_shard [tokens, local_in] @ W_row [local_in, out_features]
        gemm_dispatch(ctx, dtype,
                      tokens, local_in, out_features,
                      X_shard, d_weight, d_output,
                      1.0f, 0.0f, stream);

        // AllReduce across TP group (sum partial products)
        allreduce_sum(ctx, d_output,
                      (size_t)tokens * out_features,
                      dtype, stream);

        // Add bias (after reduce, on every rank — identical bias)
        if (has_bias && d_bias) {
            dim3 grid((out_features + HPC_BLOCK - 1) / HPC_BLOCK, tokens);
            if (dtype == Dtype::FP32) {
                k_bias_add<float><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<float*>(d_output),
                    static_cast<const float*>(d_bias),
                    tokens, out_features);
            } else if (dtype == Dtype::FP16) {
                k_bias_add<half><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<half*>(d_output),
                    static_cast<const half*>(d_bias),
                    tokens, out_features);
            } else {
                k_bias_add<__nv_bfloat16><<<grid, HPC_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat16*>(d_output),
                    static_cast<const __nv_bfloat16*>(d_bias),
                    tokens, out_features);
            }
        }

        return d_output;
    }

    void destroy() {
        if (d_weight) { cudaFree(d_weight); d_weight = nullptr; }
        if (d_bias)   { cudaFree(d_bias);   d_bias   = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
    }

    ~RowParallelLinear() { destroy(); }
    RowParallelLinear()  = default;
    RowParallelLinear(const RowParallelLinear&) = delete;
    RowParallelLinear& operator=(const RowParallelLinear&) = delete;

    size_t weight_bytes() const {
        return (size_t)local_in * out_features *
               (dtype == Dtype::FP32 ? 4 : 2);
    }
};

// ===========================================================================
// VocabParallelEmbedding
//   Vocabulary split across T ranks:
//     Full vocab: V tokens, embedding dim H
//     Local shard: [V/T, H]
//
//   Forward lookup:
//     If input token is in [tp_rank*V/T, (tp_rank+1)*V/T) → use local embedding
//     Otherwise → zero contribution (AllReduce sums across ranks)
//   AllReduce produces correct embedding for every token on every rank.
// ===========================================================================
template<typename T>
__global__ void k_vocab_embed_lookup(
        const T*       __restrict__ embed,   // [local_V, H]
        const int32_t* __restrict__ tokens,  // [batch_seq]
        T*             __restrict__ out,     // [batch_seq, H]
        int            H,
        int            vocab_lo,             // tp_rank * local_V
        int            vocab_hi,             // (tp_rank+1) * local_V
        int            n_tokens)
{
    int tok_idx = blockIdx.x;
    if (tok_idx >= n_tokens) return;

    int tok = tokens[tok_idx];

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        if (tok >= vocab_lo && tok < vocab_hi) {
            out[tok_idx * H + h] = embed[(tok - vocab_lo) * H + h];
        } else {
            if constexpr (std::is_same_v<T, float>)
                out[tok_idx * H + h] = 0.0f;
            else
                out[tok_idx * H + h] = T(0);
        }
    }
}

struct VocabParallelEmbedding {
    int    vocab_size   = 0;
    int    local_vocab  = 0;
    int    hidden_dim   = 0;
    int    rank         = 0;
    int    tp_size      = 1;
    Dtype  dtype        = Dtype::FP16;

    void*  d_embed      = nullptr;  // [local_vocab, hidden_dim]
    void*  d_out        = nullptr;  // [max_tokens, hidden_dim]
    size_t max_tokens   = 0;

    void init(int V, int H, int tp_rank, int tp_sz, Dtype dt, size_t max_tok = 4096) {
        vocab_size  = V;
        local_vocab = V / tp_sz;
        assert(V % tp_sz == 0 && "vocab_size must be divisible by tp_size");
        hidden_dim  = H;
        rank        = tp_rank;
        tp_size     = tp_sz;
        dtype       = dt;
        max_tokens  = max_tok;

        size_t elem_bytes = (dt == Dtype::FP32) ? 4 : 2;
        HPC_CUDA_CHECK(cudaMalloc(&d_embed,
                       (size_t)local_vocab * H * elem_bytes));
        HPC_CUDA_CHECK(cudaMalloc(&d_out,
                       max_tok * (size_t)H * elem_bytes));
        HPC_CUDA_CHECK(cudaMemset(d_embed, 0, (size_t)local_vocab * H * elem_bytes));
    }

    void* forward(const TPContext& ctx,
                  const int32_t* d_token_ids, int n_tokens,
                  cudaStream_t stream)
    {
        assert(n_tokens <= (int)max_tokens);
        int vocab_lo = rank * local_vocab;
        int vocab_hi = vocab_lo + local_vocab;

        if (dtype == Dtype::FP32)
            k_vocab_embed_lookup<float><<<n_tokens, HPC_BLOCK, 0, stream>>>(
                static_cast<const float*>(d_embed), d_token_ids,
                static_cast<float*>(d_out), hidden_dim,
                vocab_lo, vocab_hi, n_tokens);
        else if (dtype == Dtype::FP16)
            k_vocab_embed_lookup<half><<<n_tokens, HPC_BLOCK, 0, stream>>>(
                static_cast<const half*>(d_embed), d_token_ids,
                static_cast<half*>(d_out), hidden_dim,
                vocab_lo, vocab_hi, n_tokens);
        else
            k_vocab_embed_lookup<__nv_bfloat16><<<n_tokens, HPC_BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(d_embed), d_token_ids,
                static_cast<__nv_bfloat16*>(d_out), hidden_dim,
                vocab_lo, vocab_hi, n_tokens);

        // AllReduce: contributions from all TP ranks (sum gives correct embedding)
        allreduce_sum(ctx, d_out, (size_t)n_tokens * hidden_dim, dtype, stream);

        return d_out;
    }

    void destroy() {
        if (d_embed) { cudaFree(d_embed); d_embed = nullptr; }
        if (d_out)   { cudaFree(d_out);   d_out   = nullptr; }
    }

    ~VocabParallelEmbedding() { destroy(); }
    VocabParallelEmbedding()  = default;
    VocabParallelEmbedding(const VocabParallelEmbedding&) = delete;
    VocabParallelEmbedding& operator=(const VocabParallelEmbedding&) = delete;
};

// ===========================================================================
// SequenceParallelLayerNorm
//   In sequence-parallel mode, each rank holds [S/T, H] of the sequence.
//   LayerNorm is applied locally; the mean and variance are computed
//   over the local chunk and no cross-rank sync is needed (each rank's
//   tokens are independent).
//   This works because LayerNorm is per-token (over H), not over S.
// ===========================================================================
template<typename T>
__global__ void k_layernorm_fwd(
        T*           __restrict__ out,
        const T*     __restrict__ inp,
        const float* __restrict__ weight,    // gamma [H]
        const float* __restrict__ bias_ln,   // beta  [H] — named bias_ln to avoid shadowing
        int H, float eps)
{
    int row = blockIdx.x;
    const T* x = inp + row * H;
    T*       y = out + row * H;

    // Compute mean and variance in float for numerical stability
    float mean = 0.0f, var = 0.0f;

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float xf = to_float(x[h]);
        mean += xf;
    }
    for (int d = 16; d >= 1; d >>= 1)
        mean += __shfl_xor_sync(0xffffffff, mean, d);
    mean /= H;

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float xf = to_float(x[h]) - mean;
        var += xf * xf;
    }
    for (int d = 16; d >= 1; d >>= 1)
        var += __shfl_xor_sync(0xffffffff, var, d);
    var /= H;

    float inv_std = rsqrtf(var + eps);

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float xn = (to_float(x[h]) - mean) * inv_std;
        y[h] = from_float<T>(xn * weight[h] + bias_ln[h]);
    }
}

struct SequenceParallelLayerNorm {
    int     hidden_dim = 0;
    Dtype   dtype      = Dtype::FP16;
    float   eps        = 1e-5f;

    float*  d_gamma    = nullptr;  // [H] — all ranks identical (replicated)
    float*  d_beta     = nullptr;  // [H]
    void*   d_out      = nullptr;

    size_t  max_tokens = 0;

    void init(int H, Dtype dt, float epsilon, size_t max_tok) {
        hidden_dim = H;
        dtype      = dt;
        eps        = epsilon;
        max_tokens = max_tok;

        size_t elem_bytes = (dt == Dtype::FP32) ? 4 : 2;
        HPC_CUDA_CHECK(cudaMalloc(&d_gamma, H * sizeof(float)));
        HPC_CUDA_CHECK(cudaMalloc(&d_beta,  H * sizeof(float)));
        HPC_CUDA_CHECK(cudaMalloc(&d_out,   max_tok * (size_t)H * elem_bytes));

        // Default: gamma=1, beta=0
        std::vector<float> ones(H, 1.0f), zeros(H, 0.0f);
        HPC_CUDA_CHECK(cudaMemcpy(d_gamma, ones.data(),  H*sizeof(float), cudaMemcpyHostToDevice));
        HPC_CUDA_CHECK(cudaMemcpy(d_beta,  zeros.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    }

    void* forward(const void* inp, int n_tokens, cudaStream_t stream) {
        int block = std::min(hidden_dim, 1024);
        if (dtype == Dtype::FP32)
            k_layernorm_fwd<float><<<n_tokens, block, 0, stream>>>(
                static_cast<float*>(d_out),
                static_cast<const float*>(inp),
                d_gamma, d_beta, hidden_dim, eps);
        else if (dtype == Dtype::FP16)
            k_layernorm_fwd<half><<<n_tokens, block, 0, stream>>>(
                static_cast<half*>(d_out),
                static_cast<const half*>(inp),
                d_gamma, d_beta, hidden_dim, eps);
        else
            k_layernorm_fwd<__nv_bfloat16><<<n_tokens, block, 0, stream>>>(
                static_cast<__nv_bfloat16*>(d_out),
                static_cast<const __nv_bfloat16*>(inp),
                d_gamma, d_beta, hidden_dim, eps);
        return d_out;
    }

    void destroy() {
        if (d_gamma) { cudaFree(d_gamma); d_gamma = nullptr; }
        if (d_beta)  { cudaFree(d_beta);  d_beta  = nullptr; }
        if (d_out)   { cudaFree(d_out);   d_out   = nullptr; }
    }

    ~SequenceParallelLayerNorm() { destroy(); }
    SequenceParallelLayerNorm()  = default;
    SequenceParallelLayerNorm(const SequenceParallelLayerNorm&) = delete;
    SequenceParallelLayerNorm& operator=(const SequenceParallelLayerNorm&) = delete;
};

// ===========================================================================
// TPTransformerFFN  —  convenience wrapper for a full MLP block
//   Composes: ColParallel → GELU → RowParallel
//   This is the standard Megatron-LM FFN construction.
// ===========================================================================
template<typename T>
__global__ void k_gelu_inplace(T* __restrict__ x, size_t n) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < n;
         i += gridDim.x*blockDim.x) {
        float v = to_float(x[i]);
        // GELU approximation: 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715*x³)))
        float c = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[i] = from_float<T>(0.5f * v * (1.0f + tanhf(c)));
    }
}

struct TPTransformerFFN {
    ColumnParallelLinear fc1;
    RowParallelLinear    fc2;

    int     hidden_dim   = 0;
    int     ffn_dim      = 0;  // 4 * hidden in standard transformer
    int     tp_size      = 1;
    Dtype   dtype        = Dtype::FP16;

    void init(int H, int ffn, int tp_rank, int tp_sz, Dtype dt,
              size_t max_tok = 4096, bool bias = true)
    {
        hidden_dim = H;
        ffn_dim    = ffn;
        tp_size    = tp_sz;
        dtype      = dt;

        // fc1: [H, ffn/T]  — column-parallel, NO gather (feeds into fc2 directly)
        fc1.init(H, ffn, tp_rank, tp_sz, /*gather=*/false, bias, dt, max_tok);

        // fc2: [ffn/T, H]  — row-parallel, with AllReduce
        fc2.init(ffn, H, tp_rank, tp_sz, bias, dt, max_tok);
    }

    // -----------------------------------------------------------------------
    // forward:  Y = RowParallel(GELU(ColParallel(X)))
    //   X:  [tokens, H]  (full hidden dim on each rank — replicated or from LayerNorm)
    //   out: [tokens, H] (full output — allreduced)
    // -----------------------------------------------------------------------
    void* forward(const TPContext& ctx,
                  const void* X, int tokens,
                  cudaStream_t stream)
    {
        // fc1: X [T,H] @ W_col [H, ffn/T] = Y1 [T, ffn/T]
        void* y1 = fc1.forward(ctx, X, tokens, stream);

        // GELU in-place on y1 [tokens, ffn/T]
        size_t n = (size_t)tokens * fc1.local_out;
        if (dtype == Dtype::FP32)
            k_gelu_inplace<float><<<hpc_blocks(n), HPC_BLOCK, 0, stream>>>(
                static_cast<float*>(y1), n);
        else if (dtype == Dtype::FP16)
            k_gelu_inplace<half><<<hpc_blocks(n), HPC_BLOCK, 0, stream>>>(
                static_cast<half*>(y1), n);
        else
            k_gelu_inplace<__nv_bfloat16><<<hpc_blocks(n), HPC_BLOCK, 0, stream>>>(
                static_cast<__nv_bfloat16*>(y1), n);

        // fc2: y1 [T, ffn/T] @ W_row [ffn/T, H] → allreduce → out [T, H]
        return fc2.forward(ctx, y1, tokens, stream);
    }

    void print_config() const {
        printf("[TP-FFN] H=%d  FFN=%d  tp_size=%d  "
               "local_ffn=%d  dtype=%s\n",
               hidden_dim, ffn_dim, tp_size, fc1.local_out,
               (dtype==Dtype::FP32 ? "fp32" :
                dtype==Dtype::FP16 ? "fp16" : "bf16"));
        double shard_gb = (double)(fc1.weight_bytes() + fc2.weight_bytes()) / 1e9;
        double full_gb  = (double)(fc1.full_weight_bytes() + fc2.weight_bytes() * tp_size) / 1e9;
        printf("[TP-FFN] Weight per rank: %.3f GB  (full: %.3f GB)  %.1fx less\n",
               shard_gb, full_gb, full_gb / shard_gb);
    }
};

// ===========================================================================
// TPAttentionQKV  —  convenience for attention QKV + output projections
//   Q, K, V projections: ColumnParallel(H, H/T) per head-shard
//   Output projection:   RowParallel(H/T, H) with AllReduce
// ===========================================================================
struct TPAttentionQKV {
    ColumnParallelLinear qkv_proj;  // [H, 3*H/T] — all three stacked
    RowParallelLinear    out_proj;  // [H/T, H]

    int    hidden_dim  = 0;
    int    n_heads     = 0;
    int    local_heads = 0;
    Dtype  dtype       = Dtype::FP16;

    void init(int H, int heads, int tp_rank, int tp_sz, Dtype dt,
              size_t max_tok = 4096, bool bias = true)
    {
        assert(heads % tp_sz == 0 && "n_heads must be divisible by tp_size");
        hidden_dim  = H;
        n_heads     = heads;
        local_heads = heads / tp_sz;
        dtype       = dt;

        // QKV: stacked 3 projections, each [H, H/T] → together [H, 3H/T]
        qkv_proj.init(H, 3*H, tp_rank, tp_sz, /*gather=*/false, bias, dt, max_tok);

        // Output: H/T → H
        out_proj.init(H, H, tp_rank, tp_sz, bias, dt, max_tok);
    }

    // QKV forward: X [tokens, H] → packed [tokens, 3*H/T]
    // The QKV tensor can be split into Q, K, V each of [tokens, H/T]
    void* qkv_forward(const TPContext& ctx,
                      const void* X, int tokens, cudaStream_t stream)
    {
        return qkv_proj.forward(ctx, X, tokens, stream);
    }

    // Output proj: attn_out [tokens, H/T] → Y [tokens, H] (allreduced)
    void* out_forward(const TPContext& ctx,
                      const void* attn_out, int tokens, cudaStream_t stream)
    {
        return out_proj.forward(ctx, attn_out, tokens, stream);
    }
};

// ===========================================================================
// memory_report_tp  —  print TP memory stats to stdout
// ===========================================================================
inline void memory_report_tp(const char* label,
                              int tp_size,
                              size_t total_params,
                              Dtype  dtype = Dtype::BF16)
{
    size_t elem_bytes = (dtype == Dtype::FP32) ? 4 : 2;
    double full_gb    = (double)(total_params * elem_bytes) / 1e9;
    double shard_gb   = full_gb / tp_size;

    printf("[TP] %s\n"
           "     tp_size=%d  params=%.1fM  dtype=%s\n"
           "     Full-model weight:  %.3f GB\n"
           "     Per-rank weight:    %.3f GB  (%.1fx less)\n\n",
           label, tp_size, (double)total_params/1e6,
           (dtype==Dtype::FP32?"fp32":dtype==Dtype::FP16?"fp16":"bf16"),
           full_gb, shard_gb, (double)tp_size);
}

} // namespace tp
} // namespace hpc_opt
