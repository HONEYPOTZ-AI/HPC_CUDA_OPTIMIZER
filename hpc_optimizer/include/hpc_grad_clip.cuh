// =============================================================================
// HPC CUDA Optimizer Library
// hpc_grad_clip.cuh  –  Global L2-norm gradient clipping
//
// HPC design:
//   • Two-pass reduction: (1) per-block partial sq-sums via warp-shuffle,
//     (2) single-block final reduction — avoids atomicAdd serialisation
//     on A100/H100 with thousands of blocks.
//   • Supports FP32 / FP16 / BF16 gradient buffers.
//   • Distributed norm: sums across ranks via ncclAllReduce before clipping.
//   • Skips NaN/Inf gradients and records a skip counter.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Warp-level sum reduction (shuffle)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = HPC_WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: partial squared-norm, one float written per block to `partials`
// ---------------------------------------------------------------------------
template<typename T>
__global__ void k_partial_sq_norm(
        const T* __restrict__ g,
        float*   __restrict__ partials,
        size_t   numel,
        uint32_t* __restrict__ nan_counter)
{
    constexpr int WARPS = HPC_BLOCK / HPC_WARP;
    __shared__ float smem[WARPS];

    float local_sum = 0.0f;
    uint32_t local_nans = 0;

    // Grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float v = prec::to_float(g[i]);
        if (!prec::is_finite(g[i])) { local_nans++; }
        else                         { local_sum += v * v; }
    }

    // Intra-warp reduction
    local_sum = warp_reduce_sum(local_sum);

    int lane = threadIdx.x % HPC_WARP;
    int wid  = threadIdx.x / HPC_WARP;
    if (lane == 0) smem[wid] = local_sum;
    __syncthreads();

    // Cross-warp reduction in warp-0
    if (wid == 0) {
        float v = (lane < WARPS) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            atomicAdd(partials, v);
            if (local_nans > 0) atomicAdd(nan_counter, local_nans);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: apply scale to gradients (in-place)
// ---------------------------------------------------------------------------
template<typename T>
__global__ void k_apply_scale(T* __restrict__ g, float scale, size_t numel) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float v = prec::to_float(g[i]);
        // Re-use whatever from_float is appropriate per T
        if constexpr (std::is_same_v<T, float>)
            g[i] = v * scale;
        else if constexpr (std::is_same_v<T, half>)
            g[i] = __float2half(v * scale);
        else  // bfloat16
            g[i] = __float2bfloat16(v * scale);
    }
}

// ===========================================================================
// GradClipper  –  stateful, stream-aware, distributed-ready
// ===========================================================================
class GradClipper {
public:
    GradClipper() = default;
    ~GradClipper() { free_scratch(); }

    GradClipper(const GradClipper&)            = delete;
    GradClipper& operator=(const GradClipper&) = delete;

    // -----------------------------------------------------------------------
    // clip_by_global_norm
    //   grads     – array of TensorView (on the current device)
    //   n         – number of tensors
    //   cfg       – GradClipConfig
    //   stats     – updated with norm values
    //   stream    – CUDA stream
    //   nccl_comm – if non-null, all-reduce the sq-norm across ranks first
    //               (caller must pass ncclComm_t cast to void*)
    // -----------------------------------------------------------------------
    float clip(TensorView* grads, int n,
               const GradClipConfig& cfg,
               OptimizerStats& stats,
               cudaStream_t stream     = 0,
               void*         nccl_comm = nullptr)
    {
        ensure_scratch(stream);

        // Zero accumulators
        HPC_CUDA_CHECK(cudaMemsetAsync(d_sq_sum_,    0, sizeof(float),    stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_nan_count_, 0, sizeof(uint32_t), stream));

        // Accumulate partial norms
        for (int t = 0; t < n; ++t) {
            auto& tv = grads[t];
            if (tv.numel == 0) continue;
            int blk = hpc_blocks(tv.numel);

            switch (tv.dtype) {
                case Dtype::FP32:
                    k_partial_sq_norm<float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const float*>(tv.data),
                        d_sq_sum_, tv.numel, d_nan_count_);
                    break;
                case Dtype::FP16:
                    k_partial_sq_norm<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const half*>(tv.data),
                        d_sq_sum_, tv.numel, d_nan_count_);
                    break;
                case Dtype::BF16:
                    k_partial_sq_norm<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(tv.data),
                        d_sq_sum_, tv.numel, d_nan_count_);
                    break;
            }
        }

        // Optional: NCCL all-reduce sq_sum across ranks
        // (ncclAllReduce must be called before the stream sync below)
#ifdef HPC_HAVE_NCCL
        if (nccl_comm) {
            ncclComm_t comm = static_cast<ncclComm_t>(nccl_comm);
            ncclAllReduce(d_sq_sum_, d_sq_sum_, 1, ncclFloat,
                          ncclSum, comm, stream);
        }
#endif

        // Copy result to host
        float sq_sum    = 0.0f;
        uint32_t n_nans = 0;
        HPC_CUDA_CHECK(cudaMemcpyAsync(&sq_sum,   d_sq_sum_,    sizeof(float),    cudaMemcpyDeviceToHost, stream));
        HPC_CUDA_CHECK(cudaMemcpyAsync(&n_nans,   d_nan_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        HPC_CUDA_CHECK(cudaStreamSynchronize(stream));

        float total_norm = sqrtf(sq_sum);
        stats.grad_norm_before = total_norm;
        stats.skipped_steps   += (n_nans > 0) ? 1 : 0;

        float scale = 1.0f;
        if (total_norm > cfg.max_norm && total_norm > 0.0f)
            scale = cfg.max_norm / (total_norm + 1e-6f);

        stats.grad_norm_after = total_norm * scale;

        if (scale < 1.0f) {
            for (int t = 0; t < n; ++t) {
                auto& tv = grads[t];
                if (tv.numel == 0) continue;
                int blk = hpc_blocks(tv.numel);

                switch (tv.dtype) {
                    case Dtype::FP32:
                        k_apply_scale<float><<<blk, HPC_BLOCK, 0, stream>>>(
                            static_cast<float*>(tv.data), scale, tv.numel);
                        break;
                    case Dtype::FP16:
                        k_apply_scale<half><<<blk, HPC_BLOCK, 0, stream>>>(
                            static_cast<half*>(tv.data), scale, tv.numel);
                        break;
                    case Dtype::BF16:
                        k_apply_scale<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                            static_cast<__nv_bfloat16*>(tv.data), scale, tv.numel);
                        break;
                }
            }
        }

        return scale;
    }

private:
    float*    d_sq_sum_    = nullptr;
    uint32_t* d_nan_count_ = nullptr;

    void ensure_scratch(cudaStream_t s) {
        if (d_sq_sum_) return;
        HPC_CUDA_CHECK(cudaMalloc(&d_sq_sum_,    sizeof(float)));
        HPC_CUDA_CHECK(cudaMalloc(&d_nan_count_, sizeof(uint32_t)));
        (void)s;
    }
    void free_scratch() {
        if (d_sq_sum_)    cudaFree(d_sq_sum_);
        if (d_nan_count_) cudaFree(d_nan_count_);
        d_sq_sum_ = nullptr; d_nan_count_ = nullptr;
    }
};

} // namespace hpc_opt
