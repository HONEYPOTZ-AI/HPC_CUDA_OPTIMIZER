// =============================================================================
// HPC CUDA Optimizer Library
// hpc_lamb.cuh  –  LAMB (Layer-wise Adaptive Moments) optimizer
//
// Reference: You et al., "Large Batch Optimization for Deep Learning:
//            Training BERT in 76 minutes", ICLR 2020.
//
// LAMB is Adam + a per-layer trust-ratio that scales the update by
//   phi(p_norm) / update_norm
// This allows stable training at very large batch sizes (4096+) without
// LR warmup tuning.
//
// HPC design:
//   • Two-kernel per tensor: (1) compute m/v/update, (2) compute norms, (3) apply
//   • Norms computed via warp-shuffle reduction (no atomics on critical path)
//   • Supports FP32 / FP16+master / BF16+master
//   • trust_ratio clamped to clamp_value (default 10) for stability
//   • LAMB-W variant: decoupled weight decay (recommended)
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Kernel 1: compute moment update buffer (stored in temp_update)
// ---------------------------------------------------------------------------
template<typename ParamT, typename GradT>
__global__ void k_lamb_update(
        const ParamT* __restrict__ params,
        const float*  __restrict__ master,   // FP32 master (may be nullptr)
        const GradT*  __restrict__ grads,
        float*        __restrict__ m,
        float*        __restrict__ v,
        float*        __restrict__ update,   // output: unscaled step direction
        size_t  numel,
        float   beta1, float beta2, float eps,
        float   weight_decay, bool lamb_w)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float p = master ? __ldg(master + i) : prec::to_float(__ldg(params + i));
        float g = prec::to_float(__ldg(grads + i));

        if (!prec::is_finite(g)) { update[i] = 0.0f; continue; }

        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi; v[i] = vi;

        float u = mi / (sqrtf(vi) + eps);

        // Weight-decay: LAMB-W adds wd*p to update (decoupled); classic LAMB adds to grad
        if (weight_decay != 0.0f)
            u += lamb_w ? weight_decay * p : 0.0f;   // classic WD already in g

        update[i] = u;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 (two passes on shared memory): compute ||p||₂ and ||u||₂
//   Partial results written to d_p_norm_partial and d_u_norm_partial,
//   both of size gridDim.x.
// ---------------------------------------------------------------------------
template<typename ParamT>
__global__ void k_lamb_norms(
        const ParamT* __restrict__ params,
        const float*  __restrict__ master,
        const float*  __restrict__ update,
        float*        __restrict__ p_norm_partial,
        float*        __restrict__ u_norm_partial,
        size_t numel)
{
    constexpr int WARPS = HPC_BLOCK / HPC_WARP;
    __shared__ float sp[WARPS], su[WARPS];

    float lp = 0.0f, lu = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float p = master ? __ldg(master + i) : prec::to_float(__ldg(params + i));
        float u = __ldg(update + i);
        lp += p * p;
        lu += u * u;
    }

    // Warp reduction
    lp = warp_reduce_sum(lp);
    lu = warp_reduce_sum(lu);

    int lane = threadIdx.x % HPC_WARP;
    int wid  = threadIdx.x / HPC_WARP;
    if (lane == 0) { sp[wid] = lp; su[wid] = lu; }
    __syncthreads();

    if (wid == 0) {
        lp = (lane < WARPS) ? sp[lane] : 0.0f;
        lu = (lane < WARPS) ? su[lane] : 0.0f;
        lp = warp_reduce_sum(lp);
        lu = warp_reduce_sum(lu);
        if (lane == 0) {
            atomicAdd(p_norm_partial, lp);
            atomicAdd(u_norm_partial, lu);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: apply scaled update
// ---------------------------------------------------------------------------
template<typename ParamT>
__global__ void k_lamb_apply(
        ParamT*       __restrict__ params,
        float*        __restrict__ master,
        const float*  __restrict__ update,
        size_t  numel,
        float   trust_ratio,
        float   lr)
{
    float step = lr * trust_ratio;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float p = master ? __ldg(master + i) : prec::to_float(__ldg(params + i));
        p -= step * __ldg(update + i);

        if (master) {
            master[i] = p;
            if constexpr (std::is_same_v<ParamT, half>)
                params[i] = __float2half(p);
            else if constexpr (std::is_same_v<ParamT, __nv_bfloat16>)
                params[i] = __float2bfloat16(p);
        } else {
            params[i] = p;
        }
    }
}

// ===========================================================================
// LAMBOptimizer  –  stateful HPC wrapper
// ===========================================================================
class LAMBOptimizer {
public:
    explicit LAMBOptimizer(const LAMBConfig& cfg) : cfg_(cfg) {}
    ~LAMBOptimizer() { free_state(); }

    LAMBOptimizer(const LAMBOptimizer&)            = delete;
    LAMBOptimizer& operator=(const LAMBOptimizer&) = delete;

    void init(const TensorView* params, int n,
              float** master = nullptr,
              cudaStream_t stream = 0)
    {
        free_state();
        n_tensors_ = n;
        master_    = master;
        m_         = new float*[n];
        v_         = new float*[n];
        update_    = new float*[n];

        for (int i = 0; i < n; ++i) {
            size_t bytes = params[i].numel * sizeof(float);
            HPC_CUDA_CHECK(cudaMalloc(&m_[i],      bytes));
            HPC_CUDA_CHECK(cudaMalloc(&v_[i],      bytes));
            HPC_CUDA_CHECK(cudaMalloc(&update_[i], bytes));
            HPC_CUDA_CHECK(cudaMemsetAsync(m_[i], 0, bytes, stream));
            HPC_CUDA_CHECK(cudaMemsetAsync(v_[i], 0, bytes, stream));
        }

        // Scratch for per-tensor norm partials
        HPC_CUDA_CHECK(cudaMalloc(&d_p_norm_, sizeof(float)));
        HPC_CUDA_CHECK(cudaMalloc(&d_u_norm_, sizeof(float)));
    }

    void step(TensorView* params, const TensorView* grads, int n,
              cudaStream_t stream = 0)
    {
        step_++;
        const float bc1 = 1.0f - powf(cfg_.beta1, (float)step_);
        const float bc2 = 1.0f - powf(cfg_.beta2, (float)step_);

        for (int i = 0; i < n; ++i) {
            size_t numel = params[i].numel;
            int    blk   = hpc_blocks(numel);

            // --- Step 1: compute update direction ---
            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_lamb_update<float, float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const float*>(params[i].data), nullptr,
                        static_cast<const float*>(grads[i].data),
                        m_[i], v_[i], update_[i], numel,
                        cfg_.beta1 / bc1, cfg_.beta2 / bc2,   // bias-corrected inline
                        cfg_.eps, cfg_.weight_decay, cfg_.adam_w_mode);
                    break;
                case Dtype::FP16:
                    k_lamb_update<half, half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const half*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const half*>(grads[i].data),
                        m_[i], v_[i], update_[i], numel,
                        cfg_.beta1, cfg_.beta2, cfg_.eps,
                        cfg_.weight_decay, cfg_.adam_w_mode);
                    break;
                case Dtype::BF16:
                    k_lamb_update<__nv_bfloat16, __nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const __nv_bfloat16*>(grads[i].data),
                        m_[i], v_[i], update_[i], numel,
                        cfg_.beta1, cfg_.beta2, cfg_.eps,
                        cfg_.weight_decay, cfg_.adam_w_mode);
                    break;
            }

            // --- Step 2: compute norms ---
            HPC_CUDA_CHECK(cudaMemsetAsync(d_p_norm_, 0, sizeof(float), stream));
            HPC_CUDA_CHECK(cudaMemsetAsync(d_u_norm_, 0, sizeof(float), stream));

            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_lamb_norms<float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const float*>(params[i].data), nullptr,
                        update_[i], d_p_norm_, d_u_norm_, numel);
                    break;
                case Dtype::FP16:
                    k_lamb_norms<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const half*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        update_[i], d_p_norm_, d_u_norm_, numel);
                    break;
                case Dtype::BF16:
                    k_lamb_norms<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        update_[i], d_p_norm_, d_u_norm_, numel);
                    break;
            }

            // --- Host-side trust ratio (requires stream sync) ---
            float p_sq = 0.0f, u_sq = 0.0f;
            HPC_CUDA_CHECK(cudaMemcpyAsync(&p_sq, d_p_norm_, sizeof(float), cudaMemcpyDeviceToHost, stream));
            HPC_CUDA_CHECK(cudaMemcpyAsync(&u_sq, d_u_norm_, sizeof(float), cudaMemcpyDeviceToHost, stream));
            HPC_CUDA_CHECK(cudaStreamSynchronize(stream));

            float p_norm = sqrtf(p_sq);
            float u_norm = sqrtf(u_sq);
            float trust  = (p_norm > 0.0f && u_norm > 0.0f)
                           ? p_norm / u_norm : 1.0f;
            trust = fminf(trust, cfg_.clamp_value);

            // --- Step 3: apply scaled update ---
            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_lamb_apply<float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<float*>(params[i].data), nullptr,
                        update_[i], numel, trust, cfg_.lr);
                    break;
                case Dtype::FP16:
                    k_lamb_apply<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<half*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        update_[i], numel, trust, cfg_.lr);
                    break;
                case Dtype::BF16:
                    k_lamb_apply<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<__nv_bfloat16*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        update_[i], numel, trust, cfg_.lr);
                    break;
            }
        }
    }

    uint32_t           step_count() const { return step_; }
    LAMBConfig&        config()       { return cfg_; }
    const LAMBConfig&  config() const { return cfg_; }

private:
    LAMBConfig cfg_;
    float**    m_         = nullptr;
    float**    v_         = nullptr;
    float**    update_    = nullptr;
    float**    master_    = nullptr;
    float*     d_p_norm_  = nullptr;
    float*     d_u_norm_  = nullptr;
    int        n_tensors_ = 0;
    uint32_t   step_      = 0;

    void free_state() {
        if (!m_) return;
        for (int i = 0; i < n_tensors_; ++i) {
            cudaFree(m_[i]); cudaFree(v_[i]); cudaFree(update_[i]);
        }
        delete[] m_; delete[] v_; delete[] update_;
        if (d_p_norm_) cudaFree(d_p_norm_);
        if (d_u_norm_) cudaFree(d_u_norm_);
        m_ = v_ = update_ = nullptr;
        d_p_norm_ = d_u_norm_ = nullptr;
        n_tensors_ = 0;
    }
};

} // namespace hpc_opt
