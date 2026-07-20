// =============================================================================
// HPC CUDA Optimizer Library
// hpc_lion.cuh  –  Lion optimizer (EvoLved Sign Momentum)
//
// Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms",
//            NeurIPS 2023.
//
// Update rule:
//   update = sign(β₁·m + (1-β₁)·g)     ← interpolated gradient sign
//   p      = p - lr * (update + wd·p)   ← decoupled weight decay
//   m      = β₂·m + (1-β₂)·g           ← EMA update of momentum buffer
//
// HPC advantages over Adam:
//   • Only ONE moment buffer (vs Adam's two) → 33% less optimizer state memory
//   • All arithmetic is sign-based → no sqrt, no bias-correction, cheaper kernels
//   • float4 vectorised for FP32; half2 for FP16; bfloat162 for BF16
//   • Recommended lr ~3-10× smaller than Adam for the same model
//
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include <cuda_runtime.h>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Generic Lion kernel (scalar path – all dtypes)
// ---------------------------------------------------------------------------
template<typename ParamT, typename GradT>
__global__ void k_lion(
        ParamT*        __restrict__ params,
        float*         __restrict__ master,   // nullptr → pure FP32
        const GradT*   __restrict__ grads,
        float*         __restrict__ m,        // momentum buffer (FP32)
        size_t  numel,
        float   lr,
        float   beta1,
        float   beta2,
        float   weight_decay)
{
    const float one_m_b1 = 1.0f - beta1;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float p = master ? __ldg(master + i) : prec::to_float(__ldg(params + i));
        float g = prec::to_float(__ldg(grads + i));
        float mi = __ldg(m + i);

        if (!prec::is_finite(g)) continue;

        // sign of interpolation: sign(β₁·m + (1-β₁)·g)
        float interp = beta1 * mi + one_m_b1 * g;
        float update = (interp > 0.0f) - (interp < 0.0f);  // sign: -1/0/+1

        // decoupled weight decay + update step
        p = p - lr * (update + weight_decay * p);

        // momentum EMA
        mi = beta2 * mi + (1.0f - beta2) * g;
        m[i] = mi;

        // write back
        if (master) {
            master[i] = p;
            if constexpr (std::is_same_v<ParamT, half>)
                params[i] = __float2half(p);
            else
                params[i] = __float2bfloat16(p);
        } else {
            params[i] = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorised FP32 Lion — float4, 4 elements / thread
// ---------------------------------------------------------------------------
__global__ void k_lion_vec4_fp32(
        float*        __restrict__ params,
        const float*  __restrict__ grads,
        float*        __restrict__ m,
        size_t  numel4,
        float   lr,
        float   beta1,
        float   beta2,
        float   weight_decay)
{
    const float one_m_b1 = 1.0f - beta1;
    const float one_m_b2 = 1.0f - beta2;

    for (size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
         i4 < numel4;
         i4 += gridDim.x * blockDim.x)
    {
        size_t b = i4 * 4;
        float4 p4 = *reinterpret_cast<const float4*>(params + b);
        float4 g4 = *reinterpret_cast<const float4*>(grads  + b);
        float4 m4 = *reinterpret_cast<const float4*>(m      + b);

#define LION_STEP(x) \
        { \
            float interp = beta1 * m4.x + one_m_b1 * g4.x; \
            float upd    = (interp > 0.0f) - (interp < 0.0f); \
            p4.x = p4.x - lr * (upd + weight_decay * p4.x); \
            m4.x = beta2 * m4.x + one_m_b2 * g4.x; \
        }

        LION_STEP(x) LION_STEP(y) LION_STEP(z) LION_STEP(w)
#undef LION_STEP

        *reinterpret_cast<float4*>(params + b) = p4;
        *reinterpret_cast<float4*>(m      + b) = m4;
    }
}

// ---------------------------------------------------------------------------
// Vectorised BF16 Lion — bfloat162, 2 elements / thread (Ampere+)
// ---------------------------------------------------------------------------
__global__ void k_lion_vec2_bf16(
        __nv_bfloat16* __restrict__       params,
        float*         __restrict__       master,
        const __nv_bfloat16* __restrict__ grads,
        float*         __restrict__       m,
        size_t  numel2,
        float   lr,
        float   beta1,
        float   beta2,
        float   weight_decay)
{
    const float one_m_b1 = 1.0f - beta1;
    const float one_m_b2 = 1.0f - beta2;

    for (size_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
         i2 < numel2;
         i2 += gridDim.x * blockDim.x)
    {
        size_t b = i2 * 2;
        float g0, g1;
        prec::load2_bf16(grads, b, g0, g1);

        float p0 = master[b],     p1 = master[b + 1];
        float m0 = m[b],          m1 = m[b + 1];

        // Slot 0
        float i0 = beta1 * m0 + one_m_b1 * g0;
        float u0 = (i0 > 0.0f) - (i0 < 0.0f);
        p0 = p0 - lr * (u0 + weight_decay * p0);
        m0 = beta2 * m0 + one_m_b2 * g0;

        // Slot 1
        float i1 = beta1 * m1 + one_m_b1 * g1;
        float u1 = (i1 > 0.0f) - (i1 < 0.0f);
        p1 = p1 - lr * (u1 + weight_decay * p1);
        m1 = beta2 * m1 + one_m_b2 * g1;

        master[b] = p0; master[b + 1] = p1;
        m[b]      = m0; m[b + 1]      = m1;
        prec::store2_bf16(params, b, p0, p1);
    }
}

// ===========================================================================
// LionOptimizer  –  stateful HPC wrapper
// ===========================================================================
class LionOptimizer {
public:
    explicit LionOptimizer(const LionConfig& cfg) : cfg_(cfg) {}
    ~LionOptimizer() { free_state(); }

    LionOptimizer(const LionOptimizer&)            = delete;
    LionOptimizer& operator=(const LionOptimizer&) = delete;

    void init(const TensorView* params, int n,
              float** master = nullptr,
              cudaStream_t stream = 0)
    {
        free_state();
        n_tensors_ = n;
        master_    = master;
        m_         = new float*[n];

        for (int i = 0; i < n; ++i) {
            size_t bytes = params[i].numel * sizeof(float);
            HPC_CUDA_CHECK(cudaMalloc(&m_[i], bytes));
            HPC_CUDA_CHECK(cudaMemsetAsync(m_[i], 0, bytes, stream));
        }
    }

    void step(TensorView* params, const TensorView* grads, int n,
              cudaStream_t stream = 0)
    {
        for (int i = 0; i < n; ++i) {
            size_t numel = params[i].numel;
            int    blk   = hpc_blocks(numel);

            // FP32 vectorised fast path
            if (params[i].dtype == Dtype::FP32 &&
                grads[i].dtype  == Dtype::FP32 &&
                (numel % 4 == 0))
            {
                k_lion_vec4_fp32<<<blk, HPC_BLOCK, 0, stream>>>(
                    static_cast<float*>(params[i].data),
                    static_cast<const float*>(grads[i].data),
                    m_[i], numel / 4,
                    cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.weight_decay);
                continue;
            }

            // BF16 + master path
            if (params[i].dtype == Dtype::BF16 && master_ && (numel % 2 == 0)) {
                k_lion_vec2_bf16<<<blk, HPC_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat16*>(params[i].data),
                    master_[i],
                    static_cast<const __nv_bfloat16*>(grads[i].data),
                    m_[i], numel / 2,
                    cfg_.lr, cfg_.beta1, cfg_.beta2, cfg_.weight_decay);
                continue;
            }

            // Scalar fallback
            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_lion<float, float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<float*>(params[i].data), nullptr,
                        static_cast<const float*>(grads[i].data),
                        m_[i], numel, cfg_.lr,
                        cfg_.beta1, cfg_.beta2, cfg_.weight_decay);
                    break;
                case Dtype::FP16:
                    k_lion<half, half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<half*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const half*>(grads[i].data),
                        m_[i], numel, cfg_.lr,
                        cfg_.beta1, cfg_.beta2, cfg_.weight_decay);
                    break;
                case Dtype::BF16:
                    k_lion<__nv_bfloat16, __nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<__nv_bfloat16*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const __nv_bfloat16*>(grads[i].data),
                        m_[i], numel, cfg_.lr,
                        cfg_.beta1, cfg_.beta2, cfg_.weight_decay);
                    break;
            }
        }
        step_++;
    }

    // Lion only has one moment buffer — 33% less memory than Adam
    float**           m_buffers()  { return m_; }
    uint32_t          step_count() const { return step_; }
    LionConfig&       config()       { return cfg_; }
    const LionConfig& config() const { return cfg_; }

private:
    LionConfig cfg_;
    float**    m_         = nullptr;
    float**    master_    = nullptr;
    int        n_tensors_ = 0;
    uint32_t   step_      = 0;

    void free_state() {
        if (!m_) return;
        for (int i = 0; i < n_tensors_; ++i) cudaFree(m_[i]);
        delete[] m_; m_ = nullptr; n_tensors_ = 0;
    }
};

} // namespace hpc_opt
