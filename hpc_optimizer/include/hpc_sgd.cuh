// =============================================================================
// HPC CUDA Optimizer Library
// hpc_sgd.cuh  –  Fused SGD + Nesterov momentum
//
// HPC choices:
//   • float4 vectorised FP32 path (4 elements / thread, 128-bit loads)
//   • half2 vectorised FP16 path  (2 elements / thread)
//   • bfloat162 vectorised BF16 path (Ampere+)
//   • Nesterov update: effective_grad = g + momentum * v  (before v update)
//   • weight-decay applied to raw gradient (L2 regularisation)
//   • Large-batch HPC: dampening=0 recommended; dampening path included
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include <cuda_runtime.h>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Generic SGD kernel (scalar, all dtypes)
// ---------------------------------------------------------------------------
template<typename ParamT, typename GradT>
__global__ void k_sgd(
        ParamT* __restrict__       params,
        float*  __restrict__       master,   // nullptr → pure FP32
        const GradT* __restrict__  grads,
        float*  __restrict__       vel,      // velocity buffer (always FP32)
        size_t  numel,
        float   lr,
        float   momentum,
        float   dampening,
        float   weight_decay,
        bool    nesterov)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += gridDim.x * blockDim.x)
    {
        float p = master ? __ldg(master + i) : prec::to_float(__ldg(params + i));
        float g = prec::to_float(__ldg(grads + i));

        if (!prec::is_finite(g)) continue;

        // Weight decay (L2 on gradient)
        if (weight_decay != 0.0f) g += weight_decay * p;

        // Momentum
        if (momentum != 0.0f) {
            float vt = momentum * __ldg(vel + i) + (1.0f - dampening) * g;
            vel[i] = vt;
            g = nesterov ? (g + momentum * vt) : vt;
        }

        p -= lr * g;

        if (master) {
            master[i] = p;
            if constexpr (std::is_same_v<ParamT, half>)
                params[i] = __float2half(p);
            else if constexpr (std::is_same_v<ParamT, __nv_bfloat16>)
                params[i] = __float2bfloat16(p);
        } else {
            params[i] = static_cast<float>(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Vectorised FP32 SGD — float4 (no Nesterov, dampening=0 assumed for HPC)
// ---------------------------------------------------------------------------
__global__ void k_sgd_vec4_fp32(
        float* __restrict__       params,
        const float* __restrict__ grads,
        float* __restrict__       vel,
        size_t  numel4,
        float   lr,
        float   momentum,
        float   weight_decay)
{
    for (size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
         i4 < numel4;
         i4 += gridDim.x * blockDim.x)
    {
        size_t b = i4 * 4;
        float4 p4 = *reinterpret_cast<const float4*>(params + b);
        float4 g4 = *reinterpret_cast<const float4*>(grads  + b);
        float4 v4 = *reinterpret_cast<const float4*>(vel    + b);

#define SGD_STEP(x) \
        g4.x += weight_decay * p4.x; \
        v4.x = momentum * v4.x + g4.x; \
        p4.x -= lr * v4.x;

        SGD_STEP(x) SGD_STEP(y) SGD_STEP(z) SGD_STEP(w)
#undef SGD_STEP

        *reinterpret_cast<float4*>(params + b) = p4;
        *reinterpret_cast<float4*>(vel    + b) = v4;
    }
}

// ---------------------------------------------------------------------------
// Vectorised BF16 SGD — bfloat162 + FP32 master
// ---------------------------------------------------------------------------
__global__ void k_sgd_vec2_bf16(
        __nv_bfloat16* __restrict__       params,
        float*         __restrict__       master,
        const __nv_bfloat16* __restrict__ grads,
        float*         __restrict__       vel,
        size_t  numel2,
        float   lr,
        float   momentum,
        float   weight_decay)
{
    for (size_t i2 = blockIdx.x * blockDim.x + threadIdx.x;
         i2 < numel2;
         i2 += gridDim.x * blockDim.x)
    {
        size_t b = i2 * 2;
        float g0, g1;
        prec::load2_bf16(grads, b, g0, g1);

        float p0 = master[b],     p1 = master[b + 1];
        float v0 = vel[b],        v1 = vel[b + 1];

        g0 += weight_decay * p0;
        v0 = momentum * v0 + g0;
        p0 -= lr * v0;

        g1 += weight_decay * p1;
        v1 = momentum * v1 + g1;
        p1 -= lr * v1;

        master[b] = p0; master[b + 1] = p1;
        vel[b]    = v0; vel[b + 1]    = v1;
        prec::store2_bf16(params, b, p0, p1);
    }
}

// ===========================================================================
// SGDOptimizer — stateful HPC wrapper
// ===========================================================================
class SGDOptimizer {
public:
    explicit SGDOptimizer(const SGDConfig& cfg) : cfg_(cfg) {}
    ~SGDOptimizer() { free_state(); }

    SGDOptimizer(const SGDOptimizer&)            = delete;
    SGDOptimizer& operator=(const SGDOptimizer&) = delete;

    void init(const TensorView* params, int n,
              float** master = nullptr,
              cudaStream_t stream = 0)
    {
        free_state();
        n_tensors_ = n;
        master_    = master;
        vel_       = new float*[n];

        for (int i = 0; i < n; ++i) {
            size_t bytes = params[i].numel * sizeof(float);
            HPC_CUDA_CHECK(cudaMalloc(&vel_[i], bytes));
            HPC_CUDA_CHECK(cudaMemsetAsync(vel_[i], 0, bytes, stream));
        }
    }

    void step(TensorView* params, const TensorView* grads, int n,
              cudaStream_t stream = 0)
    {
        for (int i = 0; i < n; ++i) {
            size_t numel = params[i].numel;
            int    blk   = hpc_blocks(numel);

            // FP32 vectorised fast path
            if (params[i].dtype == Dtype::FP32 && grads[i].dtype == Dtype::FP32
                && !cfg_.nesterov && cfg_.dampening == 0.0f && (numel % 4 == 0))
            {
                k_sgd_vec4_fp32<<<blk, HPC_BLOCK, 0, stream>>>(
                    static_cast<float*>(params[i].data),
                    static_cast<const float*>(grads[i].data),
                    vel_[i], numel / 4,
                    cfg_.lr, cfg_.momentum, cfg_.weight_decay);
                continue;
            }

            // BF16 vectorised (Ampere+)
            if (params[i].dtype == Dtype::BF16 && grads[i].dtype == Dtype::BF16
                && master_ && (numel % 2 == 0))
            {
                k_sgd_vec2_bf16<<<blk, HPC_BLOCK, 0, stream>>>(
                    static_cast<__nv_bfloat16*>(params[i].data),
                    master_[i],
                    static_cast<const __nv_bfloat16*>(grads[i].data),
                    vel_[i], numel / 2,
                    cfg_.lr, cfg_.momentum, cfg_.weight_decay);
                continue;
            }

            // Generic scalar fallback
            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_sgd<float, float><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<float*>(params[i].data), nullptr,
                        static_cast<const float*>(grads[i].data),
                        vel_[i], numel, cfg_.lr, cfg_.momentum,
                        cfg_.dampening, cfg_.weight_decay, cfg_.nesterov);
                    break;
                case Dtype::FP16:
                    k_sgd<half, half><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<half*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const half*>(grads[i].data),
                        vel_[i], numel, cfg_.lr, cfg_.momentum,
                        cfg_.dampening, cfg_.weight_decay, cfg_.nesterov);
                    break;
                case Dtype::BF16:
                    k_sgd<__nv_bfloat16, __nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        static_cast<__nv_bfloat16*>(params[i].data),
                        master_ ? master_[i] : nullptr,
                        static_cast<const __nv_bfloat16*>(grads[i].data),
                        vel_[i], numel, cfg_.lr, cfg_.momentum,
                        cfg_.dampening, cfg_.weight_decay, cfg_.nesterov);
                    break;
            }
        }
        step_++;
    }

    float**           vel_buffers() { return vel_; }
    uint32_t          step_count() const { return step_; }
    SGDConfig&        config()       { return cfg_; }
    const SGDConfig&  config() const { return cfg_; }

private:
    SGDConfig cfg_;
    float**   vel_       = nullptr;
    float**   master_    = nullptr;
    int       n_tensors_ = 0;
    uint32_t  step_      = 0;

    void free_state() {
        if (!vel_) return;
        for (int i = 0; i < n_tensors_; ++i) cudaFree(vel_[i]);
        delete[] vel_; vel_ = nullptr; n_tensors_ = 0;
    }
};

} // namespace hpc_opt
