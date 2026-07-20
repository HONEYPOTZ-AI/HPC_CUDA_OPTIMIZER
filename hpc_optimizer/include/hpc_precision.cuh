// =============================================================================
// HPC CUDA Optimizer Library
// hpc_precision.cuh  –  Precision conversion + arithmetic helpers
//
// Provides uniform load/store/compute across FP32 / FP16 / BF16 so kernels
// can be templated on a single <DType> parameter.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace hpc_opt {
namespace prec {

// ---------------------------------------------------------------------------
// load<T>  –  load element from any supported dtype pointer
// ---------------------------------------------------------------------------

// Unified storage type tag wrappers
template<Dtype D> struct StorageType;
template<> struct StorageType<Dtype::FP32> { using type = float;          };
template<> struct StorageType<Dtype::FP16> { using type = half;           };
template<> struct StorageType<Dtype::BF16> { using type = __nv_bfloat16;  };

// Convert to float for compute
__device__ __forceinline__ float to_float(float x)             { return x; }
__device__ __forceinline__ float to_float(half x)              { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x)     { return __bfloat162float(x); }

// Convert from float to storage type
__device__ __forceinline__ float          from_float_fp32(float x) { return x; }
__device__ __forceinline__ half           from_float_fp16(float x) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16  from_float_bf16(float x) { return __float2bfloat16(x); }

// ---------------------------------------------------------------------------
// Vectorized load/store (128-bit aligned) for bandwidth efficiency
// Supports float4 (FP32), half2 (FP16), nv_bfloat162 (BF16)
// ---------------------------------------------------------------------------

// FP32 – 4 elements at once via float4
__device__ __forceinline__ void load4_fp32(const float* __restrict__ src,
                                           size_t i,
                                           float& a, float& b, float& c, float& d)
{
    float4 v = *reinterpret_cast<const float4*>(src + i);
    a = v.x; b = v.y; c = v.z; d = v.w;
}

__device__ __forceinline__ void store4_fp32(float* __restrict__ dst,
                                            size_t i,
                                            float a, float b, float c, float d)
{
    float4 v; v.x = a; v.y = b; v.z = c; v.w = d;
    *reinterpret_cast<float4*>(dst + i) = v;
}

// FP16 – 2 elements at once via half2
__device__ __forceinline__ void load2_fp16(const half* __restrict__ src,
                                           size_t i, float& a, float& b)
{
    half2 v = *reinterpret_cast<const half2*>(src + i);
    a = __half2float(v.x);
    b = __half2float(v.y);
}

__device__ __forceinline__ void store2_fp16(half* __restrict__ dst,
                                            size_t i, float a, float b)
{
    half2 v;
    v.x = __float2half(a);
    v.y = __float2half(b);
    *reinterpret_cast<half2*>(dst + i) = v;
}

// BF16 – 2 elements at once via nv_bfloat162
__device__ __forceinline__ void load2_bf16(const __nv_bfloat16* __restrict__ src,
                                           size_t i, float& a, float& b)
{
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(src + i);
    a = __bfloat162float(v.x);
    b = __bfloat162float(v.y);
}

__device__ __forceinline__ void store2_bf16(__nv_bfloat16* __restrict__ dst,
                                            size_t i, float a, float b)
{
    __nv_bfloat162 v;
    v.x = __float2bfloat16(a);
    v.y = __float2bfloat16(b);
    *reinterpret_cast<__nv_bfloat162*>(dst + i) = v;
}

// ---------------------------------------------------------------------------
// NaN / Inf guard  –  used to detect and skip bad gradients
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool is_finite(float x) {
    return isfinite(x);
}

__device__ __forceinline__ bool is_finite(half x) {
    return __hisnan(x) == 0 && __hisinf(x) == 0;
}

__device__ __forceinline__ bool is_finite(__nv_bfloat16 x) {
    return is_finite(__bfloat162float(x));
}

// ---------------------------------------------------------------------------
// stochastic_round_fp16  –  stochastic rounding for FP16 storage
// Better training dynamics vs truncation for large-batch HPC
// ---------------------------------------------------------------------------
__device__ __forceinline__ half stochastic_round(float x, unsigned& rng_state) {
    // LCG noise in [0, 2^-11) added before truncation
    rng_state = rng_state * 1664525u + 1013904223u;
    float noise = static_cast<float>(rng_state >> 21) * (1.0f / (1 << 11));
    // Add noise in the direction that makes rounding probabilistic
    uint32_t xi; memcpy(&xi, &x, 4);
    float lsb = __uint_as_float((xi & 0x7F800000u) | 0x00001000u) - __uint_as_float((xi & 0x7F800000u));
    return __float2half(x + noise * lsb - 0.5f * lsb);
}

} // namespace prec
} // namespace hpc_opt
