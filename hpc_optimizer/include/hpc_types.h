// =============================================================================
// HPC CUDA Optimizer Library
// hpc_types.h  –  Types, precision tags, HPC config structs
//
// Targets: V100 (sm_70) | A100 (sm_80) | H100 (sm_90) | RTX 4090 (sm_89)
// CUDA 12.x | C++17 | MPI + NCCL | cuBLAS optional
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>      // __nv_bfloat16  (Ampere+)
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Compile-time SM capability detection
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__)
#  if __CUDA_ARCH__ >= 900
#    define HPC_SM_HOPPER  1
#  endif
#  if __CUDA_ARCH__ >= 800
#    define HPC_SM_AMPERE  1
#  endif
#  if __CUDA_ARCH__ >= 700
#    define HPC_SM_VOLTA   1
#  endif
#endif

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------
enum class Dtype : uint8_t {
    FP32  = 0,   // float
    FP16  = 1,   // half
    BF16  = 2,   // __nv_bfloat16  (Ampere+, better dynamic range than FP16)
};

// BF16 availability guard
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#  define HPC_BF16_DEVICE_SUPPORTED 0
#else
#  define HPC_BF16_DEVICE_SUPPORTED 1
#endif

// ---------------------------------------------------------------------------
// Optimizer hyperparameter bundles (POD — safe to pass to device)
// ---------------------------------------------------------------------------
struct SGDConfig {
    float lr           = 1e-2f;
    float momentum     = 0.9f;
    float weight_decay = 1e-4f;
    float dampening    = 0.0f;
    bool  nesterov     = true;
};

struct AdamConfig {
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 1e-2f;   // decoupled (AdamW) or L2 (Adam)
    bool  amsgrad      = false;
};

struct LAMBConfig {
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-6f;
    float weight_decay = 1e-2f;
    float clamp_value  = 10.0f;   // trust-ratio clamp
    bool  adam_w_mode  = true;    // decouple WD (LAMB-W)
};

struct LionConfig {
    float lr           = 1e-4f;  // Lion uses 3-10x smaller LR than Adam
    float beta1        = 0.9f;
    float beta2        = 0.99f;
    float weight_decay = 1e-2f;
};

struct GradClipConfig {
    float max_norm  = 1.0f;
    float norm_type = 2.0f;   // L2 only
};

// ---------------------------------------------------------------------------
// Runtime stats (host-accessible after cudaMemcpy or UVM mapping)
// ---------------------------------------------------------------------------
struct alignas(16) OptimizerStats {
    float    grad_norm_before = 0.0f;
    float    grad_norm_after  = 0.0f;
    float    param_norm       = 0.0f;   // used by LAMB for trust ratio
    uint32_t step             = 0;
    uint32_t skipped_steps    = 0;      // NaN/Inf gradients detected
    uint32_t rank             = 0;      // MPI rank that owns this stat
    float    last_lr          = 0.0f;
};

// ---------------------------------------------------------------------------
// Multi-precision tensor descriptor (non-owning view)
// ---------------------------------------------------------------------------
struct TensorView {
    void*   data   = nullptr;
    size_t  numel  = 0;         // number of elements (not bytes)
    Dtype   dtype  = Dtype::FP32;

    TensorView() = default;
    TensorView(float*             p, size_t n) : data(p), numel(n), dtype(Dtype::FP32) {}
    TensorView(half*              p, size_t n) : data(p), numel(n), dtype(Dtype::FP16) {}
    TensorView(__nv_bfloat16*     p, size_t n) : data(p), numel(n), dtype(Dtype::BF16) {}

    size_t byte_size() const {
        switch (dtype) {
            case Dtype::FP32: return numel * 4;
            case Dtype::FP16: return numel * 2;
            case Dtype::BF16: return numel * 2;
        }
        return 0;
    }
};

// ---------------------------------------------------------------------------
// Distributed context (single-node multi-GPU or multi-node via MPI)
// ---------------------------------------------------------------------------
struct DistConfig {
    int  rank          = 0;    // global MPI rank
    int  world_size    = 1;
    int  local_rank    = 0;    // GPU index on this node
    int  local_size    = 1;    // GPUs per node
    bool use_nccl      = false;
    bool use_fp16_comm = true; // compress all-reduce to FP16
};

// ---------------------------------------------------------------------------
// CUDA error macros
// ---------------------------------------------------------------------------
#define HPC_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "[hpc_opt] CUDA error @ %s:%d  %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            abort();                                                          \
        }                                                                     \
    } while (0)

#define HPC_CUDA_CHECK_KERNEL()                                               \
    do {                                                                      \
        cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "[hpc_opt] Kernel error @ %s:%d  %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            abort();                                                          \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// HPC kernel tuning constants
// ---------------------------------------------------------------------------
static constexpr int HPC_BLOCK       = 512;   // threads/block (fills A100 SMs)
static constexpr int HPC_WARP        = 32;
static constexpr int HPC_MAX_BLOCKS  = 1024;  // cap for large tensors

// Optimal blocks for a given element count
__host__ inline int hpc_blocks(size_t numel, int tpb = HPC_BLOCK) {
    int b = static_cast<int>((numel + tpb - 1) / tpb);
    return (b > HPC_MAX_BLOCKS) ? HPC_MAX_BLOCKS : b;
}

} // namespace hpc_opt
