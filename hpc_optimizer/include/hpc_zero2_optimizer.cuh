// =============================================================================
// HPC CUDA Optimizer Library — ZeRO-2 Extension
// hpc_zero2_optimizer.cuh  —  Sharded optimizer state + mixed-precision step
//
// This file owns the optimizer state (moments, master weights) only for
// the elements this rank is responsible for (its shard).
//
// Memory layout per rank (W = world_size, Ψ = total param elements):
//   m_shard[Ψ/W]          — FP32 first moment (Adam)
//   v_shard[Ψ/W]          — FP32 second moment (Adam)
//   master_shard[Ψ/W]     — FP32 master weight copy (for BF16/FP16 params)
//
// Step protocol:
//   1. ZeRO2Engine::pack_grads()            — pack tensors → flat FP32 bucket
//   2. ZeRO2Engine::reduce_scatter_grads()  — NCCL ReduceScatter → d_shard
//   3. ZeRO2Engine::wait_reduce_scatter()   — stream gate
//   4. ZeRO2ShardedOptimizer::step()        — AdamW on shard (grad + params)
//   5. ZeRO2Engine::commit_shard()          — write updated master → param_flat
//   6. ZeRO2Engine::all_gather_params()     — NCCL AllGather → unpack tensors
//
// Supported optimizers: AdamW, SGD, Lion (via template specialisation).
// Moments live only for Ψ/W elements — optimizer state per rank is 1/W
// the cost of non-ZeRO training.
//
// Mixed-precision:
//   The shard operates in FP32 internally regardless of param storage dtype.
//   master_shard tracks the high-precision copy and writes back to the
//   low-precision param_flat after each step.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_zero2.cuh"
#include "hpc_adam.cuh"
#include "hpc_sgd.cuh"
#include "hpc_lion.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <vector>
#include <cstdio>

namespace hpc_opt {
namespace zero2 {

// ===========================================================================
// Sharded AdamW/Adam kernel — operates on a single flat FP32 shard
//
// params_shard  — FP32 master weight shard (in/out)  [shard_size]
// grads_shard   — FP32 mean gradient shard (in)      [shard_size]
// m, v          — FP32 moment shards (in/out)         [shard_size]
// ===========================================================================
__global__ void k_adamw_shard_fp32(
        float* __restrict__ params_shard,
        const float* __restrict__ grads_shard,
        float* __restrict__ m,
        float* __restrict__ v,
        float* __restrict__ v_max,      // nullptr unless amsgrad
        size_t  shard_size,
        float   lr_eff,                 // lr * sqrt(1-β2^t) / (1-β1^t)
        float   lr,
        float   beta1,
        float   beta2,
        float   eps,
        float   weight_decay,
        bool    adamw,
        bool    amsgrad)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < shard_size;
         i += gridDim.x * blockDim.x)
    {
        float p = __ldg(params_shard + i);
        float g = __ldg(grads_shard  + i);

        // NaN/Inf guard
        if (!prec::is_finite(g)) continue;

        // L2 weight decay (non-decoupled / classic Adam)
        if (!adamw && weight_decay != 0.0f) g += weight_decay * p;

        // Moment update
        float mi = beta1 * __ldg(m + i) + (1.0f - beta1) * g;
        float vi = beta2 * __ldg(v + i) + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        // Denominator
        float denom;
        if (amsgrad) {
            float vmax = fmaxf(__ldg(v_max + i), vi);
            v_max[i] = vmax;
            denom = sqrtf(vmax) + eps;
        } else {
            denom = sqrtf(vi) + eps;
        }

        // Update
        p -= lr_eff * mi / denom;

        // Decoupled weight decay (AdamW)
        if (adamw && weight_decay != 0.0f) p -= lr * weight_decay * p;

        params_shard[i] = p;
    }
}

// Vectorised float4 path — 4 elements/thread
__global__ void k_adamw_shard_vec4(
        float* __restrict__ ps,
        const float* __restrict__ gs,
        float* __restrict__ m,
        float* __restrict__ v,
        size_t  shard4,
        float   lr_eff,
        float   lr,
        float   beta1,
        float   beta2,
        float   eps,
        float   weight_decay,
        bool    adamw)
{
    for (size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
         i4 < shard4;
         i4 += gridDim.x * blockDim.x)
    {
        size_t b = i4 * 4;
        float4 p4 = *reinterpret_cast<const float4*>(ps + b);
        float4 g4 = *reinterpret_cast<const float4*>(gs + b);
        float4 m4 = *reinterpret_cast<const float4*>(m  + b);
        float4 v4 = *reinterpret_cast<const float4*>(v  + b);

#define SHARD_STEP(x)                                                    \
        do {                                                             \
            if (!adamw) g4.x += weight_decay * p4.x;                   \
            m4.x = beta1 * m4.x + (1.0f - beta1) * g4.x;               \
            v4.x = beta2 * v4.x + (1.0f - beta2) * g4.x * g4.x;       \
            p4.x -= lr_eff * m4.x / (sqrtf(v4.x) + eps);               \
            if (adamw && weight_decay != 0.0f)                          \
                p4.x -= lr * weight_decay * p4.x;                       \
        } while (0)

        SHARD_STEP(x); SHARD_STEP(y); SHARD_STEP(z); SHARD_STEP(w);
#undef SHARD_STEP

        *reinterpret_cast<float4*>(ps + b) = p4;
        *reinterpret_cast<float4*>(m  + b) = m4;
        *reinterpret_cast<float4*>(v  + b) = v4;
    }
}

// ===========================================================================
// Sharded SGD kernel
// ===========================================================================
__global__ void k_sgd_shard_vec4(
        float* __restrict__ ps,
        const float* __restrict__ gs,
        float* __restrict__ vel,
        size_t  shard4,
        float   lr,
        float   momentum,
        float   weight_decay,
        bool    nesterov)
{
    for (size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
         i4 < shard4;
         i4 += gridDim.x * blockDim.x)
    {
        size_t b = i4 * 4;
        float4 p4 = *reinterpret_cast<const float4*>(ps  + b);
        float4 g4 = *reinterpret_cast<const float4*>(gs  + b);
        float4 v4 = *reinterpret_cast<const float4*>(vel + b);

#define SGD_STEP(x)                                                       \
        do {                                                              \
            g4.x += weight_decay * p4.x;                                 \
            v4.x = momentum * v4.x + g4.x;                               \
            float eff = nesterov ? (g4.x + momentum * v4.x) : v4.x;     \
            p4.x -= lr * eff;                                            \
        } while (0)

        SGD_STEP(x); SGD_STEP(y); SGD_STEP(z); SGD_STEP(w);
#undef SGD_STEP

        *reinterpret_cast<float4*>(ps  + b) = p4;
        *reinterpret_cast<float4*>(vel + b) = v4;
    }
}

// ===========================================================================
// Sharded Lion kernel
// ===========================================================================
__global__ void k_lion_shard_vec4(
        float* __restrict__ ps,
        const float* __restrict__ gs,
        float* __restrict__ m,
        size_t  shard4,
        float   lr,
        float   beta1,
        float   beta2,
        float   weight_decay)
{
    const float one_m_b1 = 1.0f - beta1;
    const float one_m_b2 = 1.0f - beta2;

    for (size_t i4 = blockIdx.x * blockDim.x + threadIdx.x;
         i4 < shard4;
         i4 += gridDim.x * blockDim.x)
    {
        size_t b = i4 * 4;
        float4 p4 = *reinterpret_cast<const float4*>(ps + b);
        float4 g4 = *reinterpret_cast<const float4*>(gs + b);
        float4 m4 = *reinterpret_cast<const float4*>(m  + b);

#define LION_STEP(x)                                                      \
        do {                                                              \
            float interp = beta1 * m4.x + one_m_b1 * g4.x;              \
            float upd    = (interp > 0.0f) - (interp < 0.0f);           \
            p4.x = p4.x - lr * (upd + weight_decay * p4.x);             \
            m4.x = beta2 * m4.x + one_m_b2 * g4.x;                      \
        } while (0)

        LION_STEP(x); LION_STEP(y); LION_STEP(z); LION_STEP(w);
#undef LION_STEP

        *reinterpret_cast<float4*>(ps + b) = p4;
        *reinterpret_cast<float4*>(m  + b) = m4;
    }
}

// ===========================================================================
// ZeRO2ShardedOptimizer<ConfigT>
//   Template on config type. Internally stores only the moment/velocity
//   buffers for this rank's shard (Ψ/W elements).
// ===========================================================================
enum class ShardedOptKind { AdamW, Adam, SGD, Lion };

class ZeRO2ShardedOptimizer {
public:
    // -----------------------------------------------------------------------
    // Construction — choose optimizer variant at runtime
    // -----------------------------------------------------------------------
    explicit ZeRO2ShardedOptimizer(ShardedOptKind kind) : kind_(kind) {}
    ~ZeRO2ShardedOptimizer() { free_state(); }

    ZeRO2ShardedOptimizer(const ZeRO2ShardedOptimizer&)            = delete;
    ZeRO2ShardedOptimizer& operator=(const ZeRO2ShardedOptimizer&) = delete;

    // Common hyperparameters exposed directly
    float lr           = 3e-4f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 1e-2f;
    float momentum     = 0.9f;   // SGD
    bool  nesterov     = true;   // SGD
    bool  amsgrad      = false;  // Adam

    // -----------------------------------------------------------------------
    // init  —  allocate sharded state buffers
    //   shard_size: number of elements this rank owns
    //   master_fp32_shard: caller-provided FP32 master weight shard buffer
    //                      (points into ZeRO2Engine::param_flat()[rank_lo])
    //   stream: CUDA stream for init memset
    // -----------------------------------------------------------------------
    void init(size_t shard_size, float* master_fp32_shard,
              cudaStream_t stream = 0)
    {
        free_state();
        shard_size_  = shard_size;
        master_      = master_fp32_shard;

        size_t bytes = shard_size * sizeof(float);

        switch (kind_) {
            case ShardedOptKind::AdamW:
            case ShardedOptKind::Adam:
                HPC_CUDA_CHECK(cudaMalloc(&m_,   bytes));
                HPC_CUDA_CHECK(cudaMalloc(&v_,   bytes));
                HPC_CUDA_CHECK(cudaMemsetAsync(m_, 0, bytes, stream));
                HPC_CUDA_CHECK(cudaMemsetAsync(v_, 0, bytes, stream));
                if (amsgrad) {
                    HPC_CUDA_CHECK(cudaMalloc(&v_max_, bytes));
                    HPC_CUDA_CHECK(cudaMemsetAsync(v_max_, 0, bytes, stream));
                }
                break;
            case ShardedOptKind::SGD:
                HPC_CUDA_CHECK(cudaMalloc(&vel_, bytes));
                HPC_CUDA_CHECK(cudaMemsetAsync(vel_, 0, bytes, stream));
                break;
            case ShardedOptKind::Lion:
                HPC_CUDA_CHECK(cudaMalloc(&m_,   bytes));
                HPC_CUDA_CHECK(cudaMemsetAsync(m_, 0, bytes, stream));
                break;
        }

        initialized_ = true;
    }

    // -----------------------------------------------------------------------
    // step  —  one optimizer step on this rank's gradient shard
    //   grad_shard: FP32 mean gradient shard from ZeRO2Engine (Ψ/W elements)
    //   stream:     compute stream
    //
    // After this call, master_fp32_shard (passed at init) holds the updated
    // parameter values for this rank's shard. ZeRO2Engine::commit_shard()
    // then copies these into param_flat for AllGather.
    // -----------------------------------------------------------------------
    void step(const float* grad_shard, cudaStream_t stream = 0) {
        assert(initialized_);
        step_++;

        switch (kind_) {
            case ShardedOptKind::AdamW:
            case ShardedOptKind::Adam: {
                float bc1    = 1.0f - powf(beta1, static_cast<float>(step_));
                float bc2    = 1.0f - powf(beta2, static_cast<float>(step_));
                float lr_eff = lr * sqrtf(bc2) / bc1;
                bool  adamw  = (kind_ == ShardedOptKind::AdamW);

                // Prefer vec4 when aligned
                if (shard_size_ % 4 == 0 && !amsgrad) {
                    int blk = hpc_blocks(shard_size_ / 4);
                    k_adamw_shard_vec4<<<blk, HPC_BLOCK, 0, stream>>>(
                        master_, grad_shard,
                        m_, v_,
                        shard_size_ / 4,
                        lr_eff, lr, beta1, beta2, eps,
                        weight_decay, adamw);
                } else {
                    int blk = hpc_blocks(shard_size_);
                    k_adamw_shard_fp32<<<blk, HPC_BLOCK, 0, stream>>>(
                        master_, grad_shard,
                        m_, v_, v_max_,
                        shard_size_,
                        lr_eff, lr, beta1, beta2, eps,
                        weight_decay, adamw, amsgrad);
                }
                break;
            }
            case ShardedOptKind::SGD: {
                if (shard_size_ % 4 == 0) {
                    int blk = hpc_blocks(shard_size_ / 4);
                    k_sgd_shard_vec4<<<blk, HPC_BLOCK, 0, stream>>>(
                        master_, grad_shard, vel_,
                        shard_size_ / 4,
                        lr, momentum, weight_decay, nesterov);
                }
                break;
            }
            case ShardedOptKind::Lion: {
                if (shard_size_ % 4 == 0) {
                    int blk = hpc_blocks(shard_size_ / 4);
                    k_lion_shard_vec4<<<blk, HPC_BLOCK, 0, stream>>>(
                        master_, grad_shard, m_,
                        shard_size_ / 4,
                        lr, beta1, beta2, weight_decay);
                }
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Checkpoint helpers — save/load this rank's sharded state
    // -----------------------------------------------------------------------
    void save_shard_state(const std::string& path, int rank) const {
        if (rank != 0) return;  // each rank saves its own file separately

        std::string rpath = path + ".rank" + std::to_string(rank);
        FILE* f = fopen(rpath.c_str(), "wb");
        if (!f) return;

        fwrite(&step_, sizeof(step_), 1, f);
        fwrite(&shard_size_, sizeof(shard_size_), 1, f);

        auto write_buf = [&](float* d, size_t n) {
            if (!d) { size_t z = 0; fwrite(&z, sizeof(z), 1, f); return; }
            fwrite(&n, sizeof(n), 1, f);
            std::vector<float> h(n);
            cudaMemcpy(h.data(), d, n*sizeof(float), cudaMemcpyDeviceToHost);
            fwrite(h.data(), sizeof(float), n, f);
        };

        write_buf(m_,     shard_size_);
        write_buf(v_,     shard_size_);
        write_buf(v_max_, v_max_ ? shard_size_ : 0);
        write_buf(vel_,   vel_   ? shard_size_ : 0);
        write_buf(master_, shard_size_);
        fclose(f);
    }

    uint32_t load_shard_state(const std::string& path, int rank,
                               cudaStream_t stream = 0)
    {
        std::string rpath = path + ".rank" + std::to_string(rank);
        FILE* f = fopen(rpath.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "[ZeRO2] checkpoint not found: %s\n", rpath.c_str());
            return 0;
        }

        uint32_t saved_step; size_t saved_shard;
        fread(&saved_step, sizeof(saved_step), 1, f);
        fread(&saved_shard, sizeof(saved_shard), 1, f);

        if (saved_shard != shard_size_) {
            fprintf(stderr, "[ZeRO2] shard size mismatch %zu != %zu\n",
                    saved_shard, shard_size_);
            fclose(f); return 0;
        }

        auto read_buf = [&](float* d) {
            size_t n; fread(&n, sizeof(n), 1, f);
            if (n == 0 || !d) { fseek(f, (long)(n*4), SEEK_CUR); return; }
            std::vector<float> h(n);
            fread(h.data(), sizeof(float), n, f);
            cudaMemcpyAsync(d, h.data(), n*sizeof(float),
                            cudaMemcpyHostToDevice, stream);
        };

        read_buf(m_);
        read_buf(v_);
        read_buf(v_max_);
        read_buf(vel_);
        read_buf(master_);
        fclose(f);

        step_ = saved_step;
        return saved_step;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    uint32_t    step_count()  const { return step_; }
    size_t      shard_size()  const { return shard_size_; }
    float*      m_buf()       const { return m_; }
    float*      v_buf()       const { return v_; }
    float*      vel_buf()     const { return vel_; }
    float*      master_buf()  const { return master_; }
    ShardedOptKind kind()     const { return kind_; }

    // State memory bytes this rank holds (excludes full params)
    size_t state_bytes() const {
        size_t n = shard_size_ * 4UL;
        switch (kind_) {
            case ShardedOptKind::AdamW:
            case ShardedOptKind::Adam:
                return n * (2 + (v_max_ ? 1 : 0) + 1); // m+v+(v_max)+master
            case ShardedOptKind::SGD:   return n * 2;   // vel + master
            case ShardedOptKind::Lion:  return n * 2;   // m   + master
        }
        return 0;
    }

private:
    ShardedOptKind kind_;
    bool      initialized_ = false;
    size_t    shard_size_  = 0;
    uint32_t  step_        = 0;

    float*  m_      = nullptr;   // first moment   (AdamW/Adam/Lion)
    float*  v_      = nullptr;   // second moment  (AdamW/Adam)
    float*  v_max_  = nullptr;   // AMSGrad max    (optional)
    float*  vel_    = nullptr;   // velocity       (SGD)
    float*  master_ = nullptr;   // FP32 master    (external, not owned)

    void free_state() {
        if (m_)     { cudaFree(m_);     m_     = nullptr; }
        if (v_)     { cudaFree(v_);     v_     = nullptr; }
        if (v_max_) { cudaFree(v_max_); v_max_ = nullptr; }
        if (vel_)   { cudaFree(vel_);   vel_   = nullptr; }
        // master_ is external — do not free
        initialized_ = false;
    }
};

// ===========================================================================
// ZeRO2Trainer  —  complete all-in-one ZeRO-2 training object
//
// Owns the ZeRO2Engine + ZeRO2ShardedOptimizer and exposes a single
// step() call that executes the entire ZeRO-2 protocol:
//   pack → reduce_scatter → local_opt_step → commit → all_gather → unpack
//
// Usage:
//   ZeRO2Trainer trainer;
//   trainer.init(params, n, comm, adamw_config, stream);
//
//   for each iteration:
//       backward();            // fill grads (any dtype)
//       trainer.step(params, grads, n);
// ===========================================================================
struct ZeRO2TrainerConfig {
    ShardedOptKind kind         = ShardedOptKind::AdamW;
    float          lr           = 3e-4f;
    float          beta1        = 0.9f;
    float          beta2        = 0.95f;   // GPT-3 default
    float          eps          = 1e-8f;
    float          weight_decay = 0.1f;
    float          max_grad_norm= 1.0f;    // 0 = disabled
    bool           amsgrad      = false;
};

class ZeRO2Trainer {
public:
    ZeRO2Trainer() = default;
    ~ZeRO2Trainer() = default;

    ZeRO2Trainer(const ZeRO2Trainer&)            = delete;
    ZeRO2Trainer& operator=(const ZeRO2Trainer&) = delete;

    // -----------------------------------------------------------------------
    // init
    // -----------------------------------------------------------------------
    void init(const TensorView* params, int n,
              CommContext&           comm,
              const ZeRO2TrainerConfig& cfg,
              cudaStream_t          compute_stream,
              cudaStream_t          comm_stream = nullptr)
    {
        cfg_            = cfg;
        compute_stream_ = compute_stream;
        comm_stream_    = comm_stream ? comm_stream : compute_stream;
        n_tensors_      = n;

        // Initialise ZeRO2Engine
        engine_.init(params, n, comm, compute_stream_, comm_stream_);

        // Allocate per-tensor numel list
        numel_list_.resize(n);
        for (int i = 0; i < n; ++i) numel_list_[i] = params[i].numel;

        // Initialise sharded optimizer
        // master_fp32_shard = engine_.param_shard() which points into
        // param_flat[rank_lo .. rank_hi)  — already allocated by engine
        opt_ = std::make_unique<ZeRO2ShardedOptimizer>(cfg.kind);
        opt_->lr           = cfg.lr;
        opt_->beta1        = cfg.beta1;
        opt_->beta2        = cfg.beta2;
        opt_->eps          = cfg.eps;
        opt_->weight_decay = cfg.weight_decay;
        opt_->amsgrad      = cfg.amsgrad;
        opt_->init(engine_.shard_numel(), engine_.param_shard(),
                   compute_stream_);

        // Copy initial param values into master FP32 shard
        _init_master_from_params(params, n, compute_stream_);

        // Allocate grad-clip scratch (reuse for per-shard norm)
        HPC_CUDA_CHECK(cudaMalloc(&d_shard_sq_, sizeof(float)));

        if (comm.is_root()) {
            fprintf(stdout,
                "[ZeRO2Trainer] optimizer=%s  lr=%.2e  wd=%.3f  "
                "max_norm=%.1f\n",
                _kind_name(cfg.kind), cfg.lr, cfg.weight_decay,
                cfg.max_grad_norm);
            engine_.print_memory_report();
        }
    }

    // -----------------------------------------------------------------------
    // step  —  full ZeRO-2 protocol
    // -----------------------------------------------------------------------
    void step(TensorView* params, TensorView* grads, int n) {
        assert(engine_.initialized());

        // 1. Pack all gradient tensors → flat FP32 bucket
        engine_.pack_grads(grads, n, compute_stream_);

        // 2. ReduceScatter  (NCCL on comm_stream)
        engine_.reduce_scatter_grads(comm_stream_);

        // 3. Gate compute on ReduceScatter
        engine_.wait_reduce_scatter(compute_stream_);

        // 4. Optional: clip shard-local gradient norm
        if (cfg_.max_grad_norm > 0.0f)
            _clip_shard_grad(engine_.grad_shard(),
                             engine_.shard_numel(), compute_stream_);

        // 5. Local optimizer step (operates on master weight shard)
        opt_->step(engine_.grad_shard(), compute_stream_);

        // 6. Commit updated master shard into param_flat at rank offset
        engine_.commit_shard(engine_.param_shard(), compute_stream_);

        // 7. AllGather + unpack into individual param tensors
        engine_.all_gather_params(params, n,
                                  compute_stream_, comm_stream_);

        step_++;
        stats_.step = step_;
        stats_.last_lr = cfg_.lr;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    uint32_t          step_count()     const { return step_; }
    float             current_lr()     const { return cfg_.lr; }
    float&            lr()                   { return cfg_.lr; }
    const ShardLayout& layout()        const { return engine_.layout(); }
    OptimizerStats&   stats()                { return stats_; }
    ZeRO2Engine&      engine()               { return engine_; }

    void log(int step, float loss = -1.0f, bool root = true) const {
        if (!root) return;
        if (loss >= 0.0f)
            fprintf(stdout,
                "[ZeRO2] step=%-6d  lr=%.2e  loss=%.4f  "
                "shard_grad_norm=%.3f\n",
                step, cfg_.lr, loss, stats_.grad_norm_after);
        else
            fprintf(stdout,
                "[ZeRO2] step=%-6d  lr=%.2e  "
                "shard_grad_norm=%.3f\n",
                step, cfg_.lr, stats_.grad_norm_after);
    }

private:
    ZeRO2TrainerConfig cfg_;
    ZeRO2Engine        engine_;
    std::unique_ptr<ZeRO2ShardedOptimizer> opt_;
    OptimizerStats     stats_{};

    cudaStream_t  compute_stream_ = nullptr;
    cudaStream_t  comm_stream_    = nullptr;
    int           n_tensors_      = 0;
    uint32_t      step_           = 0;
    std::vector<size_t> numel_list_;

    float*  d_shard_sq_ = nullptr;  // scratch for shard-local grad norm

    // -----------------------------------------------------------------------
    // Initialise FP32 master shard by reading from params
    // -----------------------------------------------------------------------
    void _init_master_from_params(const TensorView* params, int n,
                                  cudaStream_t stream)
    {
        float* master = engine_.param_shard();
        const auto& layout = engine_.layout();

        for (int i = 0; i < n; ++i) {
            const auto& s = layout.shards[i];
            if (s.shard_hi <= s.shard_lo) continue;

            size_t lo       = s.shard_lo;
            size_t cnt      = s.shard_hi - s.shard_lo;
            size_t flat_off = s.global_offset + lo - layout.rank_lo;

            // Cast param slice → FP32 into master shard
            switch (params[i].dtype) {
                case Dtype::FP32: {
                    const float* src = static_cast<const float*>(params[i].data) + lo;
                    HPC_CUDA_CHECK(cudaMemcpyAsync(master + flat_off, src,
                                                   cnt * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream));
                    break;
                }
                case Dtype::FP16: {
                    const half* src = static_cast<const half*>(params[i].data) + lo;
                    int blk = hpc_blocks(cnt);
                    k_pack_tensor<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        master, src, flat_off, cnt);
                    break;
                }
                case Dtype::BF16: {
                    const __nv_bfloat16* src =
                        static_cast<const __nv_bfloat16*>(params[i].data) + lo;
                    int blk = hpc_blocks(cnt);
                    k_pack_tensor<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        master, src, flat_off, cnt);
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Clip the gradient shard by its local L2 norm
    // (shard-local clip — does NOT sync across ranks; use pre-clip if needed)
    // -----------------------------------------------------------------------
    void _clip_shard_grad(float* grad, size_t n, cudaStream_t stream) {
        if (!d_shard_sq_) return;

        HPC_CUDA_CHECK(cudaMemsetAsync(d_shard_sq_, 0, sizeof(float), stream));

        // Reuse partial norm accumulator from hpc_grad_clip.cuh
        int blk = hpc_blocks(n);
        uint32_t dummy_nans = 0;
        uint32_t* d_nans;
        HPC_CUDA_CHECK(cudaMalloc(&d_nans, sizeof(uint32_t)));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_nans, 0, sizeof(uint32_t), stream));

        k_partial_sq_norm<float><<<blk, HPC_BLOCK, 0, stream>>>(
            grad, d_shard_sq_, n, d_nans);

        float sq = 0.0f;
        HPC_CUDA_CHECK(cudaMemcpyAsync(&sq, d_shard_sq_, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        HPC_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_nans);

        float norm = sqrtf(sq);
        stats_.grad_norm_after = norm;

        if (norm > cfg_.max_grad_norm && norm > 0.0f) {
            float scale = cfg_.max_grad_norm / (norm + 1e-6f);
            int blk2 = hpc_blocks(n);
            k_apply_scale<float><<<blk2, HPC_BLOCK, 0, stream>>>(
                grad, scale, n);
        }
    }

    static const char* _kind_name(ShardedOptKind k) {
        switch (k) {
            case ShardedOptKind::AdamW: return "AdamW";
            case ShardedOptKind::Adam:  return "Adam";
            case ShardedOptKind::SGD:   return "SGD";
            case ShardedOptKind::Lion:  return "Lion";
        }
        return "?";
    }
};

} // namespace zero2
} // namespace hpc_opt
