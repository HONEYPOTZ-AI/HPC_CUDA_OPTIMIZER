// =============================================================================
// HPC CUDA Optimizer Library
// hpc_optimizer.cuh  –  Unified HPC Optimizer facade
//
// Single include that wires together all optimizers, the comm layer,
// gradient clipper, checkpoint I/O, NVTX profiler, and LR schedulers.
//
// Design goals:
//   • Minimal per-step overhead – one virtual dispatch max (scheduler)
//   • All heavy work deferred to streams; no implicit sync on hot path
//   • Distributed-aware: all_reduce before step; ZeRO-1 optional
//   • Checkpointing on any schedule (every N steps, on signal, etc.)
//
// Typical usage (single GPU, FP32 AdamW):
//   hpc_opt::HPCOptimizer<hpc_opt::AdamOptimizer> opt(
//       hpc_opt::make_adamw_config(3e-4f), stream);
//   opt.init(params, n);
//   for (int s = 0; s < N; ++s) {
//       backward();
//       opt.step(params, grads, n);   // clip + all_reduce + optimizer
//   }
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_grad_clip.cuh"
#include "hpc_adam.cuh"
#include "hpc_sgd.cuh"
#include "hpc_lamb.cuh"
#include "hpc_lion.cuh"
#include "hpc_comm.cuh"
#include "hpc_checkpoint.cuh"
#include "hpc_profiler.cuh"
#include "hpc_lr_scheduler.cuh"

#include <memory>
#include <functional>
#include <string>
#include <cstdio>

namespace hpc_opt {

// ---------------------------------------------------------------------------
// Config factory helpers
// ---------------------------------------------------------------------------
inline AdamConfig make_adam_config(float lr = 1e-3f, float wd = 0.0f) {
    AdamConfig c; c.lr = lr; c.weight_decay = wd; return c;
}
inline AdamConfig make_adamw_config(float lr = 3e-4f, float wd = 1e-2f) {
    AdamConfig c; c.lr = lr; c.weight_decay = wd; return c;
}
inline SGDConfig make_sgd_config(float lr = 1e-2f, float mom = 0.9f) {
    SGDConfig c; c.lr = lr; c.momentum = mom; return c;
}
inline LAMBConfig make_lamb_config(float lr = 1e-3f, float wd = 1e-2f) {
    LAMBConfig c; c.lr = lr; c.weight_decay = wd; return c;
}
inline LionConfig make_lion_config(float lr = 1e-4f, float wd = 1e-2f) {
    LionConfig c; c.lr = lr; c.weight_decay = wd; return c;
}

// ---------------------------------------------------------------------------
// Optimizer type enum (for factory / display)
// ---------------------------------------------------------------------------
enum class OptimizerKind { Adam, AdamW, SGD, LAMB, Lion };

inline const char* optimizer_name(OptimizerKind k) {
    switch(k) {
        case OptimizerKind::Adam:  return "Adam";
        case OptimizerKind::AdamW: return "AdamW";
        case OptimizerKind::SGD:   return "SGD";
        case OptimizerKind::LAMB:  return "LAMB";
        case OptimizerKind::Lion:  return "Lion";
    }
    return "Unknown";
}

// ===========================================================================
// HPCOptimizer<BackendT>
//   BackendT is one of: AdamOptimizer, SGDOptimizer, LAMBOptimizer, LionOptimizer
// ===========================================================================
template<typename BackendT, typename ConfigT>
class HPCOptimizer {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    HPCOptimizer(const ConfigT& cfg,
                 cudaStream_t   stream      = 0,
                 CommContext*   comm        = nullptr,   // nullptr = single GPU
                 float**        master_fp32 = nullptr)  // for mixed-precision
        : backend_(cfg), stream_(stream), comm_(comm), master_fp32_(master_fp32)
    {}

    ~HPCOptimizer() = default;

    HPCOptimizer(const HPCOptimizer&)            = delete;
    HPCOptimizer& operator=(const HPCOptimizer&) = delete;

    // -----------------------------------------------------------------------
    // init  –  allocate optimizer state; must be called once before step().
    // -----------------------------------------------------------------------
    void init(const TensorView* params, int n) {
        n_tensors_  = n;
        params_ref_ = params;

        backend_.init(params, n, master_fp32_, stream_);

        // Cache per-tensor numel for checkpoint helpers
        numel_list_.resize(n);
        for (int i = 0; i < n; ++i) numel_list_[i] = params[i].numel;

        HPC_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // -----------------------------------------------------------------------
    // step  –  one full optimiser step:
    //   (1) optional grad clipping
    //   (2) NCCL all-reduce (if distributed)
    //   (3) backend optimizer kernel
    //   (4) optional LR schedule advance
    //   (5) optional checkpoint
    // -----------------------------------------------------------------------
    void step(TensorView* params, TensorView* grads, int n) {
        HPC_RANGE_SCOPED("hpc_opt::step", 0xFF2196F3);

        // ---- Gradient clipping ----
        if (clip_enabled_) {
            HPC_RANGE_PUSH("grad_clip", 0xFFFFC107);
            clipper_.clip(grads, n, clip_cfg_, stats_, stream_,
                          (comm_ && comm_->initialized()) ? nullptr : nullptr);
            HPC_RANGE_POP();
        }

        // ---- All-reduce gradients across GPUs ----
        if (comm_ && comm_->initialized() && comm_->world_size() > 1) {
            HPC_RANGE_PUSH("nccl_allreduce", 0xFF4CAF50);
            comm_->all_reduce_grads(grads, n, /*fp16_compress=*/fp16_comm_, stream_);
            HPC_RANGE_POP();
        }

        // ---- Optimizer kernel ----
        {
            HPC_RANGE_PUSH("optimizer_kernel", 0xFF2196F3);
            backend_.step(params, grads, n, stream_);
            HPC_RANGE_POP();
        }

        // ---- Update stats ----
        stats_.step++;
        stats_.last_lr = current_lr();

        // ---- LR scheduler ----
        if (scheduler_) scheduler_->step();

        // ---- Periodic checkpoint ----
        if (ckpt_every_ > 0 && ckpt_rank0_path_.size() > 0
            && stats_.step % ckpt_every_ == 0)
        {
            save_checkpoint(ckpt_rank0_path_
                            + "/ckpt_step"
                            + std::to_string(stats_.step)
                            + ".bin");
        }
    }

    // -----------------------------------------------------------------------
    // Gradient clip config
    // -----------------------------------------------------------------------
    void enable_clipping(float max_norm = 1.0f, bool fp16_comm = true) {
        clip_enabled_    = true;
        clip_cfg_.max_norm = max_norm;
        fp16_comm_       = fp16_comm;
    }

    // -----------------------------------------------------------------------
    // LR scheduler
    // -----------------------------------------------------------------------
    void set_scheduler(std::unique_ptr<LRScheduler> s) {
        scheduler_ = std::move(s);
    }

    LRScheduler* lr_scheduler() { return scheduler_.get(); }

    // -----------------------------------------------------------------------
    // Checkpoint
    // -----------------------------------------------------------------------
    void set_checkpoint_dir(const std::string& dir, int every_n_steps) {
        ckpt_rank0_path_ = dir;
        ckpt_every_      = every_n_steps;
    }

    void save_checkpoint(const std::string& path) {
        int rank = comm_ ? comm_->rank() : 0;
        CheckpointIO::save(path,
                           backend_.m_buffers(),
                           numel_list_.data(),
                           n_tensors_,
                           n_tensors_,
                           stats_.step,
                           &backend_.config(),
                           sizeof(backend_.config()),
                           rank);
    }

    uint32_t load_checkpoint(const std::string& path) {
        uint32_t s = CheckpointIO::load(path,
                                        backend_.m_buffers(),
                                        numel_list_.data(),
                                        n_tensors_,
                                        &backend_.config(),
                                        sizeof(backend_.config()),
                                        stream_);
        stats_.step = s;
        return s;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    float          current_lr() const { return backend_.config().lr; }
    uint32_t       step_count() const { return stats_.step; }
    OptimizerStats& stats()           { return stats_; }
    BackendT&       backend()         { return backend_; }
    ConfigT&        config()          { return backend_.config(); }

    void zero_grad(TensorView* grads, int n) {
        for (int i = 0; i < n; ++i) {
            HPC_CUDA_CHECK(cudaMemsetAsync(grads[i].data, 0,
                                           grads[i].byte_size(), stream_));
        }
    }

    // Print a one-line status (rank-0 only)
    void log_step(int step, float loss = -1.0f) const {
        bool root = !comm_ || comm_->is_root();
        if (!root) return;

        if (loss >= 0.0f) {
            fprintf(stdout,
                "[opt] step=%-6d  lr=%.2e  loss=%.4f  "
                "grad_norm=%.3f→%.3f  skipped=%u\n",
                step, current_lr(), loss,
                stats_.grad_norm_before, stats_.grad_norm_after,
                stats_.skipped_steps);
        } else {
            fprintf(stdout,
                "[opt] step=%-6d  lr=%.2e  "
                "grad_norm=%.3f→%.3f  skipped=%u\n",
                step, current_lr(),
                stats_.grad_norm_before, stats_.grad_norm_after,
                stats_.skipped_steps);
        }
    }

private:
    BackendT       backend_;
    cudaStream_t   stream_       = nullptr;
    CommContext*   comm_         = nullptr;
    float**        master_fp32_  = nullptr;
    int            n_tensors_    = 0;
    const TensorView* params_ref_= nullptr;

    GradClipper    clipper_;
    GradClipConfig clip_cfg_;
    bool           clip_enabled_ = false;
    bool           fp16_comm_    = true;

    OptimizerStats stats_{};

    std::unique_ptr<LRScheduler> scheduler_;

    std::string  ckpt_rank0_path_;
    int          ckpt_every_ = 0;

    std::vector<size_t> numel_list_;
};

// ---------------------------------------------------------------------------
// Convenience type aliases for common HPC setups
// ---------------------------------------------------------------------------
using AdamWOptimizer = HPCOptimizer<AdamOptimizer, AdamConfig>;
using PlainAdamOpt   = HPCOptimizer<AdamOptimizer, AdamConfig>;
using HPC_SGD        = HPCOptimizer<SGDOptimizer,  SGDConfig>;
using HPC_LAMB       = HPCOptimizer<LAMBOptimizer, LAMBConfig>;
using HPC_Lion       = HPCOptimizer<LionOptimizer, LionConfig>;

// ---------------------------------------------------------------------------
// make_* factory functions
// ---------------------------------------------------------------------------
inline std::unique_ptr<AdamWOptimizer>
make_adamw(float lr = 3e-4f, float wd = 1e-2f,
           cudaStream_t stream = 0, CommContext* comm = nullptr)
{
    return std::make_unique<AdamWOptimizer>(
        make_adamw_config(lr, wd), stream, comm);
}

inline std::unique_ptr<HPC_SGD>
make_sgd(float lr = 1e-2f, float mom = 0.9f,
         cudaStream_t stream = 0, CommContext* comm = nullptr)
{
    return std::make_unique<HPC_SGD>(
        make_sgd_config(lr, mom), stream, comm);
}

inline std::unique_ptr<HPC_LAMB>
make_lamb(float lr = 1e-3f, float wd = 1e-2f,
          cudaStream_t stream = 0, CommContext* comm = nullptr)
{
    return std::make_unique<HPC_LAMB>(
        make_lamb_config(lr, wd), stream, comm);
}

inline std::unique_ptr<HPC_Lion>
make_lion(float lr = 1e-4f, float wd = 1e-2f,
          cudaStream_t stream = 0, CommContext* comm = nullptr)
{
    return std::make_unique<HPC_Lion>(
        make_lion_config(lr, wd), stream, comm);
}

// ---------------------------------------------------------------------------
// Print device + library info on startup
// ---------------------------------------------------------------------------
inline void print_hpc_banner(const DistConfig& dist = {}) {
    int dev = dist.local_rank;
    cudaDeviceProp p;
    HPC_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    // TF32 is enabled by default on Ampere+; confirm with cublas flag if present
    bool tf32 = (p.major >= 8);

    fprintf(stdout,
        "=================================================================\n"
        " HPC CUDA Optimizer Library\n"
        " Device [%d]: %s  |  SM %d.%d  |  %.1f GB HBM\n"
        " Precision: FP32 | FP16 | BF16 (Ampere+=%d) | TF32 auto (=%d)\n"
        " rank=%d  world_size=%d  local_rank=%d\n"
        "=================================================================\n",
        dev, p.name, p.major, p.minor,
        static_cast<double>(p.totalGlobalMem) / (1 << 30),
        (p.major >= 8), tf32,
        dist.rank, dist.world_size, dist.local_rank);
}

} // namespace hpc_opt
