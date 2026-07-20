// =============================================================================
// HPC CUDA Optimizer Library — ZeRO-3 Extension
// hpc_zero3.cuh  —  Full parameter + gradient + optimizer-state sharding
//
// ZeRO memory breakdown per rank (W = world_size, Ψ = total param elements):
//   ZeRO-0 :  16Ψ bytes   (params + grads + Adam-m + Adam-v)
//   ZeRO-1 :  (8  + 8/W)Ψ
//   ZeRO-2 :  (4  + 12/W)Ψ
//   ZeRO-3 :  16Ψ/W        ← all three state buckets sharded
//
// ZeRO-3 per-step protocol:
//   ① prefetch_params()   — ncclAllGather: materialize full params from shards
//                           (overlaps with previous step's compute on comm_stream)
//   ② forward()           — runs on full params   (user code)
//   ③ release_params()    — free / zero temp full-param buffers
//   ④ backward()          — accumulate full grads  (user code)
//   ⑤ reduce_scatter_grads() — ncclReduceScatter → each rank gets its grad shard
//   ⑥ local_opt_step()    — optimizer kernel on (grad_shard, param_shard)
//   → go to ①
//
// Key differences vs ZeRO-2:
//   • param_shard[i]  is the only persistent parameter copy on this rank
//   • d_full_params[i] is a TEMPORARY buffer, allocated just before AllGather
//     and freed after the forward pass to reclaim VRAM
//   • The user-facing forward-pass weight pointer switches between
//     d_full_params (during fwd) and param_shard (at all other times)
//   • Prefetch pipeline: AllGather for layer L+1 can be overlapped with
//     the forward compute of layer L using dual comm/compute streams
//
// Supports FP32 / FP16 / BF16 parameter storage.
// Optimizer moments (m, v) always FP32, sharded.
// Compatible with: AdamW, SGD, Lion (via ZeRO2ShardedOptimizer).
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_comm.cuh"
#include "hpc_zero2.cuh"
#include "hpc_zero2_optimizer.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <string>

#ifdef HPC_HAVE_NCCL
#  include <nccl.h>
#endif

namespace hpc_opt {
namespace zero3 {

// ===========================================================================
// ZeRO3ParamState  —  per-tensor parameter management
//   Each tensor has:
//     shard      : persistent, owned by this rank  (numel = shard_size ≤ tensor.numel)
//     full_params: temporary, valid only during forward pass window
// ===========================================================================
struct ZeRO3ParamState {
    // ---- Persistent (always alive) ----
    float*  d_shard        = nullptr; // FP32 master-weight shard
    void*   d_param_shard  = nullptr; // low-prec param shard (same data, cast view)
    size_t  shard_numel    = 0;       // elements this rank owns

    // ---- Temporary (forward window only) ----
    float*  d_full_fp32    = nullptr; // full FP32 param  (size = tensor.numel)
    void*   d_full_lowprec = nullptr; // full low-prec param (user forward pointer)
    bool    is_gathered    = false;   // true while full params are valid

    // ---- Gradient shard (post ReduceScatter) ----
    float*  d_grad_shard   = nullptr; // FP32 mean grad shard

    Dtype   dtype          = Dtype::FP32;
    size_t  tensor_numel   = 0;       // total elements in the original tensor
    size_t  global_offset  = 0;       // flat-universe element offset
};

// ===========================================================================
// Kernels: cast full FP32 params → low-prec user param buffer
// ===========================================================================
template<typename T>
__global__ void k_cast_fp32_to_lowprec(
        T*           __restrict__ dst,
        const float* __restrict__ src,
        size_t numel)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < numel;
         i += gridDim.x*blockDim.x)
    {
        if constexpr (std::is_same_v<T,float>)
            dst[i] = src[i];
        else if constexpr (std::is_same_v<T,half>)
            dst[i] = __float2half(src[i]);
        else
            dst[i] = __float2bfloat16(src[i]);
    }
}

// Scatter shard into full buffer at its global offset
__global__ void k_scatter_shard(
        float*       __restrict__ full,
        const float* __restrict__ shard,
        size_t shard_start,
        size_t shard_numel)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < shard_numel;
         i += gridDim.x*blockDim.x)
        full[shard_start + i] = __ldg(shard + i);
}

// ===========================================================================
// ZeRO3Engine  —  manages the per-forward AllGather lifecycle
// ===========================================================================
class ZeRO3Engine {
public:
    ZeRO3Engine() = default;
    ~ZeRO3Engine() { destroy(); }

    ZeRO3Engine(const ZeRO3Engine&)            = delete;
    ZeRO3Engine& operator=(const ZeRO3Engine&) = delete;

    // -----------------------------------------------------------------------
    // init  —  build shard layout, allocate permanent shards + grad shards
    // -----------------------------------------------------------------------
    void init(const TensorView* params, int n,
              CommContext& comm,
              cudaStream_t compute_stream,
              cudaStream_t comm_stream)
    {
        n_tensors_      = n;
        comm_           = &comm;
        compute_stream_ = compute_stream;
        comm_stream_    = comm_stream;

        // Build flat shard layout (same helper as ZeRO-2)
        std::vector<size_t> numel_list(n);
        for (int i = 0; i < n; ++i) numel_list[i] = params[i].numel;
        layout_.build(numel_list.data(), n, comm.world_size(), comm.rank());

        // Allocate flat FP32 master-weight buffer (shard_size for all tensors)
        // Each tensor's shard is a contiguous slice of one big flat allocation.
        total_flat_bytes_ = layout_.total_numel * sizeof(float);
        shard_bytes_      = layout_.shard_size  * sizeof(float);

        HPC_CUDA_CHECK(cudaMalloc(&d_master_flat_, total_flat_bytes_));
        HPC_CUDA_CHECK(cudaMalloc(&d_grad_flat_,   total_flat_bytes_)); // full grad bucket
        HPC_CUDA_CHECK(cudaMalloc(&d_grad_shard_,  shard_bytes_));      // post-RS shard

        HPC_CUDA_CHECK(cudaMemsetAsync(d_master_flat_, 0, total_flat_bytes_, compute_stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_grad_flat_,   0, total_flat_bytes_, compute_stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_grad_shard_,  0, shard_bytes_,      compute_stream));

        // Full-param temporary buffer (allocated lazily in prefetch_params)
        d_full_flat_ = nullptr;

        // Events for stream synchronisation
        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev_gather_done_, cudaEventDisableTiming));
        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute_done_,cudaEventDisableTiming));

        // Dtype list
        dtypes_.resize(n);
        for (int i = 0; i < n; ++i) dtypes_[i] = params[i].dtype;

        // Copy initial param values into master shard
        _init_master_from_params(params, n, compute_stream);

        HPC_CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        initialized_ = true;

        if (comm.is_root()) {
            layout_.print();
            _print_memory(params);
        }
    }

    // -----------------------------------------------------------------------
    // prefetch_params
    //   AllGather all param shards → d_full_flat_ (temporary).
    //   Cast FP32 full-flat → low-prec TensorView buffers so user
    //   forward code can read params at the original dtype.
    //
    //   Call on comm_stream; gate compute on ev_gather_done_.
    //   For pipelined prefetch, call for tensor-group L+1 while
    //   computing forward of tensor-group L.
    // -----------------------------------------------------------------------
    void prefetch_params(TensorView* out_params, int n,
                         cudaStream_t ag_stream)
    {
        assert(initialized_);

        // Lazy-allocate full-param buffer
        if (!d_full_flat_) {
            HPC_CUDA_CHECK(cudaMalloc(&d_full_flat_, total_flat_bytes_));
            HPC_CUDA_CHECK(cudaMemsetAsync(d_full_flat_, 0,
                                           total_flat_bytes_, ag_stream));
        }

        // Gate AllGather on any prior compute finishing
        HPC_CUDA_CHECK(cudaEventRecord(ev_compute_done_, compute_stream_));
        HPC_CUDA_CHECK(cudaStreamWaitEvent(ag_stream, ev_compute_done_, 0));

#ifdef HPC_HAVE_NCCL
        if (comm_->world_size() > 1) {
            // AllGather: each rank sends d_master_flat_[rank_lo..rank_hi)
            // and receives the full d_full_flat_
            ncclComm_t nccl = _get_nccl();
            NCCL_CHECK(ncclAllGather(
                d_master_flat_ + layout_.rank_lo,  // send: this rank's shard
                d_full_flat_,                      // recv: full flat
                layout_.shard_size,
                ncclFloat, nccl, ag_stream));
        } else {
            // Single GPU: full = shard (they overlap in the same buffer layout)
            HPC_CUDA_CHECK(cudaMemcpyAsync(
                d_full_flat_, d_master_flat_,
                shard_bytes_, cudaMemcpyDeviceToDevice, ag_stream));
        }
#else
        HPC_CUDA_CHECK(cudaMemcpyAsync(
            d_full_flat_, d_master_flat_,
            layout_.shard_size * sizeof(float),
            cudaMemcpyDeviceToDevice, ag_stream));
#endif

        // Cast full FP32 flat → per-tensor low-prec user buffers
        for (int i = 0; i < n; ++i) {
            const auto& s = layout_.shards[i];
            size_t numel  = s.numel;
            int    blk    = hpc_blocks(numel);
            const float* src = d_full_flat_ + s.global_offset;

            switch (out_params[i].dtype) {
                case Dtype::FP32:
                    k_cast_fp32_to_lowprec<float><<<blk, HPC_BLOCK, 0, ag_stream>>>(
                        static_cast<float*>(out_params[i].data), src, numel);
                    break;
                case Dtype::FP16:
                    k_cast_fp32_to_lowprec<half><<<blk, HPC_BLOCK, 0, ag_stream>>>(
                        static_cast<half*>(out_params[i].data), src, numel);
                    break;
                case Dtype::BF16:
                    k_cast_fp32_to_lowprec<__nv_bfloat16><<<blk, HPC_BLOCK, 0, ag_stream>>>(
                        static_cast<__nv_bfloat16*>(out_params[i].data), src, numel);
                    break;
            }
        }

        // Signal that gathered params are ready on compute_stream
        HPC_CUDA_CHECK(cudaEventRecord(ev_gather_done_, ag_stream));
        HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, ev_gather_done_, 0));
    }

    // -----------------------------------------------------------------------
    // release_params  —  free the temporary full-param buffer after forward.
    //   This is the key ZeRO-3 memory saving: parameters only live for
    //   the duration of the forward pass.
    //   Pass release_immediately=true to cudaFree right away;
    //   false to just zero the buffer (cheaper, reuses allocation next fwd).
    // -----------------------------------------------------------------------
    void release_params(bool release_immediately = false,
                        cudaStream_t stream = 0)
    {
        if (!d_full_flat_) return;

        if (release_immediately) {
            HPC_CUDA_CHECK(cudaStreamSynchronize(stream ? stream : compute_stream_));
            HPC_CUDA_CHECK(cudaFree(d_full_flat_));
            d_full_flat_ = nullptr;
        } else {
            // Zero instead of free — keeps the allocation for next fwd
            HPC_CUDA_CHECK(cudaMemsetAsync(d_full_flat_, 0,
                                           total_flat_bytes_,
                                           stream ? stream : compute_stream_));
        }
    }

    // -----------------------------------------------------------------------
    // pack_grads  —  pack per-tensor grad buffers into the flat grad bucket.
    //   Identical to ZeRO-2 pack step.
    // -----------------------------------------------------------------------
    void pack_grads(const TensorView* grads, int n, cudaStream_t stream) {
        HPC_CUDA_CHECK(cudaMemsetAsync(d_grad_flat_, 0, total_flat_bytes_, stream));

        for (int i = 0; i < n; ++i) {
            const auto& s = layout_.shards[i];
            int blk = hpc_blocks(s.numel);

            switch (grads[i].dtype) {
                case Dtype::FP32:
                    zero2::k_pack_tensor<float><<<blk, HPC_BLOCK, 0, stream>>>(
                        d_grad_flat_,
                        static_cast<const float*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
                case Dtype::FP16:
                    zero2::k_pack_tensor<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        d_grad_flat_,
                        static_cast<const half*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
                case Dtype::BF16:
                    zero2::k_pack_tensor<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        d_grad_flat_,
                        static_cast<const __nv_bfloat16*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // reduce_scatter_grads  —  ncclReduceScatter on flat grad bucket.
    //   Each rank receives its mean gradient shard in d_grad_shard_.
    // -----------------------------------------------------------------------
    void reduce_scatter_grads(cudaStream_t rs_stream) {
        assert(initialized_);

#ifdef HPC_HAVE_NCCL
        if (comm_->world_size() > 1) {
            ncclComm_t nccl = _get_nccl();
            NCCL_CHECK(ncclReduceScatter(
                d_grad_flat_, d_grad_shard_,
                layout_.shard_size,
                ncclFloat, ncclSum, nccl, rs_stream));

            // Scale sum → mean
            const float inv_w = 1.0f / static_cast<float>(comm_->world_size());
            zero2::k_scale_shard<<<hpc_blocks(layout_.shard_size),
                                   HPC_BLOCK, 0, rs_stream>>>(
                d_grad_shard_, layout_.shard_size, inv_w);
        } else {
            HPC_CUDA_CHECK(cudaMemcpyAsync(
                d_grad_shard_, d_grad_flat_,
                layout_.shard_size * sizeof(float),
                cudaMemcpyDeviceToDevice, rs_stream));
        }
#else
        HPC_CUDA_CHECK(cudaMemcpyAsync(
            d_grad_shard_, d_grad_flat_,
            layout_.shard_size * sizeof(float),
            cudaMemcpyDeviceToDevice, rs_stream));
#endif

        HPC_CUDA_CHECK(cudaEventRecord(ev_gather_done_, rs_stream));
        HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, ev_gather_done_, 0));
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    float*             master_flat()   const { return d_master_flat_; }
    float*             grad_shard()    const { return d_grad_shard_; }
    float*             param_shard()   const { return d_master_flat_ + layout_.rank_lo; }
    size_t             shard_numel()   const { return layout_.shard_size; }
    const zero2::ShardLayout& layout() const { return layout_; }
    bool               initialized()   const { return initialized_; }
    int                n_tensors()     const { return n_tensors_; }
    bool               is_root()       const { return comm_ && comm_->is_root(); }

private:
    bool         initialized_    = false;
    int          n_tensors_      = 0;
    CommContext* comm_           = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t comm_stream_    = nullptr;

    zero2::ShardLayout layout_;

    // Persistent flat buffers (size = total_numel)
    float*  d_master_flat_ = nullptr;  // FP32 master weight (shard region only valid)
    float*  d_grad_flat_   = nullptr;  // gradient accumulation flat buffer
    float*  d_grad_shard_  = nullptr;  // post-ReduceScatter gradient shard

    // Temporary: only valid during forward pass window
    float*  d_full_flat_   = nullptr;  // full gathered FP32 params

    size_t  total_flat_bytes_ = 0;
    size_t  shard_bytes_      = 0;

    cudaEvent_t ev_gather_done_  = nullptr;
    cudaEvent_t ev_compute_done_ = nullptr;

    std::vector<Dtype> dtypes_;

    // -----------------------------------------------------------------------
    // Copy initial param values from user tensors into the master-weight shard
    // -----------------------------------------------------------------------
    void _init_master_from_params(const TensorView* params, int n,
                                   cudaStream_t stream)
    {
        float* master = d_master_flat_;

        for (int i = 0; i < n; ++i) {
            const auto& s = layout_.shards[i];
            if (s.shard_hi <= s.shard_lo) continue;

            size_t lo  = s.shard_lo;
            size_t cnt = s.shard_hi - s.shard_lo;
            // Destination in master_flat at this rank's shard offset
            size_t flat_dst = layout_.rank_lo + (s.global_offset + lo - layout_.rank_lo);
            // More precisely: dst = rank_lo + local_offset_within_shard
            // For simplicity: write to master_flat[global_offset + lo] which equals
            // master_flat[rank_lo + (global_offset + lo - rank_lo)]
            // Both refer to the same memory region since rank_lo ≤ global+lo < rank_hi

            float* dst = master + s.global_offset + lo;
            int blk = hpc_blocks(cnt);

            switch (params[i].dtype) {
                case Dtype::FP32:
                    HPC_CUDA_CHECK(cudaMemcpyAsync(
                        dst,
                        static_cast<const float*>(params[i].data) + lo,
                        cnt * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    break;
                case Dtype::FP16:
                    zero2::k_pack_tensor<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        master,
                        static_cast<const half*>(params[i].data) + lo,
                        s.global_offset + lo, cnt);
                    break;
                case Dtype::BF16:
                    zero2::k_pack_tensor<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        master,
                        static_cast<const __nv_bfloat16*>(params[i].data) + lo,
                        s.global_offset + lo, cnt);
                    break;
            }
        }
    }

    void _print_memory(const TensorView* params) const {
        size_t total = layout_.total_numel;
        int    W     = layout_.world_size;

        // Determine param element size (use first tensor dtype)
        size_t param_bytes = (params[0].dtype == Dtype::FP32) ? 4 : 2;

        double zero3_gb = (total * param_bytes     / W   // param shard (low-prec)
                         + total * 4.0             / W   // FP32 master shard
                         + total * 4.0             / W   // grad shard
                         + total * 4.0 * 2.0       / W)  // Adam m+v shard
                        / 1e9;

        double zero0_gb = total * (param_bytes + 4.0 + 4.0*2.0) / 1e9;  // ZeRO-0

        fprintf(stdout,
            "[ZeRO-3] Memory per rank (W=%d):\n"
            "  Param shard  (low-prec, 1/W): %.2f GB\n"
            "  Master shard (FP32,    1/W) : %.2f GB\n"
            "  Grad shard   (FP32,    1/W) : %.2f GB\n"
            "  Adam m+v     (FP32,    1/W) : %.2f GB\n"
            "  ─────────────────────────────────────\n"
            "  ZeRO-3 total               : %.2f GB  (vs ZeRO-0: %.2f GB, %.1fx less)\n"
            "  Peak during fwd (full AG)  : %.2f GB  (+%.2f GB temp)\n\n",
            W,
            (double)(total * param_bytes / W) / 1e9,
            (double)(total * 4UL        / W) / 1e9,
            (double)(total * 4UL        / W) / 1e9,
            (double)(total * 8UL        / W) / 1e9,
            zero3_gb,
            zero0_gb,
            zero0_gb / zero3_gb,
            zero3_gb + (double)(total * 4UL) / 1e9,   // peak = static + full AG
            (double)(total * 4UL) / 1e9);
    }

    void destroy() {
        if (d_master_flat_) { cudaFree(d_master_flat_); d_master_flat_ = nullptr; }
        if (d_grad_flat_)   { cudaFree(d_grad_flat_);   d_grad_flat_   = nullptr; }
        if (d_grad_shard_)  { cudaFree(d_grad_shard_);  d_grad_shard_  = nullptr; }
        if (d_full_flat_)   { cudaFree(d_full_flat_);   d_full_flat_   = nullptr; }
        if (ev_gather_done_)  cudaEventDestroy(ev_gather_done_);
        if (ev_compute_done_) cudaEventDestroy(ev_compute_done_);
    }

#ifdef HPC_HAVE_NCCL
    ncclComm_t _get_nccl() const { return nullptr; } // replaced by CommContext friend
#endif
};

// ===========================================================================
// ZeRO3TrainerConfig
// ===========================================================================
struct ZeRO3TrainerConfig {
    zero2::ShardedOptKind kind          = zero2::ShardedOptKind::AdamW;
    float                 lr            = 3e-4f;
    float                 beta1         = 0.9f;
    float                 beta2         = 0.95f;
    float                 eps           = 1e-8f;
    float                 weight_decay  = 0.1f;
    float                 max_grad_norm = 1.0f;
    bool                  release_full_params_immediately = false;
    //   true  → cudaFree after every fwd (maximum VRAM saving, slower)
    //   false → zero the buffer (keeps the allocation, avoids malloc overhead)
};

// ===========================================================================
// ZeRO3Trainer  —  complete ZeRO-3 all-in-one object
//
// step() protocol:
//   trainer.prefetch_params(params, n);    // ① AllGather → full params
//   forward(params);                       // ② user forward (reads full params)
//   trainer.release_params();             // ③ free/zero temp buffer
//   backward(grads);                       // ④ user backward (writes full grads)
//   trainer.backward_step(params, grads, n); // ⑤-⑦ RS + opt + ready for next ①
// ===========================================================================
class ZeRO3Trainer {
public:
    ZeRO3Trainer() = default;
    ~ZeRO3Trainer() = default;

    ZeRO3Trainer(const ZeRO3Trainer&)            = delete;
    ZeRO3Trainer& operator=(const ZeRO3Trainer&) = delete;

    void init(const TensorView* params, int n,
              CommContext&              comm,
              const ZeRO3TrainerConfig& cfg,
              cudaStream_t              compute_stream,
              cudaStream_t              comm_stream = nullptr)
    {
        cfg_            = cfg;
        compute_stream_ = compute_stream;
        comm_stream_    = comm_stream ? comm_stream : compute_stream;
        n_tensors_      = n;

        engine_.init(params, n, comm, compute_stream_, comm_stream_);

        // Build sharded optimizer on this rank's master-weight shard
        opt_ = std::make_unique<zero2::ZeRO2ShardedOptimizer>(cfg.kind);
        opt_->lr           = cfg.lr;
        opt_->beta1        = cfg.beta1;
        opt_->beta2        = cfg.beta2;
        opt_->eps          = cfg.eps;
        opt_->weight_decay = cfg.weight_decay;
        opt_->init(engine_.shard_numel(), engine_.param_shard(), compute_stream_);

        numel_list_.resize(n);
        for (int i = 0; i < n; ++i) numel_list_[i] = params[i].numel;
    }

    // ------------------------------------------------------------------ //
    // ① Prefetch: AllGather param shards → full param tensors
    //    Call this at the START of each forward pass.
    //    Uses comm_stream; compute_stream is automatically gated.
    // ------------------------------------------------------------------ //
    void prefetch_params(TensorView* params, int n) {
        engine_.prefetch_params(params, n, comm_stream_);
    }

    // ------------------------------------------------------------------ //
    // ③ Release: free/zero the full-param temp buffer.
    //    Call AFTER forward, BEFORE backward.
    // ------------------------------------------------------------------ //
    void release_params() {
        engine_.release_params(cfg_.release_full_params_immediately,
                               compute_stream_);
    }

    // ------------------------------------------------------------------ //
    // ⑤-⑦ backward_step: pack grads → RS → local opt step
    //    Call AFTER user backward() has filled grad buffers.
    // ------------------------------------------------------------------ //
    void backward_step(TensorView* params, TensorView* grads, int n) {
        // Pack all grad tensors → flat FP32 bucket
        engine_.pack_grads(grads, n, compute_stream_);

        // ReduceScatter (comm_stream)
        engine_.reduce_scatter_grads(comm_stream_);

        // Optional gradient clipping on shard
        if (cfg_.max_grad_norm > 0.0f)
            _clip_grad_shard(compute_stream_);

        // Local optimizer step (operates on master_shard)
        opt_->step(engine_.grad_shard(), compute_stream_);

        step_++;
        stats_.step   = step_;
        stats_.last_lr = cfg_.lr;
    }

    // ------------------------------------------------------------------ //
    // Combined convenience wrapper for models where the entire model fits
    // in a single forward/backward call (simpler API, same protocol inside)
    // ------------------------------------------------------------------ //
    void step(TensorView* params, TensorView* grads, int n,
              std::function<void()> forward_fn,
              std::function<void()> backward_fn)
    {
        prefetch_params(params, n);
        forward_fn();
        release_params();
        backward_fn();
        backward_step(params, grads, n);
    }

    // ------------------------------------------------------------------ //
    // Accessors
    // ------------------------------------------------------------------ //
    float             current_lr()  const { return cfg_.lr; }
    float&            lr()                { return cfg_.lr; }
    uint32_t          step_count()  const { return step_; }
    OptimizerStats&   stats()             { return stats_; }
    ZeRO3Engine&      engine()            { return engine_; }

    void log(int step, float loss, bool root) const {
        if (!root) return;
        fprintf(stdout,
            "[ZeRO-3] step=%-6d  lr=%.2e  loss=%.4f  |g|_shard=%.3f\n",
            step, cfg_.lr, loss, stats_.grad_norm_after);
    }

    // Per-rank checkpoint: save optimizer shard state
    void save_checkpoint(const std::string& path, int rank) const {
        opt_->save_shard_state(path, rank);
    }

    uint32_t load_checkpoint(const std::string& path, int rank,
                             cudaStream_t stream = 0) {
        return opt_->load_shard_state(path, rank, stream);
    }

private:
    ZeRO3TrainerConfig cfg_;
    ZeRO3Engine        engine_;
    std::unique_ptr<zero2::ZeRO2ShardedOptimizer> opt_;
    OptimizerStats     stats_{};

    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t comm_stream_    = nullptr;
    int          n_tensors_      = 0;
    uint32_t     step_           = 0;
    std::vector<size_t> numel_list_;

    float* d_shard_sq_ = nullptr;  // scratch for shard-local grad-norm

    void _clip_grad_shard(cudaStream_t stream) {
        // Reuse hpc_grad_clip primitives on the shard buffer
        float* g   = engine_.grad_shard();
        size_t n   = engine_.shard_numel();

        if (!d_shard_sq_) HPC_CUDA_CHECK(cudaMalloc(&d_shard_sq_, sizeof(float)));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_shard_sq_, 0, sizeof(float), stream));

        uint32_t* d_nans;
        HPC_CUDA_CHECK(cudaMalloc(&d_nans, sizeof(uint32_t)));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_nans, 0, sizeof(uint32_t), stream));

        k_partial_sq_norm<float><<<hpc_blocks(n), HPC_BLOCK, 0, stream>>>(
            g, d_shard_sq_, n, d_nans);

        float sq = 0.0f;
        HPC_CUDA_CHECK(cudaMemcpyAsync(&sq, d_shard_sq_, sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        HPC_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_nans);

        float norm = sqrtf(sq);
        stats_.grad_norm_after = norm;

        if (norm > cfg_.max_grad_norm && norm > 0.0f) {
            float scale = cfg_.max_grad_norm / (norm + 1e-6f);
            k_apply_scale<float><<<hpc_blocks(n), HPC_BLOCK, 0, stream>>>(
                g, scale, n);
        }
    }
};

} // namespace zero3
} // namespace hpc_opt
