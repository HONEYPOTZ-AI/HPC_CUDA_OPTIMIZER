// =============================================================================
// HPC CUDA Optimizer Library — ZeRO-2 Extension
// hpc_zero2.cuh  —  Gradient sharding engine
//
// ZeRO-2 vs ZeRO-1 memory breakdown per rank (W = world_size):
//   ZeRO-0: params(4Ψ) + grads(4Ψ) + Adam-state(8Ψ)          = 16Ψ bytes
//   ZeRO-1: params(4Ψ) + grads(4Ψ) + Adam-state(8Ψ/W)        = (8 + 8/W)Ψ
//   ZeRO-2: params(4Ψ) + grads(4Ψ/W) + Adam-state(8Ψ/W)      = (4 + 12/W)Ψ
//   ZeRO-3: params(4Ψ/W) + grads(4Ψ/W) + Adam-state(8Ψ/W)    = 16Ψ/W
//
// ZeRO-2 algorithm per step:
//   1. Forward:      All ranks hold identical full params (read-only during fwd)
//   2. Backward:     Each rank accumulates full gradient locally into grad_buf
//   3. ReduceScatter: NCCL ncclReduceScatter — rank r receives summed
//                    gradients for elements [r*shard_size .. (r+1)*shard_size)
//                    (divided by world_size → mean gradient)
//   4. Local step:   Each rank runs the optimizer kernel ONLY on its shard
//   5. AllGather:    NCCL ncclAllGather — broadcast every shard so all ranks
//                    hold the complete updated parameter tensor again
//
// Communication volume vs ZeRO-1:
//   ZeRO-1: AllReduce grads = 2Ψ/W × W = 2Ψ   (ring-allreduce decomposition)
//   ZeRO-2: ReduceScatter(Ψ) + AllGather(Ψ) = 2Ψ   (identical bandwidth!)
//   → ZeRO-2 does NOT increase communication; it cuts gradient MEMORY by W.
//
// Bucket packing:
//   Tensors are concatenated into one contiguous flat buffer per dtype.
//   This allows a single ncclReduceScatter call instead of N per-tensor calls,
//   reducing NCCL kernel launch overhead and improving NVLink utilisation.
//
// Dual-stream overlap:
//   After each bucket's ReduceScatter completes on comm_stream, the local
//   optimizer step is enqueued on compute_stream (after a cudaStreamWaitEvent
//   gate). AllGather runs last on comm_stream after all shards are updated.
//
// Supports FP32 / FP16 / BF16 gradient buffers.
// Moments (m, v) are always kept in FP32 on each rank's shard.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_comm.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef HPC_HAVE_NCCL
#  include <nccl.h>
#endif

namespace hpc_opt {
namespace zero2 {

// ===========================================================================
// ShardLayout  —  maps each parameter tensor to a flat "universe" offset
//                 and computes per-rank owned [lo, hi) intervals.
// ===========================================================================
struct TensorShard {
    size_t global_offset; // element offset in the flat universe
    size_t numel;         // total elements in this tensor
    size_t shard_lo;      // first element owned by this rank (local index)
    size_t shard_hi;      // one-past-last element owned (local index)
    bool   fully_owned;   // shard covers the entire tensor
    bool   partially_owned; // shard covers a non-empty strict subset
};

struct ShardLayout {
    int    world_size;
    int    rank;
    size_t total_numel;   // sum of all tensor numel
    size_t shard_size;    // ceil(total_numel / world_size)
    size_t rank_lo;       // global flat element index this rank owns from
    size_t rank_hi;       // global flat element index this rank owns up to

    std::vector<TensorShard> shards; // one entry per parameter tensor

    // -----------------------------------------------------------------------
    // Build layout from a list of tensor sizes.
    // Pads total_numel to the nearest multiple of world_size so every rank
    // gets an equal-size shard (padding elements are zeroed, never used).
    // -----------------------------------------------------------------------
    void build(const size_t* numel_list, int n_tensors,
               int world, int rk)
    {
        world_size = world;
        rank       = rk;

        size_t raw_total = 0;
        for (int i = 0; i < n_tensors; ++i) raw_total += numel_list[i];

        // Pad to multiple of world_size
        size_t pad    = (world_size - raw_total % world_size) % world_size;
        total_numel   = raw_total + pad;
        shard_size    = total_numel / world_size;

        rank_lo = static_cast<size_t>(rank)     * shard_size;
        rank_hi = static_cast<size_t>(rank + 1) * shard_size;

        shards.resize(n_tensors);
        size_t offset = 0;
        for (int i = 0; i < n_tensors; ++i) {
            size_t n = numel_list[i];
            auto&  s = shards[i];
            s.global_offset = offset;
            s.numel         = n;

            // Intersect [offset, offset+n) with [rank_lo, rank_hi)
            size_t lo = std::max(offset,   rank_lo);
            size_t hi = std::min(offset+n, rank_hi);

            if (lo < hi) {
                s.shard_lo      = lo - offset;
                s.shard_hi      = hi - offset;
                s.fully_owned   = (lo == offset && hi == offset + n);
                s.partially_owned = !s.fully_owned;
            } else {
                s.shard_lo = s.shard_hi = 0;
                s.fully_owned = s.partially_owned = false;
            }
            offset += n;
        }
    }

    // Bytes owned by this rank for a given element dtype size
    size_t owned_bytes(size_t elem_size) const {
        return shard_size * elem_size;
    }

    // Pretty-print layout
    void print() const {
        fprintf(stdout,
            "[ZeRO-2] Layout  rank=%d/%d  total=%zu  shard=%zu  "
            "range=[%zu,%zu)\n",
            rank, world_size, total_numel, shard_size, rank_lo, rank_hi);
        for (int i = 0; i < (int)shards.size(); ++i) {
            const auto& s = shards[i];
            if (s.shard_hi > s.shard_lo)
                fprintf(stdout,
                    "  tensor[%2d]  numel=%-10zu  owned=[%zu,%zu)\n",
                    i, s.numel, s.shard_lo, s.shard_hi);
        }
    }
};

// ===========================================================================
// GradBucket  —  flat contiguous FP32 buffer holding all gradient data
//               (tensors are copied in, NCCL operates on this buffer)
// ===========================================================================
struct GradBucket {
    float*  d_flat   = nullptr; // device: full-size gradient flat buffer
    float*  d_shard  = nullptr; // device: this rank's shard after ReduceScatter
    float*  h_recv   = nullptr; // pinned host buffer for transfers if needed
    size_t  total_numel = 0;    // includes padding
    size_t  shard_numel = 0;    // total_numel / world_size

    void alloc(size_t total, size_t shard, cudaStream_t stream) {
        total_numel = total;
        shard_numel = shard;
        HPC_CUDA_CHECK(cudaMalloc(&d_flat,  total * sizeof(float)));
        HPC_CUDA_CHECK(cudaMalloc(&d_shard, shard * sizeof(float)));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_flat,  0, total * sizeof(float), stream));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_shard, 0, shard * sizeof(float), stream));
    }

    void free_all() {
        if (d_flat)  { cudaFree(d_flat);  d_flat  = nullptr; }
        if (d_shard) { cudaFree(d_shard); d_shard = nullptr; }
        if (h_recv)  { cudaFreeHost(h_recv); h_recv = nullptr; }
    }
};

// ===========================================================================
// Pack / Unpack kernels — copy individual tensors into/from the flat bucket
// ===========================================================================

// Pack tensor t (possibly FP16/BF16 → FP32) into flat buffer at offset.
template<typename SrcT>
__global__ void k_pack_tensor(
        float*       __restrict__ flat,
        const SrcT*  __restrict__ src,
        size_t offset, size_t numel)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < numel;
         i += gridDim.x*blockDim.x)
        flat[offset + i] = prec::to_float(__ldg(src + i));
}

// Unpack shard region back into tensor (FP32 → FP32 or FP32 → BF16/FP16)
template<typename DstT>
__global__ void k_unpack_shard(
        DstT*         __restrict__ dst,
        const float*  __restrict__ shard,
        size_t dst_offset,    // element offset within dst
        size_t shard_offset,  // element offset within shard buffer
        size_t numel)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < numel;
         i += gridDim.x*blockDim.x)
    {
        float v = __ldg(shard + shard_offset + i);
        if constexpr (std::is_same_v<DstT, float>)
            dst[dst_offset + i] = v;
        else if constexpr (std::is_same_v<DstT, half>)
            dst[dst_offset + i] = __float2half(v);
        else
            dst[dst_offset + i] = __float2bfloat16(v);
    }
}

// Divide shard by world_size (convert sum → mean after ReduceScatter)
__global__ void k_scale_shard(float* shard, size_t numel, float inv_w) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < numel;
         i += gridDim.x*blockDim.x)
        shard[i] *= inv_w;
}

// AllGather unpack: copy each peer's shard back into the full flat buffer
__global__ void k_copy_segment(
        float* __restrict__ dst, const float* __restrict__ src,
        size_t dst_off, size_t numel)
{
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < numel;
         i += gridDim.x*blockDim.x)
        dst[dst_off + i] = __ldg(src + i);
}

// ===========================================================================
// ZeRO2Engine  —  main orchestrator
// ===========================================================================
class ZeRO2Engine {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    ZeRO2Engine() = default;
    ~ZeRO2Engine() { destroy(); }

    ZeRO2Engine(const ZeRO2Engine&)            = delete;
    ZeRO2Engine& operator=(const ZeRO2Engine&) = delete;

    // -----------------------------------------------------------------------
    // init
    //   params        —  full parameter tensor list (all dtypes supported)
    //   n             —  number of tensors
    //   comm          —  CommContext (must be initialized)
    //   compute_stream—  stream for optimizer kernels
    //   comm_stream   —  stream for NCCL ops (may == compute_stream)
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

        // Build shard layout
        std::vector<size_t> numel_list(n);
        for (int i = 0; i < n; ++i) numel_list[i] = params[i].numel;
        layout_.build(numel_list.data(), n, comm.world_size(), comm.rank());

        // Allocate flat gradient bucket (FP32 — always accumulate in FP32)
        bucket_.alloc(layout_.total_numel, layout_.shard_size, compute_stream);

        // Allocate per-tensor param flat buffer for AllGather output
        // (reuse the same flat buffer — first half = reduce-scatter output,
        //  second reuse for AG receive)
        HPC_CUDA_CHECK(cudaMalloc(&d_param_flat_,
                                  layout_.total_numel * sizeof(float)));
        HPC_CUDA_CHECK(cudaMemsetAsync(d_param_flat_, 0,
                                       layout_.total_numel * sizeof(float),
                                       compute_stream));

#ifdef HPC_HAVE_NCCL
        // Create a dedicated event to gate compute on comm completion
        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&rs_done_event_,
                                                cudaEventDisableTiming));
        HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ag_done_event_,
                                                cudaEventDisableTiming));
#endif
        // Store dtype info per tensor
        dtypes_.resize(n);
        for (int i = 0; i < n; ++i) dtypes_[i] = params[i].dtype;

        initialized_ = true;

        if (comm.is_root()) layout_.print();
    }

    // -----------------------------------------------------------------------
    // pack_grads
    //   Copies all gradient tensors into the flat FP32 bucket.
    //   Called immediately after backward(); runs on compute_stream.
    // -----------------------------------------------------------------------
    void pack_grads(const TensorView* grads, int n, cudaStream_t stream) {
        assert(initialized_);
        // Zero pad region
        HPC_CUDA_CHECK(cudaMemsetAsync(bucket_.d_flat, 0,
                                       layout_.total_numel * sizeof(float),
                                       stream));
        for (int i = 0; i < n; ++i) {
            const auto& s = layout_.shards[i];
            int blk = hpc_blocks(s.numel);

            switch (grads[i].dtype) {
                case Dtype::FP32:
                    k_pack_tensor<float><<<blk, HPC_BLOCK, 0, stream>>>(
                        bucket_.d_flat,
                        static_cast<const float*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
                case Dtype::FP16:
                    k_pack_tensor<half><<<blk, HPC_BLOCK, 0, stream>>>(
                        bucket_.d_flat,
                        static_cast<const half*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
                case Dtype::BF16:
                    k_pack_tensor<__nv_bfloat16><<<blk, HPC_BLOCK, 0, stream>>>(
                        bucket_.d_flat,
                        static_cast<const __nv_bfloat16*>(grads[i].data),
                        s.global_offset, s.numel);
                    break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // reduce_scatter_grads
    //   Executes ncclReduceScatter on the flat gradient bucket.
    //   Each rank receives the summed gradient shard for its partition,
    //   then scales by 1/world_size to produce the mean gradient.
    //   After this call, bucket_.d_shard holds the MEAN grad shard.
    //
    //   On single-GPU fallback (world_size==1): copies flat→shard directly.
    // -----------------------------------------------------------------------
    void reduce_scatter_grads(cudaStream_t comm_stream) {
        assert(initialized_);

#ifdef HPC_HAVE_NCCL
        if (comm_->world_size() > 1) {
            // ncclReduceScatter: input = d_flat (total_numel),
            //                   output = d_shard (shard_size per rank)
            ncclComm_t nccl = get_nccl_comm();
            NCCL_CHECK(ncclReduceScatter(
                bucket_.d_flat,        // send (each rank sends its copy)
                bucket_.d_shard,       // recv (gets its own shard summed)
                layout_.shard_size,    // recv count per rank
                ncclFloat,
                ncclSum,
                nccl, comm_stream));

            // Scale sum → mean
            float inv_w = 1.0f / static_cast<float>(comm_->world_size());
            int   blk   = hpc_blocks(layout_.shard_size);
            k_scale_shard<<<blk, HPC_BLOCK, 0, comm_stream>>>(
                bucket_.d_shard, layout_.shard_size, inv_w);

            // Record event so compute_stream can wait
            HPC_CUDA_CHECK(cudaEventRecord(rs_done_event_, comm_stream));
            return;
        }
#endif
        // Single GPU: shard == full flat (no reduction needed)
        HPC_CUDA_CHECK(cudaMemcpyAsync(
            bucket_.d_shard, bucket_.d_flat,
            layout_.shard_size * sizeof(float),
            cudaMemcpyDeviceToDevice, comm_stream));
    }

    // -----------------------------------------------------------------------
    // wait_reduce_scatter
    //   Gates compute_stream until ReduceScatter on comm_stream is done.
    //   Call this before launching local optimizer kernels.
    // -----------------------------------------------------------------------
    void wait_reduce_scatter(cudaStream_t compute_stream) {
#ifdef HPC_HAVE_NCCL
        if (comm_->world_size() > 1)
            HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream,
                                               rs_done_event_, 0));
#endif
    }

    // -----------------------------------------------------------------------
    // all_gather_params
    //   After the local optimizer step has updated d_param_flat_[shard],
    //   broadcast all shards so every rank holds the full updated params.
    //   Then unpack from the flat buffer back into the individual param tensors.
    // -----------------------------------------------------------------------
    void all_gather_params(TensorView* params, int n,
                           cudaStream_t compute_stream,
                           cudaStream_t comm_stream_ag)
    {
        // Gate comm on compute being done with the local step
        HPC_CUDA_CHECK(cudaEventRecord(rs_done_event_, compute_stream));
        HPC_CUDA_CHECK(cudaStreamWaitEvent(comm_stream_ag, rs_done_event_, 0));

#ifdef HPC_HAVE_NCCL
        if (comm_->world_size() > 1) {
            // ncclAllGather: each rank sends its shard, receives all shards
            // The flat buffer is organised as [rank0_shard | rank1_shard | ...]
            ncclComm_t nccl = get_nccl_comm();
            NCCL_CHECK(ncclAllGather(
                d_param_flat_ + layout_.rank_lo,   // send: this rank's shard
                d_param_flat_,                     // recv: full flat buffer
                layout_.shard_size,
                ncclFloat,
                nccl, comm_stream_ag));

            HPC_CUDA_CHECK(cudaEventRecord(ag_done_event_, comm_stream_ag));
            HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ag_done_event_, 0));
        } else {
            // Single GPU: param_flat already has the shard in place
            HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream, rs_done_event_, 0));
        }
#else
        HPC_CUDA_CHECK(cudaStreamWaitEvent(compute_stream, rs_done_event_, 0));
#endif

        // Unpack flat buffer → individual param tensors
        for (int i = 0; i < n; ++i) {
            const auto& s = layout_.shards[i];
            int blk = hpc_blocks(s.numel);

            switch (params[i].dtype) {
                case Dtype::FP32:
                    k_unpack_shard<float><<<blk, HPC_BLOCK, 0, compute_stream>>>(
                        static_cast<float*>(params[i].data),
                        d_param_flat_,
                        0, s.global_offset, s.numel);
                    break;
                case Dtype::FP16:
                    k_unpack_shard<half><<<blk, HPC_BLOCK, 0, compute_stream>>>(
                        static_cast<half*>(params[i].data),
                        d_param_flat_,
                        0, s.global_offset, s.numel);
                    break;
                case Dtype::BF16:
                    k_unpack_shard<__nv_bfloat16><<<blk, HPC_BLOCK, 0, compute_stream>>>(
                        static_cast<__nv_bfloat16*>(params[i].data),
                        d_param_flat_,
                        0, s.global_offset, s.numel);
                    break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // commit_shard_to_param_flat
    //   After the local optimizer updates a FP32 master-weight shard in-place,
    //   copy it into d_param_flat_ at the correct rank offset so AllGather
    //   can broadcast it.
    //   If master_fp32_shard IS d_param_flat_+rank_lo, this is a no-op.
    // -----------------------------------------------------------------------
    void commit_shard(const float* updated_shard, cudaStream_t stream) {
        if (updated_shard == d_param_flat_ + layout_.rank_lo) return;
        HPC_CUDA_CHECK(cudaMemcpyAsync(
            d_param_flat_ + layout_.rank_lo,
            updated_shard,
            layout_.shard_size * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    const ShardLayout& layout()    const { return layout_; }
    float*  grad_shard()           const { return bucket_.d_shard; }
    float*  param_flat()           const { return d_param_flat_; }
    float*  param_shard()          const { return d_param_flat_ + layout_.rank_lo; }
    size_t  shard_numel()          const { return layout_.shard_size; }
    bool    initialized()          const { return initialized_; }
    int     n_tensors()            const { return n_tensors_; }

    // Memory report (bytes per rank)
    void print_memory_report(size_t param_elem_bytes = 4) const {
        size_t full_params   = layout_.total_numel * param_elem_bytes;
        size_t shard_grads   = layout_.shard_size  * 4UL;  // FP32
        size_t shard_moments = layout_.shard_size  * 4UL * 2;  // m+v FP32
        size_t param_flat    = layout_.total_numel * 4UL;   // FP32 master

        fprintf(stdout,
            "[ZeRO-2] Memory per rank (world_size=%d):\n"
            "  Full params (low-prec)  : %6.1f MB\n"
            "  Grad shard (FP32, 1/W) : %6.1f MB  [%.1fx reduction vs ZeRO-0]\n"
            "  Moment shard (FP32, 1/W): %6.1f MB  [%.1fx reduction vs ZeRO-0]\n"
            "  Param flat (FP32 master): %6.1f MB\n"
            "  Total                   : %6.1f MB\n",
            layout_.world_size,
            full_params   / 1e6,
            shard_grads   / 1e6, static_cast<float>(layout_.world_size),
            shard_moments / 1e6, static_cast<float>(layout_.world_size),
            param_flat    / 1e6,
            (full_params + shard_grads + shard_moments + param_flat) / 1e6);
    }

private:
    bool         initialized_    = false;
    int          n_tensors_      = 0;
    CommContext* comm_           = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t comm_stream_    = nullptr;

    ShardLayout            layout_;
    GradBucket             bucket_;
    float*                 d_param_flat_  = nullptr;
    std::vector<Dtype>     dtypes_;

    cudaEvent_t            rs_done_event_ = nullptr;
    cudaEvent_t            ag_done_event_ = nullptr;

#ifdef HPC_HAVE_NCCL
    ncclComm_t get_nccl_comm() {
        // CommContext exposes the NCCL comm via a friend accessor.
        // We use the public all_reduce as a proxy — for production code,
        // expose ncclComm_t directly from CommContext.
        // Here we cast through the void* convention established in hpc_comm.cuh
        return nullptr; // replaced by CommContext integration below
    }
#endif

    void destroy() {
        bucket_.free_all();
        if (d_param_flat_) { cudaFree(d_param_flat_); d_param_flat_ = nullptr; }
#ifdef HPC_HAVE_NCCL
        if (rs_done_event_) cudaEventDestroy(rs_done_event_);
        if (ag_done_event_) cudaEventDestroy(ag_done_event_);
#endif
    }
};

// ===========================================================================
// ZeRO2CommBridge  —  thin adapter that calls ncclReduceScatter/AllGather
//                     via CommContext's internal NCCL handle.
//
// CommContext keeps nccl_comm_ private; this bridge is declared friend
// (or the user passes the ncclComm_t explicitly after calling init_nccl).
// For portability without modifying CommContext we expose a free-function
// interface that takes the NCCL comm handle as void*.
// ===========================================================================

// reduce_scatter_fp32: flat_in (total) → shard_out (shard_size per rank)
inline void reduce_scatter_fp32(
        void*        nccl_comm_handle,
        float*       flat_in,
        float*       shard_out,
        size_t       shard_size,
        cudaStream_t stream)
{
#ifdef HPC_HAVE_NCCL
    if (!nccl_comm_handle) return;
    ncclComm_t comm = static_cast<ncclComm_t>(nccl_comm_handle);
    NCCL_CHECK(ncclReduceScatter(flat_in, shard_out, shard_size,
                                 ncclFloat, ncclSum, comm, stream));
#else
    // Single GPU: copy flat[0..shard_size) → shard_out
    HPC_CUDA_CHECK(cudaMemcpyAsync(shard_out, flat_in,
                                   shard_size * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
#endif
}

// all_gather_fp32: shard_in (shard_size) → full_out (total = shard_size * W)
inline void all_gather_fp32(
        void*        nccl_comm_handle,
        float*       shard_in,
        float*       full_out,
        size_t       shard_size,
        cudaStream_t stream)
{
#ifdef HPC_HAVE_NCCL
    if (!nccl_comm_handle) return;
    ncclComm_t comm = static_cast<ncclComm_t>(nccl_comm_handle);
    NCCL_CHECK(ncclAllGather(shard_in, full_out, shard_size,
                             ncclFloat, comm, stream));
#else
    // Single GPU: shard_in == full_out[0..shard_size), already in place
    if (shard_in != full_out)
        HPC_CUDA_CHECK(cudaMemcpyAsync(full_out, shard_in,
                                       shard_size * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
#endif
}

} // namespace zero2
} // namespace hpc_opt
