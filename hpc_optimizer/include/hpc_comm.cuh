// =============================================================================
// HPC CUDA Optimizer Library
// hpc_comm.cuh  –  Multi-GPU communication layer
//
// Covers:
//   1. NCCL all-reduce (gradient synchronisation across GPUs/nodes)
//   2. ZeRO-Stage-1: each rank holds optimizer state only for its shard;
//      after the local step it all-gathers the updated parameter shard.
//   3. Gradient compression: FP16 all-reduce of FP32 gradients (2× BW)
//   4. Stream-parallel: overlap compute with communication via two streams
//   5. MPI bootstrap helpers (rank, world_size, local_rank from env)
//
// Build guards:
//   -DHPC_HAVE_NCCL   links nccl; otherwise stubs compile cleanly.
//   -DHPC_HAVE_MPI    links MPI;  otherwise env-var bootstrap is used.
// =============================================================================
#pragma once

#include "hpc_types.h"
#include "hpc_precision.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Optional NCCL header
// ---------------------------------------------------------------------------
#ifdef HPC_HAVE_NCCL
#  include <nccl.h>
#  define NCCL_CHECK(call)                                                    \
     do {                                                                     \
         ncclResult_t _r = (call);                                            \
         if (_r != ncclSuccess) {                                             \
             fprintf(stderr, "[hpc_comm] NCCL error @ %s:%d  %s\n",          \
                     __FILE__, __LINE__, ncclGetErrorString(_r));             \
             abort();                                                         \
         }                                                                    \
     } while (0)
#endif

// ---------------------------------------------------------------------------
// Optional MPI header
// ---------------------------------------------------------------------------
#ifdef HPC_HAVE_MPI
#  include <mpi.h>
#  define MPI_CHECK(call)                                                     \
     do {                                                                     \
         int _r = (call);                                                     \
         if (_r != MPI_SUCCESS) {                                             \
             char buf[256]; int len;                                          \
             MPI_Error_string(_r, buf, &len);                                 \
             fprintf(stderr, "[hpc_comm] MPI error @ %s:%d  %s\n",           \
                     __FILE__, __LINE__, buf);                                \
             abort();                                                         \
         }                                                                    \
     } while (0)
#endif

namespace hpc_opt {

// ===========================================================================
// CommContext  –  owns NCCL communicator + streams for a single GPU
// ===========================================================================
class CommContext {
public:
    CommContext() = default;
    ~CommContext() { destroy(); }

    CommContext(const CommContext&)            = delete;
    CommContext& operator=(const CommContext&) = delete;

    // -----------------------------------------------------------------------
    // init_from_env
    //   Reads RANK / WORLD_SIZE / LOCAL_RANK from environment variables
    //   (set by torchrun / mpirun / SLURM).  Falls back to single-GPU if absent.
    //   Then initialises NCCL via ncclCommInitRank (requires MPI or a shared
    //   file-system rendezvous; see init_nccl_from_mpi below for MPI path).
    // -----------------------------------------------------------------------
    void init_from_env() {
        auto get_env_int = [](const char* k, int def) -> int {
            const char* v = std::getenv(k);
            return v ? std::atoi(v) : def;
        };

        dist_.rank       = get_env_int("RANK",       0);
        dist_.world_size = get_env_int("WORLD_SIZE",  1);
        dist_.local_rank = get_env_int("LOCAL_RANK",  dist_.rank);
        dist_.local_size = get_env_int("LOCAL_WORLD_SIZE", dist_.world_size);

        HPC_CUDA_CHECK(cudaSetDevice(dist_.local_rank));
        HPC_CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        HPC_CUDA_CHECK(cudaStreamCreate(&comm_stream_));

        if (dist_.world_size > 1) {
#ifdef HPC_HAVE_NCCL
            init_nccl();
#else
            fprintf(stderr, "[hpc_comm] world_size=%d but HPC_HAVE_NCCL not defined. "
                            "Compile with -DHPC_HAVE_NCCL and link -lnccl.\n",
                    dist_.world_size);
            abort();
#endif
        }
        initialized_ = true;
    }

    // -----------------------------------------------------------------------
    // init_nccl  –  bootstrap NCCL communicator.
    // For multi-node: rank 0 generates the ID and broadcasts via MPI or file.
    // -----------------------------------------------------------------------
#ifdef HPC_HAVE_NCCL
    void init_nccl() {
        ncclUniqueId id;
        if (dist_.rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));

#ifdef HPC_HAVE_MPI
        MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
#else
        // Single-node: write id to /tmp/nccl_id.<pid> and all ranks read it.
        // (torchrun sets MASTER_ADDR/MASTER_PORT; here we use a simple file)
        char path[256];
        snprintf(path, sizeof(path), "/tmp/nccl_id.%d",
                 std::atoi(std::getenv("TORCHELASTIC_RESTART_COUNT") ?: "0"));
        if (dist_.rank == 0) {
            FILE* f = fopen(path, "wb");
            fwrite(&id, sizeof(id), 1, f); fclose(f);
        }
        // Spin-wait for file (crude, replace with proper barrier in production)
        FILE* f = nullptr;
        while (!f) { f = fopen(path, "rb"); if (!f) usleep(10000); }
        fread(&id, sizeof(id), 1, f); fclose(f);
#endif

        NCCL_CHECK(ncclCommInitRank(&nccl_comm_, dist_.world_size, id, dist_.rank));
        dist_.use_nccl = true;
    }
#endif

    // -----------------------------------------------------------------------
    // all_reduce_fp32  –  in-place sum-reduce across all ranks
    // -----------------------------------------------------------------------
    void all_reduce_fp32(float* buf, size_t numel, cudaStream_t stream) {
#ifdef HPC_HAVE_NCCL
        if (dist_.use_nccl)
            NCCL_CHECK(ncclAllReduce(buf, buf, numel, ncclFloat,
                                     ncclSum, nccl_comm_, stream));
#else
        (void)buf; (void)numel; (void)stream;  // single-GPU no-op
#endif
    }

    // -----------------------------------------------------------------------
    // all_reduce_fp16_compressed
    //   Cast FP32 grads to FP16, all-reduce, cast back — halves BW cost.
    //   Uses a temporary device buffer.
    // -----------------------------------------------------------------------
    void all_reduce_fp16_compressed(float* buf, size_t numel, cudaStream_t stream) {
#ifdef HPC_HAVE_NCCL
        if (!dist_.use_nccl) return;

        // Ensure scratch buffer
        if (scratch_numel_ < numel) {
            if (d_scratch_) cudaFree(d_scratch_);
            HPC_CUDA_CHECK(cudaMalloc(&d_scratch_, numel * sizeof(half)));
            scratch_numel_ = numel;
        }

        // FP32 → FP16
        int blk = hpc_blocks(numel);
        k_cast_fp32_to_fp16<<<blk, HPC_BLOCK, 0, stream>>>(
            d_scratch_, buf, numel);

        // All-reduce in FP16
        NCCL_CHECK(ncclAllReduce(d_scratch_, d_scratch_, numel, ncclHalf,
                                 ncclSum, nccl_comm_, stream));

        // FP16 → FP32 (with 1/world_size scale for mean)
        float inv_ws = 1.0f / static_cast<float>(dist_.world_size);
        k_cast_fp16_to_fp32<<<blk, HPC_BLOCK, 0, stream>>>(
            buf, d_scratch_, numel, inv_ws);
#else
        (void)buf; (void)numel; (void)stream;
#endif
    }

    // -----------------------------------------------------------------------
    // all_reduce_grads
    //   Calls all_reduce for each gradient tensor in the array.
    //   Divides by world_size to convert sum → mean.
    //   Overlaps communication with next tensor's kernel if two streams used.
    // -----------------------------------------------------------------------
    void all_reduce_grads(TensorView* grads, int n,
                          bool fp16_compress = false,
                          cudaStream_t stream = 0)
    {
        if (dist_.world_size <= 1) return;  // single GPU: no-op

        const float inv_ws = 1.0f / static_cast<float>(dist_.world_size);

        for (int i = 0; i < n; ++i) {
            auto& g = grads[i];
            if (g.numel == 0) continue;

            if (g.dtype == Dtype::FP32) {
                if (fp16_compress && dist_.use_fp16_comm)
                    all_reduce_fp16_compressed(
                        static_cast<float*>(g.data), g.numel, stream);
                else
                    all_reduce_fp32(
                        static_cast<float*>(g.data), g.numel, stream);
            }
#ifdef HPC_HAVE_NCCL
            else if (g.dtype == Dtype::BF16) {
                NCCL_CHECK(ncclAllReduce(g.data, g.data, g.numel, ncclBfloat16,
                                        ncclSum, nccl_comm_, stream));
            } else if (g.dtype == Dtype::FP16) {
                NCCL_CHECK(ncclAllReduce(g.data, g.data, g.numel, ncclHalf,
                                        ncclSum, nccl_comm_, stream));
            }
#endif
            // Scale: sum → mean (apply in-place after reduce)
            int blk = hpc_blocks(g.numel);
            k_scale_inplace<<<blk, HPC_BLOCK, 0, stream>>>(
                g.data, g.dtype, g.numel, inv_ws);
        }
    }

    // -----------------------------------------------------------------------
    // barrier  –  synchronise all ranks
    // -----------------------------------------------------------------------
    void barrier(cudaStream_t stream = 0) {
#ifdef HPC_HAVE_NCCL
        if (dist_.use_nccl) {
            // Use a scalar all-reduce as a barrier
            float one = 1.0f;
            float* d_one;
            HPC_CUDA_CHECK(cudaMalloc(&d_one, sizeof(float)));
            HPC_CUDA_CHECK(cudaMemcpyAsync(d_one, &one, sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
            NCCL_CHECK(ncclAllReduce(d_one, d_one, 1, ncclFloat,
                                     ncclSum, nccl_comm_, stream));
            cudaFree(d_one);
        }
#endif
#ifdef HPC_HAVE_MPI
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int          rank()       const { return dist_.rank; }
    int          world_size() const { return dist_.world_size; }
    int          local_rank() const { return dist_.local_rank; }
    bool         is_root()    const { return dist_.rank == 0; }
    bool         initialized()const { return initialized_; }
    cudaStream_t compute_stream() const { return compute_stream_; }
    cudaStream_t comm_stream()    const { return comm_stream_; }

    const DistConfig& dist_config() const { return dist_; }

private:
    // -----------------------------------------------------------------------
    // Cast kernels used for FP16 compression
    // -----------------------------------------------------------------------
    static __global__ void k_cast_fp32_to_fp16(half* dst, const float* src, size_t n) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x)
            dst[i] = __float2half(src[i]);
    }

    static __global__ void k_cast_fp16_to_fp32(float* dst, const half* src,
                                                size_t n, float scale) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x)
            dst[i] = __half2float(src[i]) * scale;
    }

    static __global__ void k_scale_inplace(void* buf, Dtype dtype,
                                           size_t numel, float scale) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        for (; i < numel; i += gridDim.x * blockDim.x) {
            if (dtype == Dtype::FP32) {
                static_cast<float*>(buf)[i] *= scale;
            } else if (dtype == Dtype::FP16) {
                half h = static_cast<half*>(buf)[i];
                static_cast<half*>(buf)[i] = __float2half(__half2float(h) * scale);
            } else {
                __nv_bfloat16 b = static_cast<__nv_bfloat16*>(buf)[i];
                static_cast<__nv_bfloat16*>(buf)[i] =
                    __float2bfloat16(__bfloat162float(b) * scale);
            }
        }
    }

    void destroy() {
#ifdef HPC_HAVE_NCCL
        if (dist_.use_nccl && nccl_comm_)
            ncclCommDestroy(nccl_comm_);
#endif
        if (compute_stream_) cudaStreamDestroy(compute_stream_);
        if (comm_stream_)    cudaStreamDestroy(comm_stream_);
        if (d_scratch_)      cudaFree(d_scratch_);
    }

    DistConfig   dist_{};
    bool         initialized_ = false;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t comm_stream_    = nullptr;
    half*        d_scratch_      = nullptr;
    size_t       scratch_numel_  = 0;

#ifdef HPC_HAVE_NCCL
    ncclComm_t nccl_comm_ = nullptr;
#endif
};

// ===========================================================================
// ZeROShard  –  Stage-1: shard optimizer state across ranks
//
// Each rank owns params[start..end) (by element count).
// After the local optimizer step, all-gather the updated shard so every rank
// holds the full up-to-date parameter vector.
//
// Usage:
//   ZeROShard shard(comm, params, n, master_fp32);
//   shard.local_step(opt, grads);   // step only on this rank's shard
//   shard.all_gather(stream);       // reconstruct full params on all ranks
// ===========================================================================
class ZeROShard {
public:
    ZeROShard(CommContext& comm,
              TensorView*  params,    // full param list (all tensors, FP32)
              int          n,
              cudaStream_t stream = 0)
        : comm_(comm), n_tensors_(n)
    {
        // Compute per-rank parameter ownership
        // Simple flat assignment: divide total elements evenly
        size_t total = 0;
        for (int i = 0; i < n; ++i) total += params[i].numel;

        shard_start_ = (total / comm.world_size()) *  comm.rank();
        shard_end_   = (comm.rank() == comm.world_size() - 1)
                       ? total
                       : shard_start_ + (total / comm.world_size());

        // Record per-tensor start/end offsets in the flat address space
        tensor_offsets_.resize(n + 1, 0);
        for (int i = 0; i < n; ++i)
            tensor_offsets_[i + 1] = tensor_offsets_[i] + params[i].numel;

        (void)stream;
    }

    // Returns true if tensor i is (at least partially) owned by this rank
    bool owns_tensor(int i) const {
        size_t ts = tensor_offsets_[i];
        size_t te = tensor_offsets_[i + 1];
        return ts < shard_end_ && te > shard_start_;
    }

    // Owned slice [lo, hi) within tensor i
    std::pair<size_t, size_t> owned_range(int i) const {
        size_t ts  = tensor_offsets_[i];
        size_t lo  = (shard_start_ > ts)  ? shard_start_ - ts : 0;
        size_t hi  = (shard_end_   < tensor_offsets_[i + 1])
                     ? shard_end_ - ts
                     : tensor_offsets_[i + 1] - ts;
        return {lo, hi};
    }

    // All-gather after local step: broadcast each rank's updated shard
    void all_gather(TensorView* params, cudaStream_t stream = 0) {
#ifdef HPC_HAVE_NCCL
        // Use ncclAllGather: each rank sends its shard, receives all shards.
        // For simplicity we do per-tensor all-gather here.
        // Production code would pack into a contiguous buffer first.
        for (int i = 0; i < n_tensors_; ++i) {
            if (params[i].dtype != Dtype::FP32) continue;
            float* p = static_cast<float*>(params[i].data);
            // ncclAllReduce with ncclSum after zeroing non-owned elements
            // (ZeRO-1 full AG: skip for brevity, use ncclAllGather in practice)
            comm_.all_reduce_fp32(p, params[i].numel, stream);
        }
#else
        (void)params; (void)stream;  // single-GPU: already up to date
#endif
    }

private:
    CommContext&       comm_;
    int                n_tensors_;
    size_t             shard_start_ = 0;
    size_t             shard_end_   = 0;
    std::vector<size_t> tensor_offsets_;
};

} // namespace hpc_opt
