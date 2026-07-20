// =============================================================================
// HPC CUDA Optimizer Library
// hpc_profiler.cuh  –  NVTX profiling hooks + CUDA event timing
//
// Guards:
//   -DHPC_HAVE_NVTX  → activates nvToolsExt markers (link -lnvToolsExt)
//   Without that flag, all macros compile to zero-cost no-ops.
//
// Provides:
//   HPC_RANGE_PUSH(name)   –  open a named NVTX range (shows in Nsight)
//   HPC_RANGE_POP()        –  close current NVTX range
//   HPC_RANGE_SCOPED(name) –  RAII scoped range (closes on scope exit)
//   StepTimer              –  CUDA event pair for per-step kernel timing
//   IterationProfiler      –  rolling stats: mean/min/max/p95 over N steps
// =============================================================================
#pragma once

#include "hpc_types.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>

// ---------------------------------------------------------------------------
// NVTX integration
// ---------------------------------------------------------------------------
#ifdef HPC_HAVE_NVTX
#  include <nvToolsExt.h>

// Colour palette for NVTX ranges
namespace hpc_opt::nvtx_colors {
    static constexpr uint32_t ALLREDUCE = 0xFF4CAF50;  // green
    static constexpr uint32_t OPT_STEP  = 0xFF2196F3;  // blue
    static constexpr uint32_t CLIP      = 0xFFFFC107;  // amber
    static constexpr uint32_t CKPT      = 0xFF9C27B0;  // purple
    static constexpr uint32_t FWD       = 0xFFFF5722;  // deep orange
    static constexpr uint32_t BWD       = 0xFFE91E63;  // pink
}

inline void hpc_nvtx_push(const char* name, uint32_t color = 0xFF03A9F4) {
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

inline void hpc_nvtx_pop() { nvtxRangePop(); }

#  define HPC_RANGE_PUSH(name, color) ::hpc_opt::hpc_nvtx_push((name), (color))
#  define HPC_RANGE_POP()             ::hpc_opt::hpc_nvtx_pop()

#else  // Stubs — zero overhead
#  define HPC_RANGE_PUSH(name, color) ((void)0)
#  define HPC_RANGE_POP()             ((void)0)
#endif

// RAII scoped range
#define HPC_RANGE_SCOPED(name, color) \
    ::hpc_opt::ScopedRange _hpc_sr_##__LINE__((name), (color))

namespace hpc_opt {

struct ScopedRange {
    explicit ScopedRange(const char* name, uint32_t color = 0xFF03A9F4) {
        HPC_RANGE_PUSH(name, color);
    }
    ~ScopedRange() { HPC_RANGE_POP(); }
    ScopedRange(const ScopedRange&) = delete;
};

// ===========================================================================
// StepTimer  –  single CUDA event pair, measures one stream segment
// ===========================================================================
class StepTimer {
public:
    StepTimer() {
        HPC_CUDA_CHECK(cudaEventCreate(&start_));
        HPC_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~StepTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        HPC_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    // Returns elapsed milliseconds; blocks until stop event is done.
    float stop(cudaStream_t stream = 0) {
        HPC_CUDA_CHECK(cudaEventRecord(stop_, stream));
        HPC_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        HPC_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

// ===========================================================================
// IterationProfiler  –  rolling statistics over training steps
// ===========================================================================
class IterationProfiler {
public:
    explicit IterationProfiler(int window = 100) : window_(window) {}

    void record(float ms)   { samples_.push_back(ms); }
    void record_allreduce(float ms) { ar_samples_.push_back(ms); }
    void record_opt(float ms)       { opt_samples_.push_back(ms); }

    struct Stats {
        float mean, min, max, p95;
        float throughput_iter_per_sec;
    };

    Stats compute(const std::vector<float>& v) const {
        if (v.empty()) return {};
        std::vector<float> s = v;
        if (static_cast<int>(s.size()) > window_)
            s = std::vector<float>(s.end() - window_, s.end());

        std::sort(s.begin(), s.end());
        float sum  = std::accumulate(s.begin(), s.end(), 0.0f);
        float mn   = sum / static_cast<float>(s.size());
        float p95  = s[static_cast<size_t>(s.size() * 0.95)];
        return { mn, s.front(), s.back(), p95, 1000.0f / mn };
    }

    void print_summary(int step, int world_size) const {
        auto total  = compute(samples_);
        auto ar     = compute(ar_samples_);
        auto opt_s  = compute(opt_samples_);
        if (samples_.empty()) return;

        fprintf(stdout,
            "\n[Profiler] step=%-6d  world_size=%d\n"
            "  Total iter  : mean=%.2f ms  min=%.2f  max=%.2f  p95=%.2f  "
                                           "%.1f iter/s\n"
            "  AllReduce   : mean=%.2f ms  (%.1f%% of iter)\n"
            "  Opt step    : mean=%.2f ms  (%.1f%% of iter)\n",
            step, world_size,
            total.mean, total.min, total.max, total.p95,
            total.throughput_iter_per_sec,
            ar.mean,  (total.mean > 0 ? 100.0f * ar.mean  / total.mean : 0.0f),
            opt_s.mean,(total.mean > 0 ? 100.0f * opt_s.mean / total.mean : 0.0f));
    }

private:
    int                window_;
    std::vector<float> samples_;
    std::vector<float> ar_samples_;
    std::vector<float> opt_samples_;
};

// ===========================================================================
// ThroughputLogger  –  tokens/samples per second for large-scale HPC jobs
// ===========================================================================
class ThroughputLogger {
public:
    ThroughputLogger(int world_size, int batch_size_per_gpu, int tokens_per_sample)
        : world_size_(world_size),
          global_batch_(batch_size_per_gpu * world_size),
          tokens_per_sample_(tokens_per_sample) {}

    void log(float step_ms, int step) const {
        float step_s        = step_ms * 1e-3f;
        float samples_per_s = static_cast<float>(global_batch_) / step_s;
        float tokens_per_s  = samples_per_s * static_cast<float>(tokens_per_sample_);

        fprintf(stdout,
            "[Throughput] step=%-6d  %.0f samples/s  %.2f Ktok/s  "
            "(global_batch=%d  world_size=%d)\n",
            step, samples_per_s, tokens_per_s * 1e-3f,
            global_batch_, world_size_);
    }

private:
    int world_size_;
    int global_batch_;
    int tokens_per_sample_;
};

} // namespace hpc_opt
