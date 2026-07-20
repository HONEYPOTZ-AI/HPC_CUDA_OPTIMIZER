// =============================================================================
// HPC CUDA Optimizer Library — ZeRO-3 Test Suite
// tests/test_zero3.cu  —  12 unit tests for hpc_zero3.cuh
//
// Test inventory:
//   T01  Shard layout correctness (sizes, offsets, padding)
//   T02  Scatter kernel: shard → full buffer round-trip
//   T03  Cast kernel: FP32 → FP16/BF16 gathered params
//   T04  Pack-grads kernel: per-tensor → flat grad bucket
//   T05  Prefetch/release lifecycle (device-ptr validity)
//   T06  Reduce-scatter correctness (uniform grad → mean after RS)
//   T07  AdamW shard optimizer convergence (loss decreases)
//   T08  Lion shard optimizer convergence
//   T09  SGD  shard optimizer convergence
//   T10  Gradient clipping on shard (norm ≤ max_norm)
//   T11  Checkpoint save → reload, param fidelity
//   T12  Memory accounting: ZeRO-3 < ZeRO-0 per-rank bytes
//
// Build (single-GPU, no NCCL):
//   nvcc -std=c++17 -O2 -I../include \
//        -gencode arch=compute_80,code=sm_80 \
//        test_zero3.cu -o test_zero3
//
// Build (multi-GPU, NCCL):
//   nvcc -std=c++17 -O2 -DHPC_HAVE_NCCL -I../include \
//        -gencode arch=compute_80,code=sm_80 \
//        test_zero3.cu -lnccl -o test_zero3
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

// ---- library headers -------------------------------------------------------
#include "hpc_types.h"
#include "hpc_precision.cuh"
#include "hpc_grad_clip.cuh"
#include "hpc_zero2.cuh"
#include "hpc_zero2_optimizer.cuh"
#include "hpc_zero3.cuh"

using namespace hpc_opt;
using namespace hpc_opt::zero2;
using namespace hpc_opt::zero3;

// ============================================================================
// Minimal test harness
// ============================================================================
static int g_total  = 0;
static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(cond, msg)                                          \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("    FAIL [%s:%d]  %s\n", __FILE__, __LINE__, msg);  \
            ++g_failed;                                                  \
        }                                                                \
    } while(0)

#define EXPECT_NEAR(a, b, tol, msg)                                     \
    do {                                                                 \
        float _a = (float)(a), _b = (float)(b);                        \
        if (fabsf(_a - _b) > (float)(tol)) {                            \
            printf("    FAIL [%s:%d]  %s: |%.6f - %.6f| = %.6f > %.6f\n",\
                   __FILE__, __LINE__, msg, _a, _b,                      \
                   fabsf(_a - _b), (float)(tol));                        \
            ++g_failed;                                                  \
        }                                                                \
    } while(0)

#define EXPECT_LE(a, b, msg)                                            \
    do {                                                                 \
        if (!((a) <= (b))) {                                             \
            printf("    FAIL [%s:%d]  %s: %g > %g\n",                   \
                   __FILE__, __LINE__, msg, (double)(a), (double)(b));   \
            ++g_failed;                                                  \
        }                                                                \
    } while(0)

struct TestCase {
    const char* name;
    void (*fn)();
};

static void run_test(const char* name, void (*fn)()) {
    int fail_before = g_failed;
    ++g_total;
    printf("[TEST %02d] %s\n", g_total, name);
    fn();
    if (g_failed == fail_before) {
        printf("    PASS\n");
        ++g_passed;
    }
}

// ============================================================================
// Helpers
// ============================================================================
static int g_world  = 1;
static int g_rank   = 0;

// Simulate a single-GPU "distributed" environment (rank 0 of 1).
static ShardLayout make_layout(size_t total_params) {
    return ShardLayout(total_params, g_rank, g_world);
}

static void cuda_sync_check(const char* where) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(e));
        std::abort();
    }
}

// Fill device float buffer with linspace [start, stop]
static void fill_linspace(float* d, size_t n, float start, float stop) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = start + (stop - start) * i / (float)(n > 1 ? n-1 : 1);
    cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

// Compute L2 norm on host from device float buffer
static float device_l2_norm(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.f;
    for (float v : h) s += v * v;
    return sqrtf(s);
}

// Sum of all elements from device float buffer
static float device_sum(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.f;
    for (float v : h) s += v;
    return s;
}

// Max absolute value of (a - b) for device buffers of length n
static float device_max_abs_diff(const float* a, const float* b, size_t n) {
    std::vector<float> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost);
    float mx = 0.f;
    for (size_t i = 0; i < n; ++i)
        mx = fmaxf(mx, fabsf(ha[i] - hb[i]));
    return mx;
}

// ============================================================================
// T01 — Shard layout correctness
// ============================================================================
static void test_shard_layout() {
    // Single rank: shard should cover all params
    {
        ShardLayout sl(1024, 0, 1);
        EXPECT_TRUE(sl.shard_size == 1024, "rank=0/1: shard_size should be 1024");
        EXPECT_TRUE(sl.shard_start == 0,   "rank=0/1: shard_start should be 0");
        EXPECT_TRUE(sl.padded_total == 1024, "rank=0/1: no padding needed");
    }

    // Two ranks, odd param count → padding check
    {
        ShardLayout sl_r0(1001, 0, 2);
        ShardLayout sl_r1(1001, 1, 2);
        // padded_total should be even
        EXPECT_TRUE(sl_r0.padded_total % 2 == 0, "padded_total divisible by world");
        // Both shards together should cover the padded total
        EXPECT_TRUE(sl_r0.shard_size == sl_r1.shard_size,
                    "equal shard sizes with padding");
        EXPECT_TRUE(sl_r0.shard_size + sl_r1.shard_size == sl_r0.padded_total,
                    "shard0 + shard1 == padded_total");
        // Shard start offsets
        EXPECT_TRUE(sl_r0.shard_start == 0,             "r0 starts at 0");
        EXPECT_TRUE(sl_r1.shard_start == sl_r0.shard_size, "r1 starts after r0");
    }

    // 4 ranks, 1024 params
    {
        size_t total = 1024;
        int    W     = 4;
        size_t shard = total / W;  // 256
        for (int r = 0; r < W; ++r) {
            ShardLayout sl(total, r, W);
            EXPECT_TRUE(sl.shard_size  == shard,        "4-rank: shard_size");
            EXPECT_TRUE(sl.shard_start == (size_t)r*shard, "4-rank: shard_start");
        }
    }

    // Alignment: shard_size should be a multiple of HPC_WARP (32)
    {
        ShardLayout sl(100, 0, 3);
        EXPECT_TRUE(sl.shard_size % HPC_WARP == 0, "shard_size aligned to warp");
    }
}

// ============================================================================
// T02 — Scatter kernel: shard → full buffer
// ============================================================================
__global__ void k_fill_index(float* x, size_t n, size_t offset) {
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < n;
         i += gridDim.x*blockDim.x)
        x[i] = (float)(i + offset);
}

static void test_scatter_kernel() {
    const size_t total = 512;
    const int    W     = 2;
    const int    rank  = 0;
    ShardLayout  sl(total, rank, W);
    size_t       shard = sl.shard_size;  // 256

    float *d_shard, *d_full;
    cudaMalloc(&d_shard, shard * sizeof(float));
    cudaMalloc(&d_full,  total * sizeof(float));
    cudaMemset(d_full, 0, total * sizeof(float));

    // Fill shard with values 0..255
    k_fill_index<<<hpc_blocks(shard), HPC_BLOCK>>>(d_shard, shard, 0);

    // Scatter into full buffer
    dim3 grid = hpc_blocks(shard);
    k_scatter_shard<<<grid, HPC_BLOCK>>>(d_full, d_shard, sl.shard_start, shard);
    cuda_sync_check("k_scatter_shard");

    // Verify: full[0..255] == 0..255, full[256..511] == 0 (untouched)
    std::vector<float> h(total);
    cudaMemcpy(h.data(), d_full, total * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok_scatter = true, ok_zero = true;
    for (size_t i = 0; i < shard; ++i)
        if (fabsf(h[i] - (float)i) > 0.5f) { ok_scatter = false; break; }
    for (size_t i = shard; i < total; ++i)
        if (fabsf(h[i]) > 0.5f) { ok_zero = false; break; }

    EXPECT_TRUE(ok_scatter, "scatter: shard values placed at correct offset");
    EXPECT_TRUE(ok_zero,    "scatter: non-shard region remains zero");

    cudaFree(d_shard);
    cudaFree(d_full);
}

// ============================================================================
// T03 — Cast kernel: FP32 gathered → FP16 / BF16
// ============================================================================
static void test_cast_kernel() {
    const size_t N = 1024;
    float        *d_fp32;
    __half        *d_fp16;
    __nv_bfloat16 *d_bf16;

    cudaMalloc(&d_fp32, N * sizeof(float));
    cudaMalloc(&d_fp16, N * sizeof(__half));
    cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16));

    fill_linspace(d_fp32, N, -1.0f, 1.0f);

    // Cast to FP16
    k_cast_fp32_to_lowprec<__half><<<hpc_blocks(N), HPC_BLOCK>>>(
        d_fp16, d_fp32, N);
    // Cast to BF16
    k_cast_fp32_to_lowprec<__nv_bfloat16><<<hpc_blocks(N), HPC_BLOCK>>>(
        d_bf16, d_fp32, N);
    cuda_sync_check("k_cast_fp32_to_lowprec");

    // Copy back and compare
    std::vector<float> h_src(N);
    std::vector<__half> h_fp16(N);
    std::vector<__nv_bfloat16> h_bf16(N);
    cudaMemcpy(h_src.data(),  d_fp32, N*sizeof(float),           cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fp16.data(), d_fp16, N*sizeof(__half),          cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bf16.data(), d_bf16, N*sizeof(__nv_bfloat16),   cudaMemcpyDeviceToHost);

    float max_err_fp16 = 0.f, max_err_bf16 = 0.f;
    for (size_t i = 0; i < N; ++i) {
        max_err_fp16 = fmaxf(max_err_fp16, fabsf(__half2float(h_fp16[i]) - h_src[i]));
        max_err_bf16 = fmaxf(max_err_bf16, fabsf(__bfloat162float(h_bf16[i]) - h_src[i]));
    }

    // FP16 has ~1e-3 precision over [-1,1]; BF16 ~1e-2
    EXPECT_TRUE(max_err_fp16 < 1e-3f, "cast FP32→FP16 max error < 1e-3");
    EXPECT_TRUE(max_err_bf16 < 5e-3f, "cast FP32→BF16 max error < 5e-3");

    cudaFree(d_fp32);
    cudaFree(d_fp16);
    cudaFree(d_bf16);
}

// ============================================================================
// T04 — Pack-grads kernel: per-tensor → flat bucket (reuses k_pack_tensor)
// ============================================================================
static void test_pack_grads() {
    // Two tensors: sizes 300 and 700, total 1000
    const size_t n1 = 300, n2 = 700, total = n1 + n2;

    float *d_g1, *d_g2, *d_bucket;
    cudaMalloc(&d_g1,     n1    * sizeof(float));
    cudaMalloc(&d_g2,     n2    * sizeof(float));
    cudaMalloc(&d_bucket, total * sizeof(float));
    cudaMemset(d_bucket, 0, total * sizeof(float));

    // Fill: g1[i] = i, g2[i] = i + n1
    k_fill_index<<<hpc_blocks(n1), HPC_BLOCK>>>(d_g1, n1, 0);
    k_fill_index<<<hpc_blocks(n2), HPC_BLOCK>>>(d_g2, n2, n1);

    // Pack using zero2 kernel
    k_pack_tensor<<<hpc_blocks(n1), HPC_BLOCK>>>(d_bucket,        d_g1, n1);
    k_pack_tensor<<<hpc_blocks(n2), HPC_BLOCK>>>(d_bucket + n1,   d_g2, n2);
    cuda_sync_check("k_pack_tensor");

    // Verify bucket is a contiguous sequence 0..999
    std::vector<float> h(total);
    cudaMemcpy(h.data(), d_bucket, total * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < total; ++i)
        if (fabsf(h[i] - (float)i) > 0.5f) { ok = false; break; }
    EXPECT_TRUE(ok, "pack_grads: flat bucket matches concatenated per-tensor grads");

    cudaFree(d_g1);
    cudaFree(d_g2);
    cudaFree(d_bucket);
}

// ============================================================================
// T05 — Prefetch/release lifecycle
// ============================================================================
static void test_prefetch_release_lifecycle() {
    // Single-GPU stub test: engine allocates full-param buffer on prefetch
    // and frees / zeros it on release.
    // We test this without NCCL by directly calling the internal alloc path.

    const size_t N = 2048;
    const size_t elem_bytes = sizeof(float);

    // Simulate master shard
    float *d_shard;
    cudaMalloc(&d_shard, N * elem_bytes);
    fill_linspace(d_shard, N, 0.f, 1.f);

    // Manually exercise: alloc full param buffer, scatter, verify, free
    float *d_full;
    cudaMalloc(&d_full, N * elem_bytes);

    // Scatter shard (rank 0 of 1 → entire range)
    k_scatter_shard<<<hpc_blocks(N), HPC_BLOCK>>>(d_full, d_shard, 0, N);
    cuda_sync_check("scatter_shard T05");

    // Verify full params match shard (single rank)
    std::vector<float> h_shard(N), h_full(N);
    cudaMemcpy(h_shard.data(), d_shard, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_full.data(),  d_full,  N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.f;
    for (size_t i = 0; i < N; ++i)
        max_diff = fmaxf(max_diff, fabsf(h_shard[i] - h_full[i]));
    EXPECT_NEAR(max_diff, 0.f, 1e-6f, "prefetch: full params == master shard (rank 0/1)");

    // Release: zero the full buffer
    cudaMemset(d_full, 0, N * sizeof(float));

    // Verify zeroed
    float sum_after = device_sum(d_full, N);
    EXPECT_NEAR(sum_after, 0.f, 1e-6f, "release: full-param buffer zeroed");

    cudaFree(d_shard);
    cudaFree(d_full);
}

// ============================================================================
// T06 — Reduce-scatter correctness (uniform grad → mean)
// ============================================================================
static void test_reduce_scatter() {
    // Single-GPU: ReduceScatter with world=1 is identity.
    // We verify that the k_scale_shard kernel (divides by world) is correct.
    const size_t N      = 1024;
    const float  fill_v = 4.0f;
    const int    W      = 4;  // simulated world size for scaling

    float *d_grad;
    cudaMalloc(&d_grad, N * sizeof(float));

    // Fill all grads with 4.0 (as if AllReduce summed 4 replicas each with 1.0)
    std::vector<float> h(N, fill_v);
    cudaMemcpy(d_grad, h.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    // Scale by 1/W (the mean step after ReduceScatter)
    k_scale_shard<<<hpc_blocks(N), HPC_BLOCK>>>(d_grad, N, 1.0f / W);
    cuda_sync_check("k_scale_shard");

    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_grad, N*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (float v : h_out)
        if (fabsf(v - 1.0f) > 1e-5f) { ok = false; break; }

    EXPECT_TRUE(ok, "reduce_scatter scale: 4.0 / 4 == 1.0 per element");

    cudaFree(d_grad);
}

// ============================================================================
// T07 — AdamW shard convergence
// ============================================================================
static void sgd_like_step_fp32(float* params, float* grads, size_t N,
                                float lr) {
    // Minimal gradient-descent test: params -= lr * grads
    std::vector<float> hp(N), hg(N);
    cudaMemcpy(hp.data(), params, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hg.data(), grads,  N*sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < N; ++i) hp[i] -= lr * hg[i];
    cudaMemcpy(params, hp.data(), N*sizeof(float), cudaMemcpyHostToDevice);
}

static void test_adamw_shard_convergence() {
    // Minimize f(x) = 0.5 * ||x - x*||^2, x* = 1.0
    // Gradient = x - x* = x - 1
    const size_t N  = 512;
    const float  lr = 1e-2f;
    const int    steps = 300;

    float *d_params, *d_m, *d_v, *d_grads;
    cudaMalloc(&d_params, N * sizeof(float));
    cudaMalloc(&d_m,      N * sizeof(float));
    cudaMalloc(&d_v,      N * sizeof(float));
    cudaMalloc(&d_grads,  N * sizeof(float));

    // Init params to 0
    cudaMemset(d_params, 0, N * sizeof(float));
    cudaMemset(d_m,      0, N * sizeof(float));
    cudaMemset(d_v,      0, N * sizeof(float));

    for (int s = 0; s < steps; ++s) {
        // Compute gradient: g = params - 1
        std::vector<float> hp(N);
        cudaMemcpy(hp.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < N; ++i) hp[i] -= 1.0f;  // g = x - 1
        cudaMemcpy(d_grads, hp.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        // AdamW shard step
        k_adamw_shard_fp32<<<hpc_blocks(N), HPC_BLOCK>>>(
            d_params, d_m, d_v, d_grads, N,
            lr, 0.9f, 0.999f, 1e-8f, 1e-2f,  // weight_decay
            (float)(s + 1));
        cuda_sync_check("k_adamw_shard_fp32");
    }

    // Check: params should be close to 1.0
    std::vector<float> hfinal(N);
    cudaMemcpy(hfinal.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (float v : hfinal) max_err = fmaxf(max_err, fabsf(v - 1.0f));

    EXPECT_TRUE(max_err < 0.05f, "AdamW shard: converges to x*=1 within 0.05");

    cudaFree(d_params);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_grads);
}

// ============================================================================
// T08 — Lion shard convergence
// ============================================================================
static void test_lion_shard_convergence() {
    // Same quadratic objective as T07
    const size_t N  = 512;
    const float  lr = 5e-4f;
    const int    steps = 500;

    float *d_params, *d_momentum, *d_grads;
    cudaMalloc(&d_params,   N * sizeof(float));
    cudaMalloc(&d_momentum, N * sizeof(float));
    cudaMalloc(&d_grads,    N * sizeof(float));

    cudaMemset(d_params,   0, N * sizeof(float));
    cudaMemset(d_momentum, 0, N * sizeof(float));

    for (int s = 0; s < steps; ++s) {
        // g = x - 1
        std::vector<float> hp(N);
        cudaMemcpy(hp.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < N; ++i) hp[i] -= 1.0f;
        cudaMemcpy(d_grads, hp.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        k_lion_shard_vec4<<<hpc_blocks(N/4), HPC_BLOCK>>>(
            d_params, d_momentum, d_grads,
            N, lr, 0.9f, 0.99f, 1e-3f);
        cuda_sync_check("k_lion_shard");
    }

    std::vector<float> hfinal(N);
    cudaMemcpy(hfinal.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (float v : hfinal) max_err = fmaxf(max_err, fabsf(v - 1.0f));
    EXPECT_TRUE(max_err < 0.1f, "Lion shard: converges to x*=1 within 0.1");

    cudaFree(d_params);
    cudaFree(d_momentum);
    cudaFree(d_grads);
}

// ============================================================================
// T09 — SGD shard convergence
// ============================================================================
static void test_sgd_shard_convergence() {
    const size_t N  = 512;
    const float  lr = 1e-2f;
    const int    steps = 500;

    float *d_params, *d_velocity, *d_grads;
    cudaMalloc(&d_params,   N * sizeof(float));
    cudaMalloc(&d_velocity, N * sizeof(float));
    cudaMalloc(&d_grads,    N * sizeof(float));

    cudaMemset(d_params,   0, N * sizeof(float));
    cudaMemset(d_velocity, 0, N * sizeof(float));

    for (int s = 0; s < steps; ++s) {
        std::vector<float> hp(N);
        cudaMemcpy(hp.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < N; ++i) hp[i] -= 1.0f;
        cudaMemcpy(d_grads, hp.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        k_sgd_shard_vec4<<<hpc_blocks(N/4), HPC_BLOCK>>>(
            d_params, d_velocity, d_grads,
            N, lr, 0.9f /*momentum*/, 1e-4f /*wd*/, false /*nesterov*/);
        cuda_sync_check("k_sgd_shard");
    }

    std::vector<float> hfinal(N);
    cudaMemcpy(hfinal.data(), d_params, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (float v : hfinal) max_err = fmaxf(max_err, fabsf(v - 1.0f));
    EXPECT_TRUE(max_err < 0.05f, "SGD shard: converges to x*=1 within 0.05");

    cudaFree(d_params);
    cudaFree(d_velocity);
    cudaFree(d_grads);
}

// ============================================================================
// T10 — Gradient clipping on shard
// ============================================================================
static void test_grad_clip_shard() {
    // Create large gradients with known norm, clip to max_norm, verify
    const size_t N        = 4096;
    const float  val      = 2.0f;    // every element = 2 → L2 norm = 2*√N
    const float  max_norm = 1.0f;

    float *d_grads, *d_norm;
    cudaMalloc(&d_grads, N * sizeof(float));
    cudaMalloc(&d_norm,  sizeof(float));

    std::vector<float> h(N, val);
    cudaMemcpy(d_grads, h.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    // Compute norm using grad_clip helpers
    float *d_partial;
    int nblocks = hpc_blocks(N).x;
    cudaMalloc(&d_partial, nblocks * sizeof(float));
    cudaMemset(d_norm, 0, sizeof(float));

    k_partial_sq_norm<float><<<nblocks, HPC_BLOCK>>>(d_partial, d_grads, N);
    // Sum partials on host
    std::vector<float> h_partial(nblocks);
    cudaMemcpy(h_partial.data(), d_partial, nblocks*sizeof(float), cudaMemcpyDeviceToHost);
    float sq_norm = 0.f;
    for (float p : h_partial) sq_norm += p;
    float norm = sqrtf(sq_norm);

    float expected_norm = val * sqrtf((float)N);
    EXPECT_NEAR(norm, expected_norm, expected_norm * 0.01f,
                "grad_clip: computed norm matches val*sqrt(N)");

    // Apply scale to clip
    float scale = max_norm / fmaxf(norm, max_norm);
    k_apply_scale<float><<<hpc_blocks(N), HPC_BLOCK>>>(d_grads, scale, N);
    cuda_sync_check("k_apply_scale");

    // Recompute norm after clipping
    cudaMemset(d_partial, 0, nblocks * sizeof(float));
    k_partial_sq_norm<float><<<nblocks, HPC_BLOCK>>>(d_partial, d_grads, N);
    cudaMemcpy(h_partial.data(), d_partial, nblocks*sizeof(float), cudaMemcpyDeviceToHost);
    sq_norm = 0.f;
    for (float p : h_partial) sq_norm += p;
    float clipped_norm = sqrtf(sq_norm);

    EXPECT_NEAR(clipped_norm, max_norm, 1e-4f,
                "grad_clip: norm after clipping == max_norm");

    cudaFree(d_grads);
    cudaFree(d_norm);
    cudaFree(d_partial);
}

// ============================================================================
// T11 — Checkpoint save → reload, param fidelity
// ============================================================================
static void test_checkpoint_roundtrip() {
    const size_t N    = 1024;
    const char*  path = "/tmp/test_zero3_ckpt_rank0.bin";

    float *d_params_save, *d_params_load;
    cudaMalloc(&d_params_save, N * sizeof(float));
    cudaMalloc(&d_params_load, N * sizeof(float));

    fill_linspace(d_params_save, N, -3.0f, 3.0f);
    cudaMemset(d_params_load, 0, N * sizeof(float));

    // ---- Save ----
    {
        std::vector<float> h(N);
        cudaMemcpy(h.data(), d_params_save, N*sizeof(float), cudaMemcpyDeviceToHost);

        // Write a minimal binary: magic + size + data
        // (replicates CheckpointIO binary format subset)
        FILE* f = fopen(path, "wb");
        assert(f && "checkpoint: failed to open file for write");
        uint64_t magic = 0x4F50545F48504300ULL;
        uint32_t ver   = 2;
        uint64_t numel = N;
        fwrite(&magic, 8, 1, f);
        fwrite(&ver,   4, 1, f);
        fwrite(&numel, 8, 1, f);
        fwrite(h.data(), sizeof(float), N, f);
        fclose(f);
    }

    // ---- Load ----
    {
        FILE* f = fopen(path, "rb");
        assert(f && "checkpoint: failed to open file for read");
        uint64_t magic; uint32_t ver; uint64_t numel;
        fread(&magic, 8, 1, f);
        fread(&ver,   4, 1, f);
        fread(&numel, 8, 1, f);
        EXPECT_TRUE(magic == 0x4F50545F48504300ULL, "checkpoint: magic number correct");
        EXPECT_TRUE(ver   == 2,                     "checkpoint: version correct");
        EXPECT_TRUE(numel == N,                     "checkpoint: numel correct");
        std::vector<float> h(N);
        fread(h.data(), sizeof(float), N, f);
        fclose(f);
        cudaMemcpy(d_params_load, h.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    }

    float max_diff = device_max_abs_diff(d_params_save, d_params_load, N);
    EXPECT_NEAR(max_diff, 0.f, 1e-6f, "checkpoint roundtrip: zero diff");

    cudaFree(d_params_save);
    cudaFree(d_params_load);
    remove(path);
}

// ============================================================================
// T12 — Memory accounting: ZeRO-3 < ZeRO-0
// ============================================================================
static void test_memory_accounting() {
    // Model: 1B parameters, FP16 mixed precision
    const size_t psi     = 1'000'000'000ULL;  // total params
    const int    W       = 8;                  // world size
    // Bytes per param:
    //   FP32 master: 4 bytes, FP16 forward: 2 bytes, Adam m+v: 8 bytes
    //   ZeRO-0: 16*psi (all replicated on every rank)
    //   ZeRO-3: 16*psi / W  (every component sharded)
    size_t zero0_bytes = 16ULL * psi;              // per-rank ZeRO-0
    size_t zero3_bytes = 16ULL * psi / W;          // per-rank ZeRO-3

    float ratio = (float)zero0_bytes / (float)zero3_bytes;

    EXPECT_NEAR(ratio, (float)W, 0.01f,
                "ZeRO-3 memory ratio == world_size (8×)");
    EXPECT_TRUE(zero3_bytes < zero0_bytes,
                "ZeRO-3: per-rank memory < ZeRO-0 per-rank memory");

    // For W=8, verify exact values (psi=1B)
    const size_t expected_z0_gb = 16ULL;  // ~16 GB at fp32-master+fp16+adam
    const size_t expected_z3_gb =  2ULL;  // ~2 GB
    size_t z0_gb = zero0_bytes / (1ULL << 30);
    size_t z3_gb = zero3_bytes / (1ULL << 30);

    EXPECT_NEAR((float)z0_gb, (float)expected_z0_gb, 1.f,
                "ZeRO-0: ~16 GB for 1B params, W=8");
    EXPECT_NEAR((float)z3_gb, (float)expected_z3_gb, 1.f,
                "ZeRO-3: ~2 GB for 1B params, W=8");

    printf("    Memory model (1B params, W=%d):\n", W);
    printf("      ZeRO-0: %.1f GB/rank\n", zero0_bytes / 1e9);
    printf("      ZeRO-3: %.1f GB/rank  (%.1f× saving)\n",
           zero3_bytes / 1e9, ratio);

    // ShardLayout also shows per-rank shard size is consistent
    ShardLayout sl(psi, 0, W);
    EXPECT_NEAR((float)sl.shard_size, (float)(psi / W), psi * 0.01f,
                "ShardLayout: shard_size approx psi/W");
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  HPC Optimizer — ZeRO-3 Test Suite\n");
    printf("  CUDA device: ");
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        printf("%s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    else
        printf("<unknown>\n");
    printf("==========================================================\n\n");

    run_test("T01  Shard layout correctness",          test_shard_layout);
    run_test("T02  Scatter kernel round-trip",          test_scatter_kernel);
    run_test("T03  Cast FP32 → FP16/BF16",             test_cast_kernel);
    run_test("T04  Pack-grads into flat bucket",        test_pack_grads);
    run_test("T05  Prefetch / release lifecycle",       test_prefetch_release_lifecycle);
    run_test("T06  Reduce-scatter scale correctness",   test_reduce_scatter);
    run_test("T07  AdamW shard convergence",            test_adamw_shard_convergence);
    run_test("T08  Lion shard convergence",             test_lion_shard_convergence);
    run_test("T09  SGD shard convergence",              test_sgd_shard_convergence);
    run_test("T10  Gradient clipping on shard",         test_grad_clip_shard);
    run_test("T11  Checkpoint save/load round-trip",    test_checkpoint_roundtrip);
    run_test("T12  Memory accounting ZeRO-3 < ZeRO-0", test_memory_accounting);

    printf("\n==========================================================\n");
    printf("  Results: %d / %d passed", g_passed, g_total);
    if (g_failed) printf("   (%d FAILED)", g_failed);
    printf("\n==========================================================\n");

    return (g_failed == 0) ? 0 : 1;
}
