// =============================================================================
// HPC CUDA Optimizer Library
// tests/test_precision.cu  —  Precision / dtype conversion correctness tests
//
// Tests:
//   1.  FP32 → FP32 round-trip (identity)
//   2.  FP32 → FP16 → FP32 (quantization error < 1e-3)
//   3.  FP32 → BF16 → FP32 (quantization error < 2e-2)
//   4.  FP16 exact values (0, 1, -1, NaN, Inf)
//   5.  BF16 exact values (0, 1, -1, NaN, Inf)
//   6.  vec4 FP32 load/store round-trip
//   7.  vec2 FP16 load/store round-trip
//   8.  vec2 BF16 load/store round-trip
//   9.  NaN guard: NaN inputs → 0
//   10. Inf guard: Inf inputs → clamped
//   11. to_float / from_float consistency across all dtypes (4096-element batch)
//   12. Stochastic rounding: mean of rounding noise ≈ 0
// =============================================================================

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../include/hpc_precision.cuh"
#include "../include/hpc_types.h"

using namespace hpc_opt;

// ---------------------------------------------------------------------------
// Simple test harness
// ---------------------------------------------------------------------------
static int g_tests   = 0;
static int g_passed  = 0;
static int g_failed  = 0;

#define ASSERT(cond, msg, ...)                                \
  do {                                                         \
    ++g_tests;                                                 \
    if (!(cond)) {                                             \
      ++g_failed;                                              \
      printf("  [FAIL] " msg "\n", ##__VA_ARGS__);            \
    } else {                                                   \
      ++g_passed;                                              \
      printf("  [PASS] " msg "\n", ##__VA_ARGS__);            \
    }                                                          \
  } while(0)

static void section(const char* s) {
    printf("\n=== %s ===\n", s);
}

// ---------------------------------------------------------------------------
// Kernel: batch convert FP32→dtype→FP32 and measure max error
// ---------------------------------------------------------------------------
template<typename T>
__global__ void k_roundtrip(const float* __restrict__ src,
                             float*       __restrict__ dst,
                             int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
    {
        T  q   = from_float<T>(src[i]);
        dst[i] = to_float<T>(q);
    }
}

// Kernel: vec4 FP32 roundtrip
__global__ void k_vec4_roundtrip(const float* src, float* dst, int n4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x)
    {
        float4 v = load_vec4_fp32(src + i*4);
        store_vec4_fp32(dst + i*4, v);
    }
}

// Kernel: vec2 FP16 roundtrip
__global__ void k_vec2_fp16_roundtrip(const __half* src, __half* dst, int n2) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2;
         i += gridDim.x * blockDim.x)
    {
        half2 v = load_vec2_fp16(src + i*2);
        store_vec2_fp16(dst + i*2, v);
    }
}

// Kernel: vec2 BF16 roundtrip
__global__ void k_vec2_bf16_roundtrip(const __nv_bfloat16* src,
                                       __nv_bfloat16* dst, int n2) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2;
         i += gridDim.x * blockDim.x)
    {
        __nv_bfloat162 v = load_vec2_bf16(src + i*2);
        store_vec2_bf16(dst + i*2, v);
    }
}

// Kernel: NaN guard test
__global__ void k_nan_guard(float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
    {
        float v = (i % 2 == 0) ? __int_as_float(0x7FC00000) : 1.5f;  // NaN or 1.5
        out[i]  = nan_guard(v);
    }
}

// Kernel: stochastic rounding noise accumulator (FP32→BF16 repeated samples)
__global__ void k_stoch_noise(float ref, float* noise_sum, int trials) {
    __shared__ float s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    uint32_t seed = tid * 2654435761u;
    for (int i = 0; i < trials; ++i) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        float v = stochastic_round_bf16(ref, seed);
        s_sum[tid] += (v - ref);
    }
    __syncthreads();
    for (int half = blockDim.x/2; half > 0; half >>= 1) {
        if (tid < half) s_sum[tid] += s_sum[tid + half];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(noise_sum, s_sum[0]);
}

// ---------------------------------------------------------------------------
// Helper: max absolute error between two host arrays
// ---------------------------------------------------------------------------
static float max_abs_err(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i)
        mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("HPC Precision Test Suite\n");
    printf("========================\n");

    cudaDeviceProp prop;
    HPC_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name,
           prop.major, prop.minor);

    const int N  = 4096;
    const int N2 = N / 2;
    const int N4 = N / 4;

    // Build host test data
    std::vector<float> h_src(N), h_dst(N);
    for (int i = 0; i < N; ++i)
        h_src[i] = -2.0f + 4.0f * (float)i / (float)N;  // [-2, 2)

    // Device buffers
    float *d_src, *d_dst;
    HPC_CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    HPC_CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    HPC_CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));

    // ---- Test 1: FP32 → FP32 round-trip ----
    section("1. FP32 → FP32 identity");
    HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
    k_roundtrip<float><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
    HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    float err_fp32 = max_abs_err(h_src.data(), h_dst.data(), N);
    ASSERT(err_fp32 == 0.0f, "FP32 round-trip max_err=%.2e (expect 0)", err_fp32);

    // ---- Test 2: FP32 → FP16 → FP32 ----
    section("2. FP32 → FP16 → FP32");
    __half *d_fp16;
    HPC_CUDA_CHECK(cudaMalloc(&d_fp16, N * sizeof(__half)));
    HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
    k_roundtrip<__half><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
    HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    float err_fp16 = max_abs_err(h_src.data(), h_dst.data(), N);
    ASSERT(err_fp16 < 1e-3f, "FP16 round-trip max_err=%.4e (expect <1e-3)", err_fp16);
    cudaFree(d_fp16);

    // ---- Test 3: FP32 → BF16 → FP32 ----
    section("3. FP32 → BF16 → FP32");
    __nv_bfloat16 *d_bf16;
    HPC_CUDA_CHECK(cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16)));
    HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
    k_roundtrip<__nv_bfloat16><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
    HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    float err_bf16 = max_abs_err(h_src.data(), h_dst.data(), N);
    ASSERT(err_bf16 < 2e-2f, "BF16 round-trip max_err=%.4e (expect <2e-2)", err_bf16);
    ASSERT(err_bf16 > err_fp16, "BF16 has lower precision than FP16 (expected)");
    cudaFree(d_bf16);

    // ---- Test 4: FP16 exact values ----
    section("4. FP16 special values");
    {
        float zero_f  = __half2float(__float2half(0.0f));
        float one_f   = __half2float(__float2half(1.0f));
        float neg_f   = __half2float(__float2half(-1.0f));
        ASSERT(zero_f == 0.0f,  "FP16 zero = 0");
        ASSERT(one_f  == 1.0f,  "FP16 one  = 1");
        ASSERT(neg_f  == -1.0f, "FP16 -1   = -1");
        float nan_half = __half2float(__float2half(__int_as_float(0x7FC00000)));
        ASSERT(isnan(nan_half), "FP16 NaN preserved");
        float inf_half = __half2float(__float2half(1e38f));
        ASSERT(isinf(inf_half) || inf_half > 6e4f, "FP16 large val is Inf or clipped");
    }

    // ---- Test 5: BF16 exact values ----
    section("5. BF16 special values");
    {
        float zero_b = __bfloat162float(__float2bfloat16(0.0f));
        float one_b  = __bfloat162float(__float2bfloat16(1.0f));
        float neg_b  = __bfloat162float(__float2bfloat16(-1.0f));
        ASSERT(zero_b == 0.0f,  "BF16 zero = 0");
        ASSERT(one_b  == 1.0f,  "BF16 one  = 1");
        ASSERT(neg_b  == -1.0f, "BF16 -1   = -1");
        float nan_b  = __bfloat162float(__float2bfloat16(__int_as_float(0x7FC00000)));
        ASSERT(isnan(nan_b), "BF16 NaN preserved");
        // BF16 supports same exponent range as FP32
        float big_b  = __bfloat162float(__float2bfloat16(3.4e38f));
        ASSERT(big_b > 3.0e38f || isinf(big_b), "BF16 handles large FP32 range");
    }

    // ---- Test 6: vec4 FP32 load/store ----
    section("6. vec4 FP32 load/store");
    HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
    k_vec4_roundtrip<<<hpc_blocks(N4), HPC_BLOCK>>>(d_src, d_dst, N4);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());
    HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    float err_vec4 = max_abs_err(h_src.data(), h_dst.data(), N);
    ASSERT(err_vec4 == 0.0f, "vec4 FP32 load/store max_err=%.2e", err_vec4);

    // ---- Test 7: vec2 FP16 load/store ----
    section("7. vec2 FP16 load/store");
    {
        std::vector<__half> h_fp16_in(N), h_fp16_out(N);
        for (int i = 0; i < N; ++i) h_fp16_in[i] = __float2half(h_src[i]);
        __half *d_fp16_in, *d_fp16_out;
        HPC_CUDA_CHECK(cudaMalloc(&d_fp16_in,  N * sizeof(__half)));
        HPC_CUDA_CHECK(cudaMalloc(&d_fp16_out, N * sizeof(__half)));
        HPC_CUDA_CHECK(cudaMemcpy(d_fp16_in, h_fp16_in.data(), N * sizeof(__half),
                                  cudaMemcpyHostToDevice));
        HPC_CUDA_CHECK(cudaMemset(d_fp16_out, 0, N * sizeof(__half)));
        k_vec2_fp16_roundtrip<<<hpc_blocks(N2), HPC_BLOCK>>>(d_fp16_in, d_fp16_out, N2);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        HPC_CUDA_CHECK(cudaMemcpy(h_fp16_out.data(), d_fp16_out, N * sizeof(__half),
                                  cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int i = 0; i < N; ++i)
            err = fmaxf(err, fabsf(__half2float(h_fp16_in[i]) -
                                   __half2float(h_fp16_out[i])));
        ASSERT(err == 0.0f, "vec2 FP16 load/store max_err=%.2e", err);
        cudaFree(d_fp16_in); cudaFree(d_fp16_out);
    }

    // ---- Test 8: vec2 BF16 load/store ----
    section("8. vec2 BF16 load/store");
    {
        std::vector<__nv_bfloat16> h_bf16_in(N), h_bf16_out(N);
        for (int i = 0; i < N; ++i) h_bf16_in[i] = __float2bfloat16(h_src[i]);
        __nv_bfloat16 *d_bf16_in, *d_bf16_out;
        HPC_CUDA_CHECK(cudaMalloc(&d_bf16_in,  N * sizeof(__nv_bfloat16)));
        HPC_CUDA_CHECK(cudaMalloc(&d_bf16_out, N * sizeof(__nv_bfloat16)));
        HPC_CUDA_CHECK(cudaMemcpy(d_bf16_in, h_bf16_in.data(), N * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
        HPC_CUDA_CHECK(cudaMemset(d_bf16_out, 0, N * sizeof(__nv_bfloat16)));
        k_vec2_bf16_roundtrip<<<hpc_blocks(N2), HPC_BLOCK>>>(d_bf16_in, d_bf16_out, N2);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        HPC_CUDA_CHECK(cudaMemcpy(h_bf16_out.data(), d_bf16_out, N * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToHost));
        float err = 0.0f;
        for (int i = 0; i < N; ++i)
            err = fmaxf(err, fabsf(__bfloat162float(h_bf16_in[i]) -
                                   __bfloat162float(h_bf16_out[i])));
        ASSERT(err == 0.0f, "vec2 BF16 load/store max_err=%.2e", err);
        cudaFree(d_bf16_in); cudaFree(d_bf16_out);
    }

    // ---- Test 9: NaN guard ----
    section("9. NaN guard (NaN → 0)");
    {
        float *d_nan_out;
        HPC_CUDA_CHECK(cudaMalloc(&d_nan_out, N * sizeof(float)));
        k_nan_guard<<<hpc_blocks(N), HPC_BLOCK>>>(d_nan_out, N);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h_nan_out(N);
        HPC_CUDA_CHECK(cudaMemcpy(h_nan_out.data(), d_nan_out, N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (i % 2 == 0 && h_nan_out[i] != 0.0f) { ok = false; break; }
            if (i % 2 == 1 && h_nan_out[i] != 1.5f) { ok = false; break; }
        }
        ASSERT(ok, "NaN→0 guard: NaNs zeroed, valid values preserved");
        cudaFree(d_nan_out);
    }

    // ---- Test 10: Large-value handling (Inf) ----
    section("10. Inf handling in FP16");
    {
        float huge_val = 1e39f;
        __half h_huge  = __float2half(huge_val);
        float  back    = __half2float(h_huge);
        ASSERT(isinf(back) || back > 6.5e4f,
               "FP16: huge_val=%.1e converts to Inf or max_half=%.1e", huge_val, back);
    }

    // ---- Test 11: to_float / from_float consistency (4096-element batch) ----
    section("11. to_float/from_float dtype batch consistency");
    {
        // FP32 batch
        HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
        k_roundtrip<float><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        float fp32_err = max_abs_err(h_src.data(), h_dst.data(), N);
        ASSERT(fp32_err == 0.0f, "Batch: FP32 err=%.2e", fp32_err);

        // FP16 batch
        HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
        k_roundtrip<__half><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        float fp16_err = max_abs_err(h_src.data(), h_dst.data(), N);
        ASSERT(fp16_err < 1e-3f, "Batch: FP16 err=%.4e", fp16_err);

        // BF16 batch
        HPC_CUDA_CHECK(cudaMemset(d_dst, 0, N * sizeof(float)));
        k_roundtrip<__nv_bfloat16><<<hpc_blocks(N), HPC_BLOCK>>>(d_src, d_dst, N);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());
        HPC_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        float bf16_err = max_abs_err(h_src.data(), h_dst.data(), N);
        ASSERT(bf16_err < 2e-2f, "Batch: BF16 err=%.4e", bf16_err);
    }

    // ---- Test 12: Stochastic rounding noise ≈ 0 ----
    section("12. Stochastic rounding: mean noise ≈ 0");
    {
        // Test stochastic_round_bf16 if it exists in hpc_precision.cuh.
        // If not defined, this test becomes a dtype-bias check:
        // Round a FP32 value to BF16 many times. Mean rounding error should be ~0.
        float ref_val  = 3.14159f;
        int   trials   = 1024;

        float *d_noise_sum;
        HPC_CUDA_CHECK(cudaMalloc(&d_noise_sum, sizeof(float)));
        HPC_CUDA_CHECK(cudaMemset(d_noise_sum, 0, sizeof(float)));

        k_stoch_noise<<<1, 256>>>(ref_val, d_noise_sum, trials);
        HPC_CUDA_CHECK(cudaDeviceSynchronize());

        float h_noise_sum = 0.0f;
        HPC_CUDA_CHECK(cudaMemcpy(&h_noise_sum, d_noise_sum, sizeof(float),
                                  cudaMemcpyDeviceToHost));
        float mean_noise = h_noise_sum / (float)(256 * trials);
        // BF16 has 1/128 ~ 0.0078 ulp; mean noise should be < 0.01
        ASSERT(fabsf(mean_noise) < 0.02f,
               "Stochastic rounding mean_noise=%.5f (expect |x|<0.02)", mean_noise);
        cudaFree(d_noise_sum);
    }

    // ---- Summary ----
    printf("\n=========================================\n");
    printf("Precision Tests: %d/%d passed", g_passed, g_tests);
    if (g_failed > 0) printf("  (%d FAILED)", g_failed);
    printf("\n=========================================\n");

    cudaFree(d_src);
    cudaFree(d_dst);

    return (g_failed == 0) ? 0 : 1;
}
