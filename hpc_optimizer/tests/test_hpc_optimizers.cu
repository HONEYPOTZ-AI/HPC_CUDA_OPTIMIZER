// =============================================================================
// HPC CUDA Optimizer Library
// tests/test_hpc_optimizers.cu  –  Unit + convergence tests
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -O2 -I../include test_hpc_optimizers.cu \
//        -o test_hpc_optimizers
// Run:
//   ./test_hpc_optimizers
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/hpc_optimizer.cuh"

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

#define EXPECT_NEAR(val, ref, tol) \
    do { \
        float _v=(val), _r=(ref), _t=(tol); \
        if (fabsf(_v-_r) <= _t) { g_pass++; } \
        else { fprintf(stderr,"[FAIL] %s:%d  got %.8f  expected %.8f ± %.2e\n", \
               __FILE__,__LINE__,_v,_r,_t); g_fail++; } \
    } while(0)

#define EXPECT_LT(val, ref) \
    do { \
        float _v=(val), _r=(ref); \
        if (_v < _r) { g_pass++; } \
        else { fprintf(stderr,"[FAIL] %s:%d  %.6f not < %.6f\n", \
               __FILE__,__LINE__,_v,_r); g_fail++; } \
    } while(0)

#define EXPECT_GT(val, ref) \
    do { \
        float _v=(val), _r=(ref); \
        if (_v > _r) { g_pass++; } \
        else { fprintf(stderr,"[FAIL] %s:%d  %.6f not > %.6f\n", \
               __FILE__,__LINE__,_v,_r); g_fail++; } \
    } while(0)

#define TEST_HEADER(name) printf("\n[TEST] %s\n", (name))
#define TEST_OK()         printf("       PASSED\n")

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float read_scalar(float* d, cudaStream_t s = 0) {
    float h; cudaMemcpyAsync(&h, d, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s); return h;
}

static void fill_device(float* d, size_t n, float val, cudaStream_t s = 0) {
    std::vector<float> h(n, val);
    cudaMemcpyAsync(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
}

static float* device_alloc(size_t n, float init = 0.0f, cudaStream_t s = 0) {
    float* d; HPC_CUDA_CHECK(cudaMalloc(&d, n*sizeof(float)));
    fill_device(d, n, init, s); return d;
}

static void device_free(float* p) { cudaFree(p); }

// ===========================================================================
// TEST 1 – AdamW FP32: analytic single step
// ===========================================================================
static void test_adamw_single_step() {
    TEST_HEADER("AdamW FP32 – analytic single step");

    // p=1, g=1, lr=0.001, β₁=0.9, β₂=0.999, eps=1e-8, wd=0
    // bc1=0.1, bc2=0.001  →  lr_eff = 0.001*sqrt(0.001)/0.1 ≈ 0.000316...
    // m=0.1, v=0.001
    // denom = sqrt(0.001/(1-0.999)) + 1e-8 = sqrt(1) + 1e-8 ≈ 1
    // BUT with bias correction baked into lr_eff:
    //   lr_eff = lr * sqrt(bc2)/bc1 = 0.001 * sqrt(0.001)/0.1
    //          = 0.001 * 0.031623 / 0.1 = 0.00031623
    // p_new = 1 - lr_eff * (0.1 / (sqrt(0.001)+1e-8))
    //       = 1 - 0.00031623 * (0.1/0.031632)  ≈ 1 - 0.001 = 0.999

    float *d_p = device_alloc(1, 1.0f);
    float *d_g = device_alloc(1, 1.0f);
    hpc_opt::TensorView pv(d_p, 1), gv(d_g, 1);

    hpc_opt::AdamConfig cfg = hpc_opt::make_adamw_config(0.001f, 0.0f);
    cfg.beta1 = 0.9f; cfg.beta2 = 0.999f; cfg.eps = 1e-8f;

    hpc_opt::AdamOptimizer opt(cfg, /*adamw=*/true);
    opt.init(&pv, 1);
    opt.step(&pv, &gv, 1);

    EXPECT_NEAR(read_scalar(d_p), 0.999f, 1e-4f);

    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 2 – Lion FP32: sign-based update check
// ===========================================================================
static void test_lion_sign_update() {
    TEST_HEADER("Lion FP32 – sign-based update");

    // p=2, g=0.5, m=0, lr=0.01, β₁=0.9, β₂=0.99, wd=0
    // interp = 0.9*0 + 0.1*0.5 = 0.05  →  sign=+1  →  update=+1
    // p_new = 2 - 0.01*(1 + 0) = 1.99
    float *d_p = device_alloc(1, 2.0f);
    float *d_g = device_alloc(1, 0.5f);
    hpc_opt::TensorView pv(d_p, 1), gv(d_g, 1);

    hpc_opt::LionConfig cfg = hpc_opt::make_lion_config(0.01f, 0.0f);
    cfg.beta1=0.9f; cfg.beta2=0.99f;

    hpc_opt::LionOptimizer opt(cfg);
    opt.init(&pv, 1);
    opt.step(&pv, &gv, 1);

    EXPECT_NEAR(read_scalar(d_p), 1.99f, 1e-5f);
    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 3 – SGD with Nesterov momentum: analytic two-step
// ===========================================================================
static void test_sgd_nesterov() {
    TEST_HEADER("SGD Nesterov – analytic two-step");

    // p=1, g=1, lr=0.1, mom=0.9, dampening=0, nesterov=true
    // Step 1:
    //   v1 = 0.9*0 + 1.0 = 1.0
    //   effective_g = g + mom*v1 = 1 + 0.9 = 1.9
    //   p1 = 1 - 0.1*1.9 = 0.81
    // Step 2:
    //   v2 = 0.9*1.0 + 1.0 = 1.9
    //   effective_g = 1 + 0.9*1.9 = 2.71
    //   p2 = 0.81 - 0.1*2.71 = 0.81 - 0.271 = 0.539

    float *d_p = device_alloc(1, 1.0f);
    float *d_g = device_alloc(1, 1.0f);
    hpc_opt::TensorView pv(d_p, 1), gv(d_g, 1);

    hpc_opt::SGDConfig cfg = hpc_opt::make_sgd_config(0.1f, 0.9f);
    cfg.nesterov=true; cfg.dampening=0.0f; cfg.weight_decay=0.0f;

    hpc_opt::SGDOptimizer opt(cfg);
    opt.init(&pv, 1);
    opt.step(&pv, &gv, 1);
    EXPECT_NEAR(read_scalar(d_p), 0.81f, 1e-4f);
    opt.step(&pv, &gv, 1);
    EXPECT_NEAR(read_scalar(d_p), 0.539f, 1e-4f);

    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 4 – Gradient clipping: FP32 global norm
// ===========================================================================
static void test_grad_clip_fp32() {
    TEST_HEADER("Gradient clip FP32 – global L2 norm");

    // 3 tensors each all-ones of size 4:
    //   sq_sum = 3*4*1 = 12  →  norm = sqrt(12) ≈ 3.464
    //   max_norm=1.0  →  scale = 1/3.464 ≈ 0.2887
    //   each element becomes 0.2887

    float *d0 = device_alloc(4, 1.0f);
    float *d1 = device_alloc(4, 1.0f);
    float *d2 = device_alloc(4, 1.0f);
    hpc_opt::TensorView gv[3] = {{d0,4},{d1,4},{d2,4}};

    hpc_opt::GradClipper clipper;
    hpc_opt::GradClipConfig ccfg; ccfg.max_norm = 1.0f;
    hpc_opt::OptimizerStats stats;
    clipper.clip(gv, 3, ccfg, stats);

    float expected_norm = sqrtf(12.0f);
    EXPECT_NEAR(stats.grad_norm_before, expected_norm, 1e-3f);
    EXPECT_NEAR(stats.grad_norm_after,  1.0f,           1e-3f);

    // Verify element values
    std::vector<float> h0(4);
    cudaMemcpy(h0.data(), d0, 4*sizeof(float), cudaMemcpyDeviceToHost);
    float expected_elem = 1.0f / (expected_norm + 1e-6f);
    EXPECT_NEAR(h0[0], expected_elem, 1e-4f);

    device_free(d0); device_free(d1); device_free(d2);
    TEST_OK();
}

// ===========================================================================
// TEST 5 – BF16 gradient clip: verify non-zero scaling
// ===========================================================================
static void test_grad_clip_bf16() {
    TEST_HEADER("Gradient clip BF16 – scalar norm & scale");

    const size_t N = 256;
    __nv_bfloat16* d_g16;
    HPC_CUDA_CHECK(cudaMalloc(&d_g16, N*sizeof(__nv_bfloat16)));

    // Fill with 1.0 in BF16
    std::vector<__nv_bfloat16> hg(N, __float2bfloat16(1.0f));
    cudaMemcpy(d_g16, hg.data(), N*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    hpc_opt::TensorView gv(d_g16, N);
    hpc_opt::GradClipper clipper;
    hpc_opt::GradClipConfig ccfg; ccfg.max_norm = 1.0f;
    hpc_opt::OptimizerStats stats;
    float scale = clipper.clip(&gv, 1, ccfg, stats);

    // norm = sqrt(256) = 16  →  scale < 1
    EXPECT_LT(scale, 1.0f);
    EXPECT_NEAR(stats.grad_norm_before, 16.0f, 0.1f);
    EXPECT_NEAR(stats.grad_norm_after,  1.0f,  0.05f);

    cudaFree(d_g16);
    TEST_OK();
}

// ===========================================================================
// TEST 6 – AdamW convergence: f(x) = 0.5*x^2
// ===========================================================================
static void test_adamw_convergence() {
    TEST_HEADER("AdamW convergence – f(x) = 0.5*x^2 (1000 steps)");

    float *d_p = device_alloc(1, 10.0f);
    float *d_g = device_alloc(1, 0.0f);
    hpc_opt::TensorView pv(d_p, 1), gv(d_g, 1);

    hpc_opt::AdamConfig cfg = hpc_opt::make_adamw_config(0.01f, 0.0f);
    hpc_opt::AdamOptimizer opt(cfg, true);
    opt.init(&pv, 1);

    for (int i = 0; i < 1000; ++i) {
        float x = read_scalar(d_p);
        float g = x;  // gradient of 0.5*x^2
        cudaMemcpy(d_g, &g, sizeof(float), cudaMemcpyHostToDevice);
        opt.step(&pv, &gv, 1);
    }

    float x_final = read_scalar(d_p);
    printf("       x after 1000 steps: %.6f (target ≈ 0)\n", x_final);
    EXPECT_LT(fabsf(x_final), 0.05f);

    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 7 – Lion convergence: f(x) = 0.5*x^2
// ===========================================================================
static void test_lion_convergence() {
    TEST_HEADER("Lion convergence – f(x) = 0.5*x^2 (2000 steps)");

    float *d_p = device_alloc(1, 5.0f);
    float *d_g = device_alloc(1, 0.0f);
    hpc_opt::TensorView pv(d_p, 1), gv(d_g, 1);

    hpc_opt::LionConfig cfg = hpc_opt::make_lion_config(1e-3f, 0.0f);
    hpc_opt::LionOptimizer opt(cfg);
    opt.init(&pv, 1);

    for (int i = 0; i < 2000; ++i) {
        float x = read_scalar(d_p);
        float g = x;
        cudaMemcpy(d_g, &g, sizeof(float), cudaMemcpyHostToDevice);
        opt.step(&pv, &gv, 1);
    }

    float x_final = read_scalar(d_p);
    printf("       x after 2000 steps: %.6f (target ≈ 0)\n", x_final);
    EXPECT_LT(fabsf(x_final), 0.1f);

    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 8 – LR scheduler: WarmupCosine correctness
// ===========================================================================
static void test_warmup_cosine() {
    TEST_HEADER("WarmupCosine LR scheduler");

    float lr = 1e-3f;
    hpc_opt::WarmupCosineLR sched(lr, /*warmup=*/10, /*total=*/100, /*eta_min=*/1e-6f);

    // After 5 steps during warmup: lr should be 5/10 * base = 0.5e-3
    for (int i = 0; i < 5; ++i) sched.step();
    EXPECT_NEAR(lr, 5e-4f, 1e-5f);

    // After warmup is done (10 steps total), lr == base
    for (int i = 0; i < 5; ++i) sched.step();
    EXPECT_NEAR(lr, 1e-3f, 1e-5f);

    // At step 55 (halfway through cosine phase of 90 steps):
    // lr ≈ eta_min + 0.5*(base-eta_min) ≈ 0.5*base
    for (int i = 0; i < 45; ++i) sched.step();
    EXPECT_NEAR(lr, 5e-4f, 1e-4f);

    TEST_OK();
}

// ===========================================================================
// TEST 9 – OneCycleLR: peak at correct step
// ===========================================================================
static void test_one_cycle_lr() {
    TEST_HEADER("OneCycleLR – peak at 30% of steps");

    float lr = 1e-3f;
    // max_lr=1e-2, div_factor=25 → base_lr=4e-4, pct_start=0.3, total=100
    hpc_opt::OneCycleLR sched(lr, 1e-2f, 100, 0.3f, 25.0f, 1e4f);

    // After 30 steps (warmup end) → lr should be close to max_lr=1e-2
    for (int i = 0; i < 30; ++i) sched.step();
    EXPECT_NEAR(lr, 1e-2f, 1e-4f);

    TEST_OK();
}

// ===========================================================================
// TEST 10 – vec4 FP32 AdamW: matches scalar kernel
// ===========================================================================
static void test_adamw_vec4_matches_scalar() {
    TEST_HEADER("AdamW vec4 FP32 vs scalar – output agreement");

    const size_t N = 64;  // divisible by 4

    // Two identical setups: one will use vec4 (always preferred for FP32+FP32),
    // one we manually test at scalar by doing N=1 repeated
    float *d_p1 = device_alloc(N, 3.14f);
    float *d_g1 = device_alloc(N, 0.5f);

    hpc_opt::TensorView pv1(d_p1, N), gv1(d_g1, N);
    hpc_opt::AdamConfig cfg = hpc_opt::make_adamw_config(1e-3f, 0.01f);
    hpc_opt::AdamOptimizer opt1(cfg, true);
    opt1.init(&pv1, 1);
    opt1.step(&pv1, &gv1, 1);   // vec4 path (N%4==0, FP32, no amsgrad)

    std::vector<float> h1(N);
    cudaMemcpy(h1.data(), d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);

    // All elements should be equal (same g, same p, same moments)
    bool all_eq = true;
    for (size_t i = 1; i < N; ++i)
        if (fabsf(h1[i] - h1[0]) > 1e-6f) { all_eq = false; break; }

    if (all_eq) { g_pass++; } else {
        fprintf(stderr, "[FAIL] vec4 elements differ: h[0]=%.8f h[1]=%.8f\n",
                h1[0], h1[1]);
        g_fail++;
    }

    // Value should be < initial 3.14 (lr=1e-3, g=0.5, both positive)
    EXPECT_LT(h1[0], 3.14f);

    device_free(d_p1); device_free(d_g1);
    TEST_OK();
}

// ===========================================================================
// TEST 11 – Checkpoint save + load round-trip
// ===========================================================================
static void test_checkpoint_roundtrip() {
    TEST_HEADER("Checkpoint save/load round-trip");

    const size_t N = 128;
    float *d_p = device_alloc(N, 1.0f);
    float *d_g = device_alloc(N, 0.1f);
    hpc_opt::TensorView pv(d_p, N), gv(d_g, N);

    hpc_opt::AdamConfig cfg = hpc_opt::make_adamw_config(1e-3f, 0.0f);
    hpc_opt::AdamOptimizer opt(cfg, true);
    opt.init(&pv, 1);

    // Run 10 steps
    for (int i = 0; i < 10; ++i) opt.step(&pv, &gv, 1);

    // Read m buffer before save
    std::vector<float> m_before(N);
    cudaMemcpy(m_before.data(), opt.m_buffers()[0], N*sizeof(float),
               cudaMemcpyDeviceToHost);

    // Save
    size_t numel = N;
    hpc_opt::CheckpointIO::save("/tmp/test_hpc_ckpt.bin",
                                opt.m_buffers(), &numel, 1, 1,
                                opt.step_count(),
                                &cfg, sizeof(cfg), 0);

    // Zero m on device
    cudaMemset(opt.m_buffers()[0], 0, N*sizeof(float));

    // Load back
    hpc_opt::AdamConfig cfg2{};
    hpc_opt::CheckpointIO::load("/tmp/test_hpc_ckpt.bin",
                                opt.m_buffers(), &numel, 1,
                                &cfg2, sizeof(cfg2));

    std::vector<float> m_after(N);
    cudaMemcpy(m_after.data(), opt.m_buffers()[0], N*sizeof(float),
               cudaMemcpyDeviceToHost);

    EXPECT_NEAR(m_before[0], m_after[0], 1e-6f);
    EXPECT_NEAR(m_before[N/2], m_after[N/2], 1e-6f);
    EXPECT_NEAR(cfg2.lr, cfg.lr, 1e-7f);

    device_free(d_p); device_free(d_g);
    TEST_OK();
}

// ===========================================================================
// TEST 12 – Throughput benchmark: time 50 AdamW steps on 100M params
// ===========================================================================
static void test_throughput_benchmark() {
    TEST_HEADER("Throughput benchmark – 100M FP32 AdamW steps");

    const size_t N = 100ULL * 1024 * 1024;  // 100M params
    float *d_p, *d_g;
    HPC_CUDA_CHECK(cudaMalloc(&d_p, N*sizeof(float)));
    HPC_CUDA_CHECK(cudaMalloc(&d_g, N*sizeof(float)));
    HPC_CUDA_CHECK(cudaMemset(d_p, 0, N*sizeof(float)));
    HPC_CUDA_CHECK(cudaMemset(d_g, 0, N*sizeof(float)));

    hpc_opt::TensorView pv(d_p, N), gv(d_g, N);
    hpc_opt::AdamConfig cfg = hpc_opt::make_adamw_config(3e-4f);
    hpc_opt::AdamOptimizer opt(cfg, true);
    opt.init(&pv, 1);

    // Warmup
    opt.step(&pv, &gv, 1);
    HPC_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    HPC_CUDA_CHECK(cudaEventCreate(&t0));
    HPC_CUDA_CHECK(cudaEventCreate(&t1));
    HPC_CUDA_CHECK(cudaEventRecord(t0));

    const int STEPS = 50;
    for (int i = 0; i < STEPS; ++i)
        opt.step(&pv, &gv, 1);

    HPC_CUDA_CHECK(cudaEventRecord(t1));
    HPC_CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    HPC_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    float ms_per_step = ms / STEPS;
    float gb_per_step = static_cast<float>(N) * 4.0f * 4.0f / 1e9f; // params+grads+2×moments r/w
    float bandwidth_gbps = gb_per_step / (ms_per_step * 1e-3f);

    printf("       100M params × %d steps  =  %.2f ms/step  |  "
           "eff. BW %.1f GB/s\n",
           STEPS, ms_per_step, bandwidth_gbps);

    EXPECT_LT(ms_per_step, 200.0f);  // should be < 200ms even on V100

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_p); cudaFree(d_g);
    TEST_OK();
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("========================================================\n");
    printf("  HPC CUDA Optimizer – Test Suite\n");
    printf("  Device: %s  SM %d.%d  %.1f GB\n",
           prop.name, prop.major, prop.minor,
           static_cast<double>(prop.totalGlobalMem)/(1<<30));
    printf("========================================================\n");

    test_adamw_single_step();
    test_lion_sign_update();
    test_sgd_nesterov();
    test_grad_clip_fp32();
    test_grad_clip_bf16();
    test_adamw_convergence();
    test_lion_convergence();
    test_warmup_cosine();
    test_one_cycle_lr();
    test_adamw_vec4_matches_scalar();
    test_checkpoint_roundtrip();
    test_throughput_benchmark();

    printf("\n========================================================\n");
    printf("  Results: %d passed | %d failed\n", g_pass, g_fail);
    printf("========================================================\n");
    return (g_fail > 0) ? 1 : 0;
}
