// =============================================================================
// HPC CUDA Optimizer Library — ZeRO-2 Extension
// tests/test_zero2.cu  —  Unit + correctness + memory benchmark tests
//
// Build (single GPU, no NCCL):
//   nvcc -std=c++17 -arch=sm_80 -O2 \
//        -I../include test_zero2.cu -o test_zero2
//
// Build (multi-GPU with NCCL):
//   nvcc -std=c++17 -arch=sm_80 -O2 -DHPC_HAVE_NCCL \
//        -I../include test_zero2.cu -lnccl -o test_zero2
//   RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 ./test_zero2
// =============================================================================

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/hpc_zero2.cuh"
#include "../include/hpc_zero2_optimizer.cuh"

using namespace hpc_opt;
using namespace hpc_opt::zero2;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

#define EXPECT_EQ(a, b) \
    do { if ((a)==(b)) { g_pass++; } else { \
         fprintf(stderr,"[FAIL] %s:%d  %lld != %lld\n", \
                 __FILE__,__LINE__,(long long)(a),(long long)(b)); g_fail++; }} while(0)

#define EXPECT_NEAR(v, r, t) \
    do { float _v=(float)(v),_r=(float)(r),_t=(float)(t); \
         if(fabsf(_v-_r)<=_t){g_pass++;}else{ \
         fprintf(stderr,"[FAIL] %s:%d  %.8f != %.8f (±%.2e)\n", \
                 __FILE__,__LINE__,_v,_r,_t);g_fail++;}} while(0)

#define EXPECT_LT(a,b) \
    do { float _a=(float)(a),_b=(float)(b); \
         if(_a<_b){g_pass++;}else{ \
         fprintf(stderr,"[FAIL] %s:%d  %.6f not < %.6f\n", \
                 __FILE__,__LINE__,_a,_b);g_fail++;}} while(0)

#define EXPECT_TRUE(expr) \
    do { if(expr){g_pass++;}else{ \
         fprintf(stderr,"[FAIL] %s:%d  expression false\n", \
                 __FILE__,__LINE__);g_fail++;}} while(0)

#define TEST_HDR(name) printf("\n[TEST] %s\n", name)
#define TEST_OK()      printf("       PASSED\n")

static float* dev_alloc(size_t n, float val = 0.0f, cudaStream_t s = 0) {
    float* p; HPC_CUDA_CHECK(cudaMalloc(&p, n*sizeof(float)));
    std::vector<float> h(n, val);
    HPC_CUDA_CHECK(cudaMemcpyAsync(p, h.data(), n*sizeof(float),
                                   cudaMemcpyHostToDevice, s));
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));
    return p;
}
static void dev_fill(float* p, size_t n, float v, cudaStream_t s=0) {
    std::vector<float> h(n,v);
    HPC_CUDA_CHECK(cudaMemcpyAsync(p,h.data(),n*sizeof(float),
                                   cudaMemcpyHostToDevice,s));
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));
}
static std::vector<float> dev_read(const float* p, size_t n, cudaStream_t s=0) {
    std::vector<float> h(n);
    HPC_CUDA_CHECK(cudaMemcpyAsync(h.data(),p,n*sizeof(float),
                                   cudaMemcpyDeviceToHost,s));
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));
    return h;
}

// ===========================================================================
// TEST 1 – ShardLayout: correct offset computation for 4 tensors, 4 ranks
// ===========================================================================
static void test_shard_layout_offsets() {
    TEST_HDR("ShardLayout – offset and ownership computation");

    // 4 tensors of sizes [10, 20, 14, 20] = 64 total
    // With 4 ranks: shard_size = 16 each
    // rank 0 owns [0,16)   → tensor[0][0..10) + tensor[1][0..6)
    // rank 1 owns [16,32)  → tensor[1][6..20) + tensor[2][0..2)
    // rank 2 owns [32,48)  → tensor[2][2..14) + tensor[3][0..4)
    // rank 3 owns [48,64)  → tensor[3][4..20) (tensor[3] numel=20 padded to +4=20, ok)

    size_t sizes[4] = {10, 20, 14, 20};

    for (int rk = 0; rk < 4; ++rk) {
        ShardLayout layout;
        layout.build(sizes, 4, /*world=*/4, rk);

        EXPECT_EQ(layout.total_numel, 64ULL);
        EXPECT_EQ(layout.shard_size,  16ULL);
        EXPECT_EQ(layout.rank_lo, (size_t)rk * 16);
        EXPECT_EQ(layout.rank_hi, (size_t)(rk + 1) * 16);
    }

    // Verify rank-0 tensor ownership
    {
        ShardLayout layout;
        layout.build(sizes, 4, 4, 0);
        // tensor[0] offset=0, numel=10: rank0=[0,16), intersect=[0,10) → owned
        EXPECT_EQ(layout.shards[0].shard_lo, 0ULL);
        EXPECT_EQ(layout.shards[0].shard_hi, 10ULL);
        // tensor[1] offset=10, numel=20: rank0=[0,16), intersect=[0,6) (global [10,16))
        EXPECT_EQ(layout.shards[1].shard_lo, 0ULL);
        EXPECT_EQ(layout.shards[1].shard_hi, 6ULL);
        // tensor[2] offset=30: rank0=[0,16) → no intersection
        EXPECT_EQ(layout.shards[2].shard_lo, layout.shards[2].shard_hi);
    }

    // Verify rank-1 tensor ownership
    {
        ShardLayout layout;
        layout.build(sizes, 4, 4, 1);
        // rank1=[16,32), tensor[1] global=[10,30) → intersect=[16,30) → local=[6,20)
        EXPECT_EQ(layout.shards[1].shard_lo, 6ULL);
        EXPECT_EQ(layout.shards[1].shard_hi, 20ULL);
        // tensor[2] global=[30,44) → intersect=[30,32) → local=[0,2)
        EXPECT_EQ(layout.shards[2].shard_lo, 0ULL);
        EXPECT_EQ(layout.shards[2].shard_hi, 2ULL);
    }

    TEST_OK();
}

// ===========================================================================
// TEST 2 – ShardLayout: power-of-2 sizes, single rank (world_size=1)
// ===========================================================================
static void test_shard_layout_single_rank() {
    TEST_HDR("ShardLayout – single rank owns everything");

    size_t sizes[3] = {1024*1024, 512*512, 256*256};
    ShardLayout layout;
    layout.build(sizes, 3, /*world=*/1, /*rank=*/0);

    size_t total = 1024*1024 + 512*512 + 256*256;
    EXPECT_EQ(layout.total_numel, total);
    EXPECT_EQ(layout.shard_size,  total);
    EXPECT_EQ(layout.rank_lo, 0ULL);
    EXPECT_EQ(layout.rank_hi, total);

    // Every tensor should be fully owned
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(layout.shards[i].shard_lo, 0ULL);
        EXPECT_EQ(layout.shards[i].shard_hi, sizes[i]);
        EXPECT_TRUE(layout.shards[i].fully_owned);
    }

    TEST_OK();
}

// ===========================================================================
// TEST 3 – Pack / unpack round-trip (FP32 single rank)
// ===========================================================================
static void test_pack_unpack_roundtrip() {
    TEST_HDR("Pack/Unpack round-trip – FP32 single rank");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    // 3 tensors: [4, 8, 4] = 16 elements
    size_t sizes[3] = {4, 8, 4};
    float known[3][8] = {
        {1,2,3,4,0,0,0,0},
        {5,6,7,8,9,10,11,12},
        {13,14,15,16,0,0,0,0}
    };

    float *d0 = dev_alloc(4, 0.0f, s);
    float *d1 = dev_alloc(8, 0.0f, s);
    float *d2 = dev_alloc(4, 0.0f, s);

    // Write known values
    HPC_CUDA_CHECK(cudaMemcpy(d0, known[0], 4*4, cudaMemcpyHostToDevice));
    HPC_CUDA_CHECK(cudaMemcpy(d1, known[1], 8*4, cudaMemcpyHostToDevice));
    HPC_CUDA_CHECK(cudaMemcpy(d2, known[2], 4*4, cudaMemcpyHostToDevice));

    TensorView params[3] = {{d0,4},{d1,8},{d2,4}};

    // Build engine (world_size=1 → no NCCL)
    CommContext comm;
    comm.init_from_env();  // rank=0, world=1 (from env or defaults)

    ZeRO2Engine engine;
    engine.init(params, 3, comm, s, s);

    // Pack grads into flat bucket
    TensorView grads[3] = {{d0,4},{d1,8},{d2,4}};
    engine.pack_grads(grads, 3, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    // Read back flat bucket
    auto flat = dev_read(engine.bucket_.d_flat, engine.layout().total_numel, s);

    // Verify flat matches original
    // Flat = [1,2,3,4, 5,6,7,8,9,10,11,12, 13,14,15,16]
    EXPECT_NEAR(flat[0], 1.0f, 1e-5f);
    EXPECT_NEAR(flat[3], 4.0f, 1e-5f);
    EXPECT_NEAR(flat[4], 5.0f, 1e-5f);
    EXPECT_NEAR(flat[11], 12.0f, 1e-5f);
    EXPECT_NEAR(flat[12], 13.0f, 1e-5f);
    EXPECT_NEAR(flat[15], 16.0f, 1e-5f);

    cudaFree(d0); cudaFree(d1); cudaFree(d2);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 4 – ZeRO2ShardedOptimizer: AdamW shard step analytic check
//   p=1, g=1, lr=0.001, β1=0.9, β2=0.999, eps=1e-8, wd=0 → p_new≈0.999
// ===========================================================================
static void test_sharded_adamw_single_step() {
    TEST_HDR("ZeRO2 Sharded AdamW – analytic single step");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    const size_t N = 4;
    float* d_master = dev_alloc(N, 1.0f, s);   // params = 1.0
    float* d_grad   = dev_alloc(N, 1.0f, s);   // grads  = 1.0

    ZeRO2ShardedOptimizer opt(ShardedOptKind::AdamW);
    opt.lr           = 0.001f;
    opt.beta1        = 0.9f;
    opt.beta2        = 0.999f;
    opt.eps          = 1e-8f;
    opt.weight_decay = 0.0f;
    opt.init(N, d_master, s);

    opt.step(d_grad, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    auto h = dev_read(d_master, N, s);
    // Same analytic result as standard AdamW test
    EXPECT_NEAR(h[0], 0.999f, 1e-4f);
    EXPECT_NEAR(h[N-1], 0.999f, 1e-4f);  // all elements identical

    cudaFree(d_master); cudaFree(d_grad);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 5 – ZeRO2ShardedOptimizer: Lion sign-based update
//   p=2, g=0.5, β1=0.9, β2=0.99, lr=0.01, wd=0
//   interp=0.05>0 → sign=+1 → p_new=2-0.01*(1+0)=1.99
// ===========================================================================
static void test_sharded_lion_sign_update() {
    TEST_HDR("ZeRO2 Sharded Lion – sign-based update");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    const size_t N = 8;  // must be multiple of 4 for vec4 path
    float* d_master = dev_alloc(N, 2.0f, s);
    float* d_grad   = dev_alloc(N, 0.5f, s);

    ZeRO2ShardedOptimizer opt(ShardedOptKind::Lion);
    opt.lr           = 0.01f;
    opt.beta1        = 0.9f;
    opt.beta2        = 0.99f;
    opt.weight_decay = 0.0f;
    opt.init(N, d_master, s);

    opt.step(d_grad, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    auto h = dev_read(d_master, N, s);
    EXPECT_NEAR(h[0], 1.99f, 1e-5f);

    cudaFree(d_master); cudaFree(d_grad);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 6 – ZeRO2ShardedOptimizer: SGD Nesterov shard step
//   p=1, g=1, mom=0.9, lr=0.1, nesterov=true, wd=0
//   v1=1, eff_g=1+0.9*1=1.9 → p=1-0.1*1.9=0.81
// ===========================================================================
static void test_sharded_sgd_nesterov() {
    TEST_HDR("ZeRO2 Sharded SGD – Nesterov momentum");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    const size_t N = 4;
    float* d_master = dev_alloc(N, 1.0f, s);
    float* d_grad   = dev_alloc(N, 1.0f, s);

    ZeRO2ShardedOptimizer opt(ShardedOptKind::SGD);
    opt.lr           = 0.1f;
    opt.momentum     = 0.9f;
    opt.nesterov     = true;
    opt.weight_decay = 0.0f;
    opt.init(N, d_master, s);

    opt.step(d_grad, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    auto h = dev_read(d_master, N, s);
    EXPECT_NEAR(h[0], 0.81f, 1e-4f);

    cudaFree(d_master); cudaFree(d_grad);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 7 – ZeRO2Trainer single-rank convergence: f(θ)=0.5‖θ‖²
// ===========================================================================
static void test_zero2_trainer_convergence() {
    TEST_HDR("ZeRO2Trainer – convergence on f(θ)=0.5‖θ‖² (1000 steps)");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    CommContext comm;
    comm.init_from_env();

    // Two tensors, starting at 5.0
    const size_t N0=512, N1=512;
    float *d_p0 = dev_alloc(N0, 5.0f, s);
    float *d_p1 = dev_alloc(N1, 5.0f, s);
    float *d_g0 = dev_alloc(N0, 0.0f, s);
    float *d_g1 = dev_alloc(N1, 0.0f, s);

    TensorView params[2] = {{d_p0,N0},{d_p1,N1}};
    TensorView grads[2]  = {{d_g0,N0},{d_g1,N1}};

    ZeRO2TrainerConfig cfg;
    cfg.kind         = ShardedOptKind::AdamW;
    cfg.lr           = 1e-2f;
    cfg.weight_decay = 0.0f;
    cfg.max_grad_norm= 0.0f;  // disable clip for clean convergence test

    ZeRO2Trainer trainer;
    trainer.init(params, 2, comm, cfg, s, s);

    // Training loop: g = p (gradient of 0.5‖p‖²)
    for (int step = 0; step < 1000; ++step) {
        // Compute gradient = current param value
        auto h0 = dev_read(d_p0, N0, s);
        auto h1 = dev_read(d_p1, N1, s);
        dev_fill(d_g0, N0, h0[0], s);
        dev_fill(d_g1, N1, h1[0], s);

        trainer.step(params, grads, 2);
        HPC_CUDA_CHECK(cudaStreamSynchronize(s));
    }

    // After AllGather, params should be reconstructed on d_p0/d_p1
    auto h0 = dev_read(d_p0, N0, s);
    printf("       p[0] after 1000 steps: %.6f (target ≈ 0)\n", h0[0]);
    EXPECT_LT(fabsf(h0[0]), 0.1f);

    cudaFree(d_p0); cudaFree(d_p1);
    cudaFree(d_g0); cudaFree(d_g1);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 8 – Pack → scale-shard → unpack consistency (end-to-end no NCCL)
// ===========================================================================
static void test_pack_scale_unpack_consistency() {
    TEST_HDR("Pack → scale shard → unpack consistency (single rank)");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    // 2 tensors of 8 elements each = 16 total
    // Values: tensor0 = [2.0,...], tensor1 = [4.0,...]
    const size_t N = 8;
    float *d_g0 = dev_alloc(N, 2.0f, s);
    float *d_g1 = dev_alloc(N, 4.0f, s);

    TensorView grads[2] = {{d_g0,N},{d_g1,N}};

    CommContext comm;
    comm.init_from_env();

    float *d_p0 = dev_alloc(N, 1.0f, s);
    float *d_p1 = dev_alloc(N, 1.0f, s);
    TensorView params[2] = {{d_p0,N},{d_p1,N}};

    ZeRO2Engine engine;
    engine.init(params, 2, comm, s, s);

    // Pack
    engine.pack_grads(grads, 2, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    // On single rank, reduce_scatter just copies flat→shard
    engine.reduce_scatter_grads(s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    // Read shard: should contain [2,2,...,2, 4,4,...,4] (first 16 elements)
    auto shard = dev_read(engine.grad_shard(), engine.shard_numel(), s);
    EXPECT_NEAR(shard[0], 2.0f, 1e-5f);   // from tensor0
    EXPECT_NEAR(shard[7], 2.0f, 1e-5f);
    EXPECT_NEAR(shard[8], 4.0f, 1e-5f);   // from tensor1
    EXPECT_NEAR(shard[15], 4.0f, 1e-5f);

    cudaFree(d_g0); cudaFree(d_g1);
    cudaFree(d_p0); cudaFree(d_p1);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 9 – Memory savings report verification
// ===========================================================================
static void test_memory_savings_formula() {
    TEST_HDR("ZeRO-2 memory savings formula");

    // For W=8, total Ψ=1e9 elements (FP32 = 4 bytes):
    //   ZeRO-0 per rank: params(4Ψ) + grads(4Ψ) + Adam(8Ψ) = 16Ψ bytes
    //   ZeRO-2 per rank: params(4Ψ) + grads(4Ψ/W) + Adam(8Ψ/W)
    //                  = 4Ψ + 4Ψ/8 + 8Ψ/8 = 4Ψ + 1.5Ψ = 5.5Ψ bytes
    //   Reduction factor ≈ 16/5.5 ≈ 2.9×

    const int    W   = 8;
    const double PSI = 1e9;  // elements
    const double B   = 4.0;  // FP32 bytes per element

    double zero0 = (4 + 4 + 8) * PSI * B;              // 16Ψ bytes
    double zero2 = (4 + 4.0/W + 8.0/W) * PSI * B;      // (4 + 12/W)Ψ bytes

    double reduction = zero0 / zero2;
    printf("       W=%d: ZeRO-0=%.1f GB  ZeRO-2=%.1f GB  reduction=%.2fx\n",
           W, zero0/1e9, zero2/1e9, reduction);

    // At W=8 the reduction should be ~2.9x
    EXPECT_LT(fabsf(reduction - (16.0/(4.0 + 12.0/W))), 0.01);

    // Verify ZeRO-2 < ZeRO-1 (grads also sharded)
    double zero1 = (4 + 4 + 8.0/W) * PSI * B;   // (8 + 8/W)Ψ bytes
    EXPECT_LT(zero2, zero1);

    double z1_reduction = zero0 / zero1;
    double z2_reduction = zero0 / zero2;
    printf("       ZeRO-1: %.2fx reduction  ZeRO-2: %.2fx reduction (%.1f%% better)\n",
           z1_reduction, z2_reduction,
           100.0 * (z2_reduction - z1_reduction) / z1_reduction);

    TEST_OK();
}

// ===========================================================================
// TEST 10 – Checkpoint round-trip for sharded state
// ===========================================================================
static void test_sharded_checkpoint_roundtrip() {
    TEST_HDR("ZeRO2 sharded optimizer – checkpoint round-trip");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    const size_t N = 64;
    float* d_master = dev_alloc(N, 3.0f, s);
    float* d_grad   = dev_alloc(N, 0.1f, s);

    ZeRO2ShardedOptimizer opt(ShardedOptKind::AdamW);
    opt.lr = 1e-3f; opt.weight_decay = 0.0f;
    opt.init(N, d_master, s);

    // Run 5 steps
    for (int i = 0; i < 5; ++i) opt.step(d_grad, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    // Read m buffer before save
    auto m_before = dev_read(opt.m_buf(), N, s);

    // Save checkpoint
    opt.save_shard_state("/tmp/zero2_ckpt", /*rank=*/0);

    // Zero m buffer
    HPC_CUDA_CHECK(cudaMemset(opt.m_buf(), 0, N*sizeof(float)));

    // Load back
    uint32_t restored_step = opt.load_shard_state("/tmp/zero2_ckpt", 0, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    EXPECT_EQ(restored_step, 5U);

    auto m_after = dev_read(opt.m_buf(), N, s);
    EXPECT_NEAR(m_before[0],    m_after[0],    1e-6f);
    EXPECT_NEAR(m_before[N/2],  m_after[N/2],  1e-6f);
    EXPECT_NEAR(m_before[N-1],  m_after[N-1],  1e-6f);

    cudaFree(d_master); cudaFree(d_grad);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 11 – Throughput benchmark: ZeRO2 shard step on 500M params
// ===========================================================================
static void test_throughput_shard_step() {
    TEST_HDR("ZeRO2 shard AdamW throughput – 500M FP32 elements");

    cudaStream_t s; HPC_CUDA_CHECK(cudaStreamCreate(&s));

    const size_t N = 500ULL * 1024 * 1024;
    float *d_master = dev_alloc(N, 1.0f, s);
    float *d_grad   = dev_alloc(N, 0.01f, s);

    ZeRO2ShardedOptimizer opt(ShardedOptKind::AdamW);
    opt.lr = 3e-4f; opt.weight_decay = 0.01f;
    opt.init(N, d_master, s);

    // Warmup
    opt.step(d_grad, s);
    HPC_CUDA_CHECK(cudaStreamSynchronize(s));

    cudaEvent_t t0, t1;
    HPC_CUDA_CHECK(cudaEventCreate(&t0));
    HPC_CUDA_CHECK(cudaEventCreate(&t1));
    HPC_CUDA_CHECK(cudaEventRecord(t0, s));

    const int STEPS = 50;
    for (int i = 0; i < STEPS; ++i) opt.step(d_grad, s);

    HPC_CUDA_CHECK(cudaEventRecord(t1, s));
    HPC_CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    HPC_CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ms_per = ms / STEPS;

    // Effective bandwidth: read params+grads+m+v, write params+m+v → 7× N × 4B
    float bw = (7.0f * N * 4.0f) / (ms_per * 1e-3f) / 1e9f;

    printf("       500M elem × %d steps = %.2f ms/step | %.1f GB/s\n",
           STEPS, ms_per, bw);

    // Should beat 100 ms/step even on older HBM2 GPUs
    EXPECT_LT(ms_per, 400.0f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_master); cudaFree(d_grad);
    HPC_CUDA_CHECK(cudaStreamDestroy(s));
    TEST_OK();
}

// ===========================================================================
// TEST 12 – Alignment: shard size always divisible by world_size
// ===========================================================================
static void test_shard_padding_alignment() {
    TEST_HDR("ShardLayout – padding ensures divisibility for any world_size");

    // Irregular tensor sizes that don't divide evenly
    size_t sizes[5] = {100, 333, 77, 512, 1};  // total = 1023

    for (int W : {1, 2, 3, 4, 5, 7, 8, 16}) {
        for (int rk = 0; rk < W; ++rk) {
            ShardLayout layout;
            layout.build(sizes, 5, W, rk);
            EXPECT_EQ(layout.total_numel % W, 0ULL);
            EXPECT_EQ(layout.shard_size, layout.total_numel / W);
        }
    }

    TEST_OK();
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    printf("============================================================\n");
    printf("  HPC ZeRO-2 Extension – Test Suite\n");
    printf("  Device: %s  SM %d.%d  %.1f GB\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem/(1<<30));
    printf("============================================================\n");

    test_shard_layout_offsets();
    test_shard_layout_single_rank();
    test_pack_unpack_roundtrip();
    test_sharded_adamw_single_step();
    test_sharded_lion_sign_update();
    test_sharded_sgd_nesterov();
    test_zero2_trainer_convergence();
    test_pack_scale_unpack_consistency();
    test_memory_savings_formula();
    test_sharded_checkpoint_roundtrip();
    test_throughput_shard_step();
    test_shard_padding_alignment();

    printf("\n============================================================\n");
    printf("  Results: %d passed | %d failed\n", g_pass, g_fail);
    printf("============================================================\n");
    return (g_fail > 0) ? 1 : 0;
}
