#!/bin/bash
# =============================================================================
# slurm_test.sh — SLURM sbatch script for running HPC Optimizer test suite
#
# Usage:
#   sbatch slurm_test.sh [--build-type Release|Debug] [--suite all|<name>]
#
# Submit from repo root:
#   sbatch scripts/slurm_test.sh
#   sbatch scripts/slurm_test.sh --suite test_zero3
#   sbatch --export=BUILD_TYPE=Debug scripts/slurm_test.sh
# =============================================================================

#SBATCH --job-name=hpc-optimizer-tests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1              # Request 1× A100 for single-GPU tests
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_test_%j.out
#SBATCH --error=logs/slurm_test_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${SLURM_NOTIFY_EMAIL:-""}

# ── Configuration (override via --export or environment) ──────────────────────
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-80}"           # 70=V100, 80=A100, 90=H100
TEST_SUITE="${TEST_SUITE:-all}"        # all | test_hpc_optimizers | test_precision | test_zero2 | test_zero3
ENABLE_NCCL="${ENABLE_NCCL:-ON}"
ENABLE_MPI="${ENABLE_MPI:-ON}"
ENABLE_NVTX="${ENABLE_NVTX:-ON}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}"    # Per-test timeout in seconds
BUILD_DIR="build_slurm_${BUILD_TYPE}_sm${CUDA_ARCH}"
LOG_DIR="logs"
RESULTS_DIR="test_results"

# ── Derived ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JUNIT_XML="${RESULTS_DIR}/junit_${TEST_SUITE}_${TIMESTAMP}_job${SLURM_JOB_ID}.xml"

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          HPC Optimizer — SLURM Test Runner                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURMD_NODENAME}"
echo "  Build type   : ${BUILD_TYPE}"
echo "  CUDA arch    : sm_${CUDA_ARCH}"
echo "  Test suite   : ${TEST_SUITE}"
echo "  Build dir    : ${BUILD_DIR}"
echo "  Timestamp    : ${TIMESTAMP}"
echo ""

# ── Module loads (adjust to your cluster's module system) ─────────────────────
echo "[1/5] Loading modules..."
module purge 2>/dev/null || true

# Try common CUDA module names (adjust for your cluster)
for cuda_mod in "cuda/12.2" "cuda/12.0" "cuda/11.8" "cuda"; do
    if module load "${cuda_mod}" 2>/dev/null; then
        echo "  Loaded: ${cuda_mod}"
        break
    fi
done

# NCCL
for nccl_mod in "nccl/2.18" "nccl/2.16" "nccl"; do
    if module load "${nccl_mod}" 2>/dev/null; then
        echo "  Loaded: ${nccl_mod}"
        break
    fi
done

# MPI
for mpi_mod in "openmpi/4.1" "openmpi/4.0" "mpich/4.0"; do
    if module load "${mpi_mod}" 2>/dev/null; then
        echo "  Loaded: ${mpi_mod}"
        break
    fi
done

# CMake
for cmake_mod in "cmake/3.27" "cmake/3.25" "cmake/3.20" "cmake"; do
    if module load "${cmake_mod}" 2>/dev/null; then
        echo "  Loaded: ${cmake_mod}"
        break
    fi
done

echo ""
nvcc --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
echo ""

# ── GPU Environment ───────────────────────────────────────────────────────────
# Use the GPUs allocated by SLURM
if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

# ── Configure ─────────────────────────────────────────────────────────────────
echo "[2/5] Configuring CMake..."
cmake -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCUDA_ARCHS="${CUDA_ARCH}" \
    -DHPC_ENABLE_NCCL="${ENABLE_NCCL}" \
    -DHPC_ENABLE_MPI="${ENABLE_MPI}" \
    -DHPC_ENABLE_NVTX="${ENABLE_NVTX}" \
    -DCMAKE_CUDA_FLAGS="-Xptxas -v" \
    2>&1 | tee "${LOG_DIR}/cmake_config_${TIMESTAMP}.log"

echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
echo "[3/5] Building (${BUILD_TYPE}, sm_${CUDA_ARCH})..."
TIME_BUILD_START=$(date +%s)
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK:-8}" \
    2>&1 | tee "${LOG_DIR}/cmake_build_${TIMESTAMP}.log"
TIME_BUILD_END=$(date +%s)
echo "  Build completed in $((TIME_BUILD_END - TIME_BUILD_START))s"
echo ""

# ── Test ──────────────────────────────────────────────────────────────────────
echo "[4/5] Running tests..."
TIME_TEST_START=$(date +%s)

cd "${BUILD_DIR}"

# Build CTest filter
if [ "${TEST_SUITE}" = "all" ]; then
    CTEST_FILTER=""
    echo "  Running all 48 tests across 4 suites"
else
    CTEST_FILTER="-R ${TEST_SUITE}"
    echo "  Running suite: ${TEST_SUITE}"
fi

# Run CTest with JUnit output
ctest \
    ${CTEST_FILTER} \
    --output-on-failure \
    --timeout "${TEST_TIMEOUT}" \
    --output-junit "../${JUNIT_XML}" \
    --parallel 1 \
    -V 2>&1 | tee "../${LOG_DIR}/ctest_${TIMESTAMP}.log"

CTEST_EXIT=$?
cd "${REPO_ROOT}"

TIME_TEST_END=$(date +%s)
echo ""
echo "  Test run completed in $((TIME_TEST_END - TIME_TEST_START))s"
echo "  JUnit XML: ${JUNIT_XML}"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "[5/5] Summary"
echo "────────────────────────────────────────────────────────────"

if [ -f "${JUNIT_XML}" ]; then
    # Parse JUnit XML for quick summary
    TOTAL=$(grep -o 'tests="[0-9]*"' "${JUNIT_XML}" | grep -o '[0-9]*' | head -1 || echo "?")
    FAILURES=$(grep -o 'failures="[0-9]*"' "${JUNIT_XML}" | grep -o '[0-9]*' | head -1 || echo "?")
    ERRORS=$(grep -o 'errors="[0-9]*"' "${JUNIT_XML}" | grep -o '[0-9]*' | head -1 || echo "?")
    SKIPPED=$(grep -o 'skipped="[0-9]*"' "${JUNIT_XML}" | grep -o '[0-9]*' | head -1 || echo "0")
    echo "  Total    : ${TOTAL}"
    echo "  Failures : ${FAILURES}"
    echo "  Errors   : ${ERRORS}"
    echo "  Skipped  : ${SKIPPED}"
fi

echo ""
if [ ${CTEST_EXIT} -eq 0 ]; then
    echo "  ✓ All tests PASSED"
else
    echo "  ✗ Some tests FAILED (CTest exit: ${CTEST_EXIT})"
fi
echo "────────────────────────────────────────────────────────────"

# ── Multi-GPU extension: submit follow-up job if 1+ GPU available ─────────────
if [ "${RUN_MULTIGPU:-OFF}" = "ON" ]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "${GPU_COUNT}" -ge 2 ]; then
        echo ""
        echo "Submitting multi-GPU follow-up job (${GPU_COUNT} GPUs available)..."
        sbatch \
            --gres=gpu:a100:2 \
            --export=ALL,BUILD_DIR="${BUILD_DIR}",TEST_SUITE="multigpu" \
            scripts/slurm_multigpu_test.sh
    fi
fi

exit ${CTEST_EXIT}
