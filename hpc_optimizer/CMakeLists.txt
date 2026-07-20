# =============================================================================
# HPC CUDA Optimizer Library
# CMakeLists.txt  –  HPC build system
#
# Targets:  V100 (sm_70) | A100 (sm_80) | H100 (sm_90) | RTX 4090 (sm_89)
# CUDA 12.x | C++17 | Optional: MPI, NCCL, NVTX
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#            -DCUDA_ARCHS="70;80;90"         # multi-arch fat binary
#            -DHPC_ENABLE_NCCL=ON            # link NCCL for multi-GPU
#            -DHPC_ENABLE_MPI=ON             # MPI bootstrap
#            -DHPC_ENABLE_NVTX=ON            # Nsight profiling markers
#   make -j$(nproc)
#   ctest --output-on-failure
# =============================================================================
cmake_minimum_required(VERSION 3.22)
project(HPC_Optimizer LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# GPU architecture selection
#   Default: sm_70 + sm_80 + sm_90 fat binary (covers V100→H100)
#   Override: -DCUDA_ARCHS="80;90"
# ---------------------------------------------------------------------------
if(NOT DEFINED CUDA_ARCHS)
  set(CUDA_ARCHS "70;80;90")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

message(STATUS "[HPC_Optimizer] GPU architectures: ${CUDA_ARCHS}")

# ---------------------------------------------------------------------------
# Optional features
# ---------------------------------------------------------------------------
option(HPC_ENABLE_NCCL "Enable NCCL multi-GPU communication" OFF)
option(HPC_ENABLE_MPI  "Enable MPI for multi-node bootstrap"  OFF)
option(HPC_ENABLE_NVTX "Enable NVTX profiling markers"        OFF)

# ---------------------------------------------------------------------------
# Global CUDA compilation flags
# ---------------------------------------------------------------------------
set(HPC_CUDA_FLAGS
    --use_fast_math                 # reciprocal sqrt, fma, etc.
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    --expt-relaxed-constexpr        # constexpr in device lambdas
    --extended-lambda               # device lambdas
    --generate-line-info            # source correlation in Nsight
    -Xcompiler=-fPIC
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND HPC_CUDA_FLAGS -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND HPC_CUDA_FLAGS -G -g -O0)
else()
  list(APPEND HPC_CUDA_FLAGS -O2 -g --generate-line-info)
endif()

# TF32: on by default for Ampere+; disable only when maximum reproducibility needed
# (cublasMath flag; nvcc itself doesn't need explicit control)

# ---------------------------------------------------------------------------
# Interface library  (header-only: all .cuh live in include/)
# ---------------------------------------------------------------------------
add_library(hpc_optimizer INTERFACE)
target_include_directories(hpc_optimizer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(hpc_optimizer INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:${HPC_CUDA_FLAGS}>
)

# ---------------------------------------------------------------------------
# NCCL
# ---------------------------------------------------------------------------
if(HPC_ENABLE_NCCL)
  find_library(NCCL_LIBRARY nccl
      HINTS ENV NCCL_HOME /usr/local/nccl ENV CUDA_HOME /usr/local/cuda
      PATH_SUFFIXES lib lib64)
  find_path(NCCL_INCLUDE_DIR nccl.h
      HINTS ENV NCCL_HOME /usr/local/nccl /usr/local/cuda
      PATH_SUFFIXES include)

  if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "[HPC_Optimizer] NCCL found: ${NCCL_LIBRARY}")
    target_compile_definitions(hpc_optimizer INTERFACE HPC_HAVE_NCCL)
    target_include_directories(hpc_optimizer INTERFACE ${NCCL_INCLUDE_DIR})
    target_link_libraries(hpc_optimizer INTERFACE ${NCCL_LIBRARY})
  else()
    message(WARNING "[HPC_Optimizer] NCCL requested but not found — disabling")
    set(HPC_ENABLE_NCCL OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
if(HPC_ENABLE_MPI)
  find_package(MPI REQUIRED CXX)
  if(MPI_FOUND)
    message(STATUS "[HPC_Optimizer] MPI found: ${MPI_CXX_LIBRARIES}")
    target_compile_definitions(hpc_optimizer INTERFACE HPC_HAVE_MPI)
    target_include_directories(hpc_optimizer INTERFACE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(hpc_optimizer INTERFACE ${MPI_CXX_LIBRARIES})
  endif()
endif()

# ---------------------------------------------------------------------------
# NVTX
# ---------------------------------------------------------------------------
if(HPC_ENABLE_NVTX)
  find_library(NVTX_LIBRARY nvToolsExt
      HINTS ENV CUDA_HOME /usr/local/cuda
      PATH_SUFFIXES lib lib64)
  if(NVTX_LIBRARY)
    message(STATUS "[HPC_Optimizer] NVTX found: ${NVTX_LIBRARY}")
    target_compile_definitions(hpc_optimizer INTERFACE HPC_HAVE_NVTX)
    target_link_libraries(hpc_optimizer INTERFACE ${NVTX_LIBRARY})
  else()
    message(WARNING "[HPC_Optimizer] NVTX not found — profiling markers disabled")
    set(HPC_ENABLE_NVTX OFF)
  endif()
endif()

# cuBLAS — required for Tensor Parallel GEMM
find_library(CUBLAS_LIB cublas
    HINTS ENV CUDA_HOME /usr/local/cuda
    PATH_SUFFIXES lib lib64)
find_library(CUBLASLT_LIB cublasLt
    HINTS ENV CUDA_HOME /usr/local/cuda
    PATH_SUFFIXES lib lib64)
if(CUBLAS_LIB)
  message(STATUS "[HPC_Optimizer] cuBLAS found: ${CUBLAS_LIB}")
  target_compile_definitions(hpc_optimizer INTERFACE HPC_HAVE_CUBLAS)
  target_link_libraries(hpc_optimizer INTERFACE ${CUBLAS_LIB})
else()
  message(WARNING "[HPC_Optimizer] cuBLAS not found — Tensor Parallel GEMM will abort at runtime")
endif()
if(CUBLASLT_LIB)
  target_link_libraries(hpc_optimizer INTERFACE ${CUBLASLT_LIB})
endif()

# ---------------------------------------------------------------------------
# Helper macro: add an HPC executable
# ---------------------------------------------------------------------------
macro(hpc_add_executable name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE hpc_optimizer)
  set_target_properties(${name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      RUNTIME_OUTPUT_DIRECTORY   ${CMAKE_BINARY_DIR}/bin
  )
endmacro()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
hpc_add_executable(train_multigpu   examples/train_multigpu.cu)
hpc_add_executable(train_single     examples/train_single.cu)

# ZeRO-2 multi-GPU training demo (requires NCCL + MPI)
hpc_add_executable(train_zero2      examples/train_zero2.cu)
if(NOT HPC_ENABLE_NCCL)
  message(WARNING
    "[HPC_Optimizer] train_zero2 built without NCCL — "
    "will run in single-rank stub mode only.")
endif()

# ZeRO-3 + Tensor Parallel combined demo
hpc_add_executable(train_zero3_tp   examples/train_zero3_tp.cu)
if(HPC_ENABLE_NCCL AND CUBLAS_LIB)
  message(STATUS
    "[HPC_Optimizer] train_zero3_tp: ZeRO-3 + TP fully enabled.")
elseif(NOT CUBLAS_LIB)
  message(WARNING
    "[HPC_Optimizer] train_zero3_tp: cuBLAS not found — "
    "TP GEMM will abort at runtime. Link with -lcublas.")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
hpc_add_executable(test_hpc_optimizers tests/test_hpc_optimizers.cu)
hpc_add_executable(test_precision       tests/test_precision.cu)
hpc_add_executable(test_zero2           tests/test_zero2.cu)
hpc_add_executable(test_zero3           tests/test_zero3.cu)

add_test(NAME HPCOptimizerTests
         COMMAND ${CMAKE_BINARY_DIR}/bin/test_hpc_optimizers)
add_test(NAME PrecisionTests
         COMMAND ${CMAKE_BINARY_DIR}/bin/test_precision)
add_test(NAME ZeRO2Tests
         COMMAND ${CMAKE_BINARY_DIR}/bin/test_zero2)
add_test(NAME ZeRO3Tests
         COMMAND ${CMAKE_BINARY_DIR}/bin/test_zero3)

# Multi-rank ZeRO-2 integration test (only when MPI is available)
if(HPC_ENABLE_MPI AND HPC_ENABLE_NCCL)
  find_program(MPIEXEC_BIN mpirun mpiexec)
  if(MPIEXEC_BIN)
    add_test(NAME ZeRO2MultiGPU
             COMMAND ${MPIEXEC_BIN} -np 2
                     ${CMAKE_BINARY_DIR}/bin/test_zero2)
    set_tests_properties(ZeRO2MultiGPU PROPERTIES
        ENVIRONMENT "NCCL_DEBUG=WARN")
    add_test(NAME ZeRO3MultiGPU
             COMMAND ${MPIEXEC_BIN} -np 2
                     ${CMAKE_BINARY_DIR}/bin/test_zero3)
    set_tests_properties(ZeRO3MultiGPU PROPERTIES
        ENVIRONMENT "NCCL_DEBUG=WARN")
    add_test(NAME ZeRO3TPIntegration
             COMMAND ${MPIEXEC_BIN} -np 4
                     ${CMAKE_BINARY_DIR}/bin/train_zero3_tp
                     --layers 2 --steps 5 --tp 2)
    set_tests_properties(ZeRO3TPIntegration PROPERTIES
        ENVIRONMENT "NCCL_DEBUG=WARN"
        TIMEOUT 120)
  endif()
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(DIRECTORY include/
    DESTINATION include/hpc_optimizer
    FILES_MATCHING PATTERN "*.cuh" PATTERN "*.h")

install(TARGETS hpc_optimizer EXPORT HPCOptimizerTargets)
install(EXPORT HPCOptimizerTargets
    FILE HPCOptimizerTargets.cmake
    NAMESPACE hpc_opt::
    DESTINATION lib/cmake/HPCOptimizer)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== HPC Optimizer Build Configuration ===")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA archs    : ${CUDA_ARCHS}")
message(STATUS "  NCCL          : ${HPC_ENABLE_NCCL}")
message(STATUS "  MPI           : ${HPC_ENABLE_MPI}")
message(STATUS "  NVTX          : ${HPC_ENABLE_NVTX}")
message(STATUS "  ZeRO-2        : enabled (hpc_zero2.cuh + hpc_zero2_optimizer.cuh)")
message(STATUS "  ZeRO-3        : enabled (hpc_zero3.cuh)")
message(STATUS "  Tensor Par.   : enabled (hpc_tensor_parallel.cuh)")
message(STATUS "  cuBLAS        : ${CUBLAS_LIB}")
message(STATUS "==========================================")
message(STATUS "")
