# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.


cmake_minimum_required(VERSION 3.18)
project(LUT_MoE LANGUAGES CXX CUDA)
# Custom cmake modules (FindCUDA override)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Set project root directory
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Prevent fatbinData error
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# RTX A5000 (Ampere 8.6) + Turing 7.5
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --default-stream per-thread")
# This determines the behavior in GDB debugging
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
# Python & Pybind11 & Torch
set(CMAKE_FIND_QUIETLY ON)
set(Python_EXECUTABLE "/home/hh/miniconda3/envs/lut_moe_cu124/bin/python")
find_package(Python 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)
# Set Pybind11 directory
set(pybind11_DIR "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Tell PyTorch cmake to skip CUDA detection (we handle it ourselves)
set(CAFFE2_USE_CUDA OFF CACHE BOOL "")
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "")
# Pre-set arch flags to avoid calling cuda_select_nvcc_arch_flags
set(CUDA_NVCC_FLAGS "" CACHE STRING "")

# Set Torch directory
set(Torch_DIR "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_library(ZSTD_LIBRARY zstd REQUIRED)
find_library(LZ4_LIBRARY lz4 REQUIRED)
file(GLOB BASE_SRCS base/*.cpp base/*.hpp)
file(GLOB BIND_SRCS binding/*.cpp binding/*.hpp)
file(GLOB COMMON_SRCS common/*.cpp common/*.hpp)
file(GLOB DISPATCH_SRCS dispatcher/*.cpp dispatcher/*.hpp)
file(GLOB LOG_SRCS logs/*.cpp logs/*.hpp)
file(GLOB MEMORY_SRCS memory/*.cpp memory/*.hpp)
file(GLOB PREFETCH_SRCS prefetch/*.cpp prefetch/*.hpp)
file(GLOB TENSOR_SRCS tensor_engine/*.cpp tensor_engine/*.hpp)
file(GLOB TOPO_SRCS topology/*.cpp topology/*.hpp)
file(GLOB UTIL_SRCS utils/*.cpp utils/*.hpp)
# CUDA
file(GLOB CUDA_SRCS kernels/*.cu)
# Build CUDA Kernels object
add_library(cuda_kernels OBJECT ${CUDA_SRCS})
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(cuda_kernels PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/home/hh/miniconda3/envs/lut_moe_cu124/include"
    "/home/hh/miniconda3/envs/lut_moe_cu124/include/python3.10"
    "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/nvtx/include"
)
# Build Python Module
pybind11_add_module(
    LUT_MoE
    binding/bindings.cpp
    ${BASE_SRCS}
    ${COMMON_SRCS}
    ${DISPATCH_SRCS}
    ${LOG_SRCS}
    ${MEMORY_SRCS}
    ${PREFETCH_SRCS}
    ${TENSOR_SRCS}
    ${TOPO_SRCS}
    ${UTIL_SRCS}
    $<TARGET_OBJECTS:cuda_kernels>
)
target_include_directories(LUT_MoE PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    "/home/hh/miniconda3/envs/lut_moe_cu124/include"
    "/home/hh/miniconda3/envs/lut_moe_cu124/include/python3.10"
    "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/nvtx/include"
)
set_target_properties(LUT_MoE PROPERTIES
    INSTALL_RPATH "$ORIGIN:/home/hh/miniconda3/envs/lut_moe_cu124/lib"
)
target_link_libraries(LUT_MoE PRIVATE
    ${TORCH_LIBRARIES}
    "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/torch/lib/libtorch_python.so"
    ${ZSTD_LIBRARY}
    ${LZ4_LIBRARY}
)
# Install to project root
install(TARGETS LUT_MoE DESTINATION ".")
