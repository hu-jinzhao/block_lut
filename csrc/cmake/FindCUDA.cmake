# Custom FindCUDA.cmake override
# Provides CUDA targets that PyTorch's cuda.cmake expects.

set(CUDA_TOOLKIT_ROOT_DIR "/home/hh/miniconda3/envs/lut_moe_cu124" CACHE PATH "CUDA root")
set(CUDA_NVCC_EXECUTABLE "/home/hh/miniconda3/envs/lut_moe_cu124/bin/nvcc" CACHE FILEPATH "nvcc")
set(CUDA_INCLUDE_DIRS "/home/hh/miniconda3/envs/lut_moe_cu124/targets/x86_64-linux/include" CACHE PATH "CUDA include")
set(CUDA_CUDART_LIBRARY "/home/hh/miniconda3/envs/lut_moe_cu124/lib/libcudart.so" CACHE FILEPATH "cudart")
set(CUDA_VERSION_STRING "12.4" CACHE STRING "")
set(CUDA_VERSION "12.4" CACHE STRING "")
set(CUDA_FOUND TRUE CACHE BOOL "")

# Provide the cuda_select_nvcc_arch_flags function
function(cuda_select_nvcc_arch_flags out_var)
    set(${out_var} "" PARENT_SCOPE)
endfunction()

# Helper to create a CUDA:: library target pointing to a real .so
macro(_create_cuda_target name lib_path)
    if(NOT TARGET CUDA::${name})
        add_library(CUDA::${name} UNKNOWN IMPORTED)
        set_target_properties(CUDA::${name} PROPERTIES
            IMPORTED_LOCATION "${lib_path}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
        )
    endif()
endmacro()

# Create all targets PyTorch expects
_create_cuda_target(cuda_driver "${CUDA_CUDART_LIBRARY}")
_create_cuda_target(cudart "${CUDA_CUDART_LIBRARY}")
_create_cuda_target(cudart_static "${CUDA_CUDART_LIBRARY}")
_create_cuda_target(nvToolsExt "${CUDA_CUDART_LIBRARY}")
_create_cuda_target(nvrtc "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so")
_create_cuda_target(cublas "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so")
_create_cuda_target(cublasLt "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so")
_create_cuda_target(curand "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/curand/lib/libcurand.so")
_create_cuda_target(curand_static "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/curand/lib/libcurand_static.a")
_create_cuda_target(cufft "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cufft/lib/libcufft.so")
_create_cuda_target(cufft_static_nocallback "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cufft/lib/libcufft.so")
_create_cuda_target(cusparse "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so")
_create_cuda_target(cusolver "/home/hh/miniconda3/envs/lut_moe_cu124/lib/python3.10/site-packages/nvidia/cusolver/lib/libcusolver.so")

# Set cuda library directory
set(CUDA_LIBRARY_DIR "/home/hh/miniconda3/envs/lut_moe_cu124/lib" CACHE PATH "")
set(CUDA_LIBRARIES "${CUDA_CUDART_LIBRARY}")

message(STATUS "Custom FindCUDA: CUDA 12.4 targets created at ${CUDA_TOOLKIT_ROOT_DIR}")
