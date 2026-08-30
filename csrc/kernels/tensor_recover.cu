// Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "tensor_recover.hpp"
#include <cuda_bf16.h>


__global__ void cuda_recover_uint16_to_bf16(
    uint16_t* __restrict__ output,  
    const uint8_t* __restrict__ exp_bits, 
    const uint8_t* __restrict__ sign_mantissa, 
    const int n  
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < n) {
        uint64_t sm_vec = *reinterpret_cast<const uint64_t*>(&sign_mantissa[idx]);
        uint64_t exp_vec = *reinterpret_cast<const uint64_t*>(&exp_bits[idx]);
        uint16_t results[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t sm = (sm_vec >> (i * 8)) & 0xFF;
            uint8_t s  = (sm >> 7) & 0x1;
            uint8_t m  = sm & 0x7F;
            uint8_t e  = (exp_vec >> (i * 8)) & 0xFF;

            results[i] = ((uint16_t)s << 15) | ((uint16_t)e << 7) | ((uint16_t)m);
        }
        *reinterpret_cast<uint64_t*>(&output[idx])      = *reinterpret_cast<uint64_t*>(&results[0]);
        *reinterpret_cast<uint64_t*>(&output[idx + 4])  = *reinterpret_cast<uint64_t*>(&results[4]);
    } else {
        for (int i = idx; i < n; ++i) {
            uint8_t sm = sign_mantissa[i];
            uint8_t s = (sm >> 7) & 0x1;
            uint8_t m = sm & 0x7F;
            uint8_t e = exp_bits[i];
            output[i] = ((uint16_t)s << 15) | ((uint16_t)e << 7) | ((uint16_t)m);
        }
    }
}



#define threads_per_block 512
#define elements_per_thread 8

// --- LUT recovery kernel: table lookup ---
__global__ void cuda_lut_recover_to_bf16(
    uint16_t* __restrict__ output,
    const uint8_t* __restrict__ indices,
    const uint16_t* __restrict__ lut,
    const int n
) {
    __shared__ uint16_t lut_smem[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_smem[i] = lut[i];
    }
    __syncthreads();

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < n) {
        uint64_t idx_vec = *reinterpret_cast<const uint64_t*>(&indices[base]);
        uint16_t results[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t lut_idx = (idx_vec >> (i * 8)) & 0xFF;
            results[i] = lut_smem[lut_idx];
        }
        *reinterpret_cast<uint64_t*>(&output[base])     = *reinterpret_cast<uint64_t*>(&results[0]);
        *reinterpret_cast<uint64_t*>(&output[base + 4]) = *reinterpret_cast<uint64_t*>(&results[4]);
    } else {
        for (int i = base; i < n; ++i) {
            output[i] = lut_smem[indices[i]];
        }
    }
}

void lut_moe_launch_lut_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_indices_ptr,
    const uint16_t* gpu_lut_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
){
    int blocks_per_grid = (num_elements + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);
    cuda_lut_recover_to_bf16<<<blocks_per_grid, threads_per_block, 0, tensor_stream>>>(
        gpu_output_tensor, gpu_indices_ptr, gpu_lut_ptr, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (lut_recover): " << cudaGetErrorString(err) << std::endl;
    }
}

// --- BlockLUT recovery kernel: LUT lookup + absmax scaling ---
// Each block of 128 elements shares one absmax (bf16).
// result[i] = LUT[indices[i]] * absmax[i / 128]
__global__ void cuda_blocklut_recover_to_bf16(
    uint16_t* __restrict__ output,
    const uint8_t* __restrict__ indices,
    const uint16_t* __restrict__ absmax,
    const uint16_t* __restrict__ lut,
    const int n
) {
    __shared__ uint16_t lut_smem[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        lut_smem[i] = lut[i];
    }
    __syncthreads();

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < n) {
        uint64_t idx_vec = *reinterpret_cast<const uint64_t*>(&indices[base]);
        __nv_bfloat16 results[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t lut_idx = (idx_vec >> (i * 8)) & 0xFF;
            int elem_idx = base + i;
            int block_id = elem_idx >> 7;  // elem_idx / 128
            __nv_bfloat16 lut_val = *reinterpret_cast<const __nv_bfloat16*>(&lut_smem[lut_idx]);
            __nv_bfloat16 amax_val = *reinterpret_cast<const __nv_bfloat16*>(&absmax[block_id]);
            float a = __bfloat162float(lut_val);
            float b = __bfloat162float(amax_val);
            results[i] = __float2bfloat16(a * b);
        }
        *reinterpret_cast<uint64_t*>(&output[base])     = *reinterpret_cast<uint64_t*>(&results[0]);
        *reinterpret_cast<uint64_t*>(&output[base + 4]) = *reinterpret_cast<uint64_t*>(&results[4]);
    } else {
        for (int i = base; i < n; ++i) {
            uint8_t lut_idx = indices[i];
            int block_id = i >> 7;
            __nv_bfloat16 lut_val = *reinterpret_cast<const __nv_bfloat16*>(&lut_smem[lut_idx]);
            __nv_bfloat16 amax_val = *reinterpret_cast<const __nv_bfloat16*>(&absmax[block_id]);
            float a = __bfloat162float(lut_val);
            float b = __bfloat162float(amax_val);
            *reinterpret_cast<__nv_bfloat16*>(&output[i]) = __float2bfloat16(a * b);
        }
    }
}

void lut_moe_launch_blocklut_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_indices_ptr,
    const uint16_t* gpu_absmax_ptr,
    const uint16_t* gpu_lut_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
){
    int blocks_per_grid = (num_elements + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);
    cuda_blocklut_recover_to_bf16<<<blocks_per_grid, threads_per_block, 0, tensor_stream>>>(
        gpu_output_tensor, gpu_indices_ptr, gpu_absmax_ptr, gpu_lut_ptr, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (blocklut_recover): " << cudaGetErrorString(err) << std::endl;
    }
}

// --- GPU unpack for progressive 4-bit packed format ---
// Two-stage in-place unpack to avoid race conditions.
// Stage 1: copy packed data from [0, N/2) to scratch [N/2, N)
// Stage 2: unpack from scratch [N/2, N) to final [0, N)

__global__ void unpack_4bit_stage1(uint8_t* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2) {
        data[n/2 + i] = data[i];  // copy to scratch (upper half)
    }
}

__global__ void unpack_4bit_stage2(uint8_t* __restrict__ data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n/2) {
        uint8_t p = data[n/2 + i];  // read from scratch
        data[2*i]     = p & 0x0F;
        data[2*i + 1] = (p >> 4) & 0x0F;
    }
}

void lut_moe_launch_unpack_4bit(
    uint8_t* gpu_data,
    size_t num_elements,
    cudaStream_t tensor_stream
){
    int n = (int)num_elements;  // original 8-bit element count (= byte count after unpack)
    int threads = 256;
    int blocks = (n/2 + threads - 1) / threads;
    unpack_4bit_stage1<<<blocks, threads, 0, tensor_stream>>>(gpu_data, n);
    unpack_4bit_stage2<<<blocks, threads, 0, tensor_stream>>>(gpu_data, n);
}

void lut_moe_launch_tensor_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_exp_ptr,
    const uint8_t* gpu_sm_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
){
    int blocks_per_grid = (num_elements + threads_per_block * elements_per_thread - 1) / (threads_per_block * elements_per_thread);
    cuda_recover_uint16_to_bf16<<<blocks_per_grid, threads_per_block, 0, tensor_stream>>>(
        gpu_output_tensor, gpu_exp_ptr, gpu_sm_ptr, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}