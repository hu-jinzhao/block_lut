/**
 * LUT Decompression CUDA Kernel.
 *
 * Decompresses BlockLUT-quantized weights back to bf16 on GPU.
 * This is a standalone version of the kernel from the LUT-MoE project.
 *
 * BlockLUT format:
 *   - indices: uint8[N] — index into the 256-entry codebook
 *   - absmax: bf16[num_blocks] — block-level scaling factors
 *   - codebook: bf16[256] — the LUT codebook
 *
 * Decompression per element:
 *   bf16_val = codebook[indices[i]] * absmax[block_id(i)]
 *   where block_id(i) = i / block_size
 *
 * Compile:
 *   nvcc -O3 --shared -Xcompiler -fPIC \
 *         -o lut_decompress_kernel.so \
 *         lut_decompress_kernel.cu \
 *         -lcuda
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Block size for shared memory LUT
#define LUT_SIZE 256
#define BLOCK_DIM 256

/**
 * BlockLUT decompression kernel.
 *
 * Loads the 256-entry codebook into shared memory,
 * then each thread decompresses one element.
 *
 * @param output   bf16 output tensor [N]
 * @param indices  uint8 LUT indices [N]
 * @param absmax   bf16 block scaling factors [num_blocks]
 * @param codebook bf16 LUT codebook [256]
 * @param N        total number of elements
 * @param block_size  elements per block (typically 128)
 */
__global__ void blocklut_decompress_kernel(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ indices,
    const __nv_bfloat16* __restrict__ absmax,
    const __nv_bfloat16* __restrict__ codebook,
    const int N,
    const int block_size
) {
    // Load codebook into shared memory
    __shared__ __nv_bfloat16 shared_lut[LUT_SIZE];
    int tid = threadIdx.x;
    if (tid < LUT_SIZE) {
        shared_lut[tid] = codebook[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // LUT lookup
    uint8_t lut_idx = indices[idx];
    __nv_bfloat16 normalized = shared_lut[lut_idx];

    // Block scaling
    int block_id = idx / block_size;
    __nv_bfloat16 scale = absmax[block_id];

    // Decompress: val = normalized * scale
    output[idx] = normalized * scale;
}

/**
 * Batched BlockLUT decompression for multiple experts.
 *
 * @param output   bf16 output [num_experts, N_per_expert]
 * @param indices  uint8 indices [num_experts, N_per_expert]
 * @param absmax   bf16 absmax [num_experts, num_blocks_per_expert]
 * @param codebook bf16 [256] (shared across all experts)
 * @param num_experts
 * @param elements_per_expert
 * @param blocks_per_expert
 * @param block_size
 */
__global__ void batch_blocklut_decompress_kernel(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ indices,
    const __nv_bfloat16* __restrict__ absmax,
    const __nv_bfloat16* __restrict__ codebook,
    const int num_experts,
    const int elements_per_expert,
    const int blocks_per_expert,
    const int block_size
) {
    __shared__ __nv_bfloat16 shared_lut[LUT_SIZE];
    int tid = threadIdx.x;
    if (tid < LUT_SIZE) {
        shared_lut[tid] = codebook[tid];
    }
    __syncthreads();

    int expert_id = blockIdx.x;
    int elem_id = threadIdx.x;

    if (expert_id >= num_experts) return;
    if (elem_id >= elements_per_expert) return;

    int global_idx = expert_id * elements_per_expert + elem_id;

    uint8_t lut_idx = indices[global_idx];
    __nv_bfloat16 normalized = shared_lut[lut_idx];

    int block_id = elem_id / block_size;
    if (block_id >= blocks_per_expert) return;
    __nv_bfloat16 scale = absmax[expert_id * blocks_per_expert + block_id];

    output[global_idx] = normalized * scale;
}

// Launch wrappers for Python/C API
extern "C" {

int launch_blocklut_decompress(
    void* output,
    const void* indices,
    const void* absmax,
    const void* codebook,
    int N,
    int block_size,
    cudaStream_t stream
) {
    int threads = BLOCK_DIM;
    int blocks = (N + threads - 1) / threads;

    blocklut_decompress_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const uint8_t*>(indices),
        static_cast<const __nv_bfloat16*>(absmax),
        static_cast<const __nv_bfloat16*>(codebook),
        N, block_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}

int launch_batch_blocklut_decompress(
    void* output,
    const void* indices,
    const void* absmax,
    const void* codebook,
    int num_experts,
    int elements_per_expert,
    int blocks_per_expert,
    int block_size,
    cudaStream_t stream
) {
    batch_blocklut_decompress_kernel<<<num_experts, elements_per_expert, 0, stream>>>(
        static_cast<__nv_bfloat16*>(output),
        static_cast<const uint8_t*>(indices),
        static_cast<const __nv_bfloat16*>(absmax),
        static_cast<const __nv_bfloat16*>(codebook),
        num_experts, elements_per_expert, blocks_per_expert, block_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}

} // extern "C"
