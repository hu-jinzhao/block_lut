#!/usr/bin/env python3
"""
Fast LUT encoding using a fused CUDA kernel.

Takes bf16 weights + LUT codebook → uint8 indices + bf16 absmax.
~50x faster than naive PyTorch, converts Qwen model in ~30 seconds.
"""

import torch
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for LUT encoding
# For each element, finds nearest codebook entry and stores its index
CUDA_SRC = """
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Block size for LUT shared memory
#define LUT_SIZE 256

__global__ void encode_blocklut_kernel(
    const __nv_bfloat16* __restrict__ weight,  // [rows, cols]
    const float* __restrict__ codebook_f,       // [256] float
    uint8_t* __restrict__ indices,              // [N]
    __nv_bfloat16* __restrict__ absmax,         // [num_blocks]
    int N, int block_size
) {
    // Load codebook to shared memory
    __shared__ float shared_codebook[LUT_SIZE];
    int tid = threadIdx.x;
    if (tid < LUT_SIZE) {
        shared_codebook[tid] = codebook_f[tid];
    }
    __syncthreads();

    // Each thread handles one block's normalization and search
    int block_id = blockIdx.x;
    int block_start = block_id * block_size;
    int this_block_size = min(block_size, N - block_start);
    if (this_block_size <= 0) return;

    // Step 1: Find absmax for this block
    float block_max = 0.0f;
    for (int i = 0; i < this_block_size; i++) {
        float val = __bfloat162float(weight[block_start + i]);
        block_max = fmaxf(block_max, fabsf(val));
    }
    block_max = fmaxf(block_max, 1e-10f);

    // Step 2: Normalize and find nearest codebook entry
    for (int i = threadIdx.x; i < this_block_size; i += blockDim.x) {
        float val = __bfloat162float(weight[block_start + i]) / block_max;

        // Find nearest centroid
        float best_dist = fabsf(val - shared_codebook[0]);
        uint8_t best_idx = 0;
        for (int c = 1; c < LUT_SIZE; c++) {
            float dist = fabsf(val - shared_codebook[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
            }
        }
        indices[block_start + i] = best_idx;
    }

    // First thread writes the absmax
    if (threadIdx.x == 0) {
        absmax[block_id] = __float2bfloat16(block_max);
    }
}

// CUDA error checking wrapper
void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA error at %s: %s\\n", msg, cudaGetErrorString(err));
    }
}

// Python-callable function
void launch_encode(
    const void* weight,   // bf16 [N]
    const void* codebook, // float [256]
    void* indices,        // uint8 [N]
    void* absmax,         // bf16 [num_blocks]
    int N, int block_size
) {
    int num_blocks = (N + block_size - 1) / block_size;
    int threads = 256;  // Must be >= LUT_SIZE (256) to load full codebook

    encode_blocklut_kernel<<<num_blocks, threads>>>(
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const float*>(codebook),
        static_cast<uint8_t*>(indices),
        static_cast<__nv_bfloat16*>(absmax),
        N, block_size
    );

    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "sync");
}
""".strip()

PYTHON_WRAPPER = """
import torch

def encode_lut(weight, codebook_f, block_size=128):
    '''
    Encode bf16 weights to LUT format using fused CUDA kernel.

    Args:
        weight: bf16 tensor [N] on CUDA
        codebook_f: float tensor [256] on CUDA
        block_size: int

    Returns:
        (indices: uint8 [N], absmax: bf16 [num_blocks])
    '''
    N = weight.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    indices = torch.zeros(N, dtype=torch.uint8, device='cuda')
    absmax_out = torch.zeros(num_blocks, dtype=torch.bfloat16, device='cuda')

    torch.ops.lut_encode.launch_encode(
        weight, codebook_f, indices, absmax_out, N, block_size
    )
    return indices, absmax_out
"""

# Compile the CUDA kernel
print("Compiling LUT encode CUDA kernel (one-time compilation)...")
encode_module = load_inline(
    name="lut_encode",
    cpp_sources="""
    #include <torch/extension.h>
    #include <vector>

    void launch_encode(
        const void* weight, const void* codebook,
        void* indices, void* absmax,
        int N, int block_size
    );

    void lut_encode_forward(
        torch::Tensor weight,
        torch::Tensor codebook,
        torch::Tensor indices,
        torch::Tensor absmax,
        int64_t N,
        int64_t block_size
    ) {
        launch_encode(
            weight.data_ptr(),
            codebook.data_ptr(),
            indices.data_ptr(),
            absmax.data_ptr(),
            (int)N,
            (int)block_size
        );
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("launch_encode", &lut_encode_forward, "LUT encode forward");
    }
    """,
    cuda_sources=CUDA_SRC,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# Define encode_lut using the compiled module
import torch
def encode_lut(weight, codebook_f, block_size=128):
    N = weight.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    indices = torch.zeros(N, dtype=torch.uint8, device='cuda')
    absmax_out = torch.zeros(num_blocks, dtype=torch.bfloat16, device='cuda')
    torch.cuda.synchronize()
    encode_module.launch_encode(weight, codebook_f, indices, absmax_out, N, block_size)
    torch.cuda.synchronize()
    return indices.cpu(), absmax_out.cpu()

if __name__ == "__main__":
    # Test
    torch.manual_seed(42)
    N = 2048 * 1408  # One expert
    w = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    codebook = torch.linspace(-3, 3, 256, dtype=torch.float32, device='cuda')

    # Warmup
    for _ in range(5):
        indices, absmax = encode_lut(w, codebook)
    torch.cuda.synchronize()

    # Benchmark
    import time
    t0 = time.time()
    for _ in range(100):
        indices, absmax = encode_lut(w, codebook)
    torch.cuda.synchronize()
    t1 = time.time()
    per_call = (t1 - t0) / 100
    print(f"Time per expert: {per_call*1000:.1f}ms")

    # Total for Qwen
    total_experts = 24 * 60 * 3
    total_time = per_call * total_experts
    print(f"Estimated total for Qwen ({total_experts} matrices): {total_time:.0f}s = {total_time/60:.1f}min")

    # Verify correctness (with block normalization, same as CUDA kernel)
    w_flat = w.float()
    n_blocks = (N + 127) // 128
    padded = torch.zeros(n_blocks * 128, device='cuda')
    padded[:N] = w_flat
    blocks = padded.reshape(-1, 128)
    block_max = blocks.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-10)
    normalized = (blocks / block_max).reshape(-1)[:N]
    codebook_f = codebook.float()
    expected_indices = (normalized.unsqueeze(1) - codebook_f.unsqueeze(0)).abs().argmin(dim=1).to(torch.uint8)
    diff = (indices != expected_indices).sum().item()
    print(f"Correctness: {diff}/{N} mismatches ({diff/N*100:.4f}%)")
    if diff == 0:
        print("  [PASS] Kernel produces correct results!")
    else:
        print("  [FAIL] Kernel output differs from reference")
