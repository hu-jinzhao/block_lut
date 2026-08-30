#!/usr/bin/env python3
"""
Fused LUT-dequant + GEMV CUDA kernel.
One kernel replaces: gather + arange + clamp + multiply + matmul.

~10x faster than PyTorch decompress+linear for bf16 batch=1.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define LUT_SIZE 256
#define BLOCK_SIZE 128
#define THREADS 256

// Fused LUT-dequant + GEMV kernel.
// Computes: output[row] = sum_col input[col] * codebook[indices[row*K+col]] * absmax[(row*K+col)/BLOCK_SIZE]
__global__ void lut_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,    // [K]
    const uint8_t* __restrict__ indices,         // [M, K]
    const float* __restrict__ codebook_f,        // [LUT_SIZE]
    const __nv_bfloat16* __restrict__ absmax,    // [num_blocks]
    __nv_bfloat16* __restrict__ output,          // [M]
    int M, int K, int num_blocks
) {
    // Load codebook to shared memory
    __shared__ float s_codebook[LUT_SIZE];
    int tid = threadIdx.x;
    if (tid < LUT_SIZE) {
        s_codebook[tid] = codebook_f[tid];
    }
    __syncthreads();

    // Each block handles one row
    // Use 1D grid: grid.x = M
    int row = blockIdx.x;
    if (row >= M) return;

    float accum = 0.0f;
    int row_offset = row * K;

    // Each thread processes a chunk of columns
    for (int col = tid; col < K; col += blockDim.x) {
        uint8_t idx = indices[row_offset + col];
        float norm = s_codebook[idx];

        int blk_id = (row_offset + col) / BLOCK_SIZE;
        if (blk_id >= num_blocks) blk_id = num_blocks - 1;
        float scale = __bfloat162float(absmax[blk_id]);

        float inp = __bfloat162float(input[col]);
        accum += inp * norm * scale;
    }

    // Warp-level reduction (each warp sums to 1 value)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        accum += __shfl_xor_sync(0xFFFFFFFF, accum, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_warp_sum[8];  // 256 threads / 32 warpSize = 8 warps
    if ((tid & 0x1F) == 0) {
        s_warp_sum[tid >> 5] = accum;
    }
    __syncthreads();
    if (tid < 8) {
        float warp_acc = s_warp_sum[tid];
        for (int offset = 4; offset > 0; offset /= 2) {
            warp_acc += __shfl_xor_sync(0xFFFFFFFF, warp_acc, offset);
        }
        if (tid == 0) {
            output[row] = __float2bfloat16(warp_acc);
        }
    }
}
""".strip()

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>

void launch_lut_gemv_kernel(
    const void* input, const void* indices, const void* codebook,
    const void* absmax, void* output, int M, int K, int num_blocks
);

torch::Tensor lut_gemv_forward(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor codebook,
    torch::Tensor absmax
) {
    int M = indices.size(0);
    int K = indices.size(1);
    int num_blocks = absmax.size(0);
    auto output = torch::empty({M}, input.options().dtype(torch::kBFloat16));

    // Ensure codebook is float32
    auto codebook_f = codebook.to(torch::kFloat32).contiguous();

    launch_lut_gemv_kernel(
        input.data_ptr(), indices.data_ptr(), codebook_f.data_ptr(),
        absmax.data_ptr(), output.data_ptr(),
        M, K, num_blocks
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lut_gemv_forward, "Fused LUT-GEMV forward");
}
"""

CUDA_LAUNCH = r"""
void launch_lut_gemv_kernel(
    const void* input, const void* indices, const void* codebook,
    const void* absmax, void* output,
    int M, int K, int num_blocks
) {
    lut_gemv_kernel<<<M, THREADS>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const uint8_t*>(indices),
        static_cast<const float*>(codebook),
        static_cast<const __nv_bfloat16*>(absmax),
        static_cast<__nv_bfloat16*>(output),
        M, K, num_blocks
    );
    cudaDeviceSynchronize();
}
"""


def load_kernel():
    """Load the CUDA kernel (compilation cached by PyTorch)."""
    full_cuda = CUDA_SRC + "\n" + CUDA_LAUNCH
    return load_inline(
        name="lut_gemv",
        cpp_sources=CPP_SRC,
        cuda_sources=full_cuda,
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


# Module-level: compile once
_lut_gemv_mod = None

def lut_gemv(input_vec, indices, codebook, absmax):
    """
    Fused LUT-dequant + GEMV.

    Args:
        input_vec: [K] bf16 on CUDA
        indices: [M, K] uint8 on CUDA
        codebook: [256] bf16
        absmax: [num_blocks] bf16

    Returns:
        [M] bf16 output
    """
    global _lut_gemv_mod
    if _lut_gemv_mod is None:
        _lut_gemv_mod = load_kernel()
    return _lut_gemv_mod.forward(input_vec, indices, codebook, absmax)


if __name__ == '__main__':
    import time

    M, K = 2816, 2048
    inp = torch.randn(K, dtype=torch.bfloat16, device='cuda')
    idx = torch.randint(0, 256, (M, K), dtype=torch.uint8, device='cuda')
    cb = torch.linspace(-3, 3, 256, dtype=torch.bfloat16, device='cuda')
    a = torch.randn((M * K + 127) // 128, dtype=torch.bfloat16, device='cuda').abs()

    # Warmup + compile
    out = lut_gemv(inp, idx, cb, a)
    torch.cuda.synchronize()
    print(f"Compiled OK. Output shape: {out.shape}")

    # Benchmark
    t0 = time.time()
    for _ in range(1000):
        out = lut_gemv(inp, idx, cb, a)
    torch.cuda.synchronize()
    per = (time.time() - t0) / 1000
    print(f"One GEMV: {per*1000:.3f}ms")

    # Full model estimate
    total = per * 4 * 2 * 24  # 4 experts × 2 GEMVs × 24 layers
    print(f"MoE total: {total*1000:.0f}ms → {1/total:.0f} tok/s")

    # Compare with PyTorch
    t0 = time.time()
    for _ in range(200):
        flat = idx.reshape(-1).long()
        n = flat.numel()
        w = (cb.to(inp.device)[flat] *
             a[torch.arange(n, device='cuda') // 128].clamp(max=a.shape[0]-1)
            ).reshape(M, K)
        _ = torch.mv(w, inp)
    torch.cuda.synchronize()
    pt = (time.time() - t0) / 200
    print(f"PyTorch GEMV: {pt*1000:.2f}ms")
    print(f"Speedup: {pt/per:.0f}x")
