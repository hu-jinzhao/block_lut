#!/usr/bin/env python3
"""
Build script for LUT decompression CUDA kernel.

Compiles the standalone CUDA kernel for block LUT decompression
and provides a Python interface via ctypes.

Usage:
    python3 build_lut_kernel.py         # Build the kernel
    python3 build_lut_kernel.py --test  # Build and run test
"""

import argparse
import ctypes
import os
import subprocess
import sys

import torch

KERNEL_SRC = os.path.join(os.path.dirname(__file__), "lut_decompress_kernel.cu")
KERNEL_SO = os.path.join(os.path.dirname(__file__), "lut_decompress_kernel.so")


def build_kernel():
    """Compile the CUDA kernel to a shared library."""
    # Find nvcc - prefer CUDA 12.x for driver compatibility
    nvcc = os.environ.get("CUDA_NVCC", None)
    if not nvcc or not os.path.exists(nvcc):
        # Prefer CUDA 12.x nvcc (better compatibility with older drivers)
        for path in [
            "/home/hh/miniconda3/envs/lut_moe_cu124/bin/nvcc",
            "/home/hh/miniconda3/envs/lut_moe/bin/nvcc",
            "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc",
            "/usr/local/cuda/bin/nvcc",
            "nvcc",
        ]:
            if os.path.exists(path) or path == "nvcc":
                nvcc = path
                break

    if not os.path.exists(nvcc) and nvcc != "nvcc":
        print(f"[LUT-MoE] nvcc not found. Install CUDA toolkit or set CUDA_NVCC.")
        return False

    # Get CUDA compute capability
    cc = "86"  # Default for RTX A5000 (Ampere)
    if torch.cuda.is_available():
        cc = f"{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"

    # Build command - link with default cudart for driver compatibility
    cmd = [
        nvcc,
        "-O3",
        "--shared",
        "-Xcompiler", "-fPIC",
        f"-arch=sm_{cc}",
        "-o", KERNEL_SO,
        KERNEL_SRC,
    ]

    print(f"[LUT-MoE] Building CUDA kernel...")
    print(f"  nvcc: {nvcc}")
    print(f"  arch: sm_{cc}")
    print(f"  output: {KERNEL_SO}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[LUT-MoE] Build failed:")
        print(result.stderr)
        return False

    print(f"[LUT-MoE] Build successful! -> {KERNEL_SO}")
    return True


class LutDecompressor:
    """Python wrapper for the CUDA LUT decompression kernel."""

    def __init__(self):
        if not os.path.exists(KERNEL_SO):
            raise RuntimeError(
                f"Kernel not built. Run: python3 {__file__}"
            )
        self.lib = ctypes.cdll.LoadLibrary(KERNEL_SO)

    def decompress(
        self,
        indices: torch.Tensor,
        absmax: torch.Tensor,
        codebook: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        """
        Decompress LUT indices to bf16 on GPU.

        Args:
            indices: uint8 tensor [N] on CUDA
            absmax: bf16 tensor [num_blocks] on CUDA
            codebook: bf16 tensor [256] on CUDA
            block_size: elements per block

        Returns:
            bf16 tensor [N] on CUDA
        """
        assert indices.is_cuda and indices.dtype == torch.uint8
        assert absmax.is_cuda and absmax.dtype == torch.bfloat16
        assert codebook.is_cuda and codebook.dtype == torch.bfloat16
        assert codebook.shape[0] == 256

        N = indices.shape[0]
        output = torch.zeros(N, dtype=torch.bfloat16, device="cuda")

        func = self.lib.launch_blocklut_decompress
        func.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,  # cudaStream_t (can be None for default stream)
        ]
        func.restype = ctypes.c_int

        # Use default stream (None) to avoid ctypes issues with cuda_stream
        result = func(
            output.data_ptr(),
            indices.data_ptr(),
            absmax.data_ptr(),
            codebook.data_ptr(),
            N, block_size,
            None,  # default stream
        )

        if result != 0:
            # Get more diagnostics
            torch.cuda.synchronize()
            raise RuntimeError(
                f"CUDA kernel launch failed (code={result})")

        return output


def test_kernel():
    """Test the CUDA kernel."""
    print("\n=== Testing LUT Decompression CUDA Kernel ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    try:
        decompressor = LutDecompressor()
    except RuntimeError as e:
        print(f"Cannot load kernel: {e}")
        return

    # Create test data
    torch.manual_seed(42)
    N = 4096
    block_size = 128
    num_blocks = (N + block_size - 1) // block_size

    # Original bf16 data
    orig = torch.randn(N, dtype=torch.bfloat16, device="cuda")

    # Manually compute BlockLUT
    blocks = orig.reshape(-1, block_size)
    absmax, _ = blocks.abs().max(dim=1, keepdim=True)
    absmax = absmax.clamp(min=1e-10)
    normalized = blocks / absmax

    # Simulate 256-entry codebook
    codebook = torch.linspace(-3, 3, 256, dtype=torch.bfloat16, device="cuda")

    # Find nearest indices
    norm_f = normalized.reshape(-1).float()
    cb_f = codebook.float()
    dists = torch.abs(norm_f.unsqueeze(1) - cb_f.unsqueeze(0))
    indices = dists.argmin(dim=1).to(torch.uint8)

    # Decompress with CUDA kernel
    output = decompressor.decompress(
        indices, absmax.reshape(-1), codebook, block_size
    )

    # Decompress with PyTorch (reference)
    expected = codebook[indices.long()] * absmax.reshape(-1)[
        torch.arange(N, device="cuda") // block_size
    ]

    # Compare
    diff = (output.float() - expected.float()).abs().max().item()
    mse = ((output.float() - expected.float()) ** 2).mean().item()

    print(f"  Max diff vs reference: {diff:.6f}")
    print(f"  MSE vs reference: {mse:.10f}")
    print(f"  Output shape: {output.shape}")

    if diff < 0.01:
        print("  [PASS] Kernel produces correct results")
    else:
        print("  [FAIL] Kernel output differs from reference")

    # Speed benchmark
    N_large = 100 * 4096  # ~3.2M elements
    indices_large = torch.randint(0, 256, (N_large,), dtype=torch.uint8, device="cuda")
    absmax_large = torch.randn(N_large // 128 + 1, dtype=torch.bfloat16, device="cuda").abs()

    # Warmup
    for _ in range(10):
        decompressor.decompress(indices_large, absmax_large, codebook, 128)

    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(100):
        decompressor.decompress(indices_large, absmax_large, codebook, 128)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    elements_per_sec = N_large * 100 / elapsed
    print(f"\n  Speed: {elapsed/100*1000:.3f} ms per call "
          f"({elements_per_sec/1e9:.2f} G elements/s)")

    # Compare with PyTorch speed
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = codebook[indices_large.long()] * absmax_large[indices_large.long() // 128]
    torch.cuda.synchronize()
    elapsed_pt = time.time() - start
    print(f"  PyTorch reference: {elapsed_pt/100*1000:.3f} ms per call "
          f"({N_large * 100 / elapsed_pt / 1e9:.2f} G elements/s)")
    print(f"  Speedup: {elapsed_pt/elapsed:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Build and test")
    args = parser.parse_args()

    success = build_kernel()

    if success and args.test:
        sys.path.insert(0, os.path.dirname(__file__))
        test_kernel()
