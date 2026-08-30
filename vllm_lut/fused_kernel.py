#!/usr/bin/env python3
"""
Fused LUT-dequant + GEMV kernel using torch.compile.

Replaces the decompress → F.linear two-step with a single fused operation.
torch.compile fuses gather + arange + clamp + multiply + matmul into one kernel.
"""

import torch
import torch.nn.functional as F


@torch.compile(dynamic=True)
def fused_lut_gemv(input_vec: torch.Tensor,
                   indices: torch.Tensor,
                   codebook: torch.Tensor,
                   absmax: torch.Tensor,
                   block_size: int = 128) -> torch.Tensor:
    """
    Fused LUT-dequant + matrix-vector product.

    Computes: output = (codebook[indices] * absmax[block_id]) @ input_vec

    All tensors must be on CUDA. The torch.compile graph captures the
    gather + broadcast + multiply + dot product as a single fused kernel.

    Args:
        input_vec: [K] bf16 input vector
        indices: [M, K] uint8 LUT indices
        codebook: [256] bf16 codebook
        absmax: [num_blocks_total] bf16 scaling factors
        block_size: elements per block (default 128)

    Returns:
        [M] bf16 output vector
    """
    M, K = indices.shape
    # Flatten to compute block IDs
    flat = indices.reshape(-1)          # [M*K]
    normalized = codebook[flat.long()]  # [M*K] gather (fused by torch.compile)
    n_total = flat.numel()
    block_id = torch.arange(n_total, device=input_vec.device) // block_size
    block_id = block_id.clamp(max=absmax.shape[0] - 1)  # saturate on last block
    scale = absmax[block_id]            # [M*K]
    weights = (normalized * scale).reshape(M, K)
    return weights @ input_vec           # [M] bf16 output


# Quick test
if __name__ == '__main__':
    torch.manual_seed(42)
    M, K = 2816, 2048
    inp = torch.randn(K, dtype=torch.bfloat16, device='cuda')
    idx = torch.randint(0, 256, (M, K), dtype=torch.uint8, device='cuda')
    cb = torch.linspace(-3, 3, 256, dtype=torch.bfloat16, device='cuda')
    n_blocks = (M * K + 127) // 128
    amax = torch.randn(n_blocks, dtype=torch.bfloat16, device='cuda').abs()

    # Warmup
    for _ in range(5):
        out = fused_lut_gemv(inp, idx, cb, amax)
    torch.cuda.synchronize()

    # Benchmark
    import time
    t0 = time.time()
    for _ in range(100):
        out = fused_lut_gemv(inp, idx, cb, amax)
    torch.cuda.synchronize()
    per = (time.time() - t0) / 100

    # Reference: decompress + F.linear
    t0 = time.time()
    for _ in range(100):
        flat = idx.reshape(-1)
        normalized = cb[flat.long()]
        n = flat.numel()
        block_id = torch.arange(n, device='cuda') // 128
        scale = amax[block_id.clamp(max=n_blocks-1)]
        w = (normalized * scale).reshape(M, K)
        _ = F.linear(inp.unsqueeze(0), w).squeeze(0)
    torch.cuda.synchronize()
    ref_per = (time.time() - t0) / 100

    print(f"fused_lut_gemv:    {per*1000:.2f}ms  ({per*1000*24:.0f}ms for 24 layers)")
    print(f"decompress+linear: {ref_per*1000:.2f}ms  ({ref_per*1000*24:.0f}ms for 24 layers)")
    print(f"Speedup: {ref_per/per:.1f}x")

    # Expert-level benchmark (w13 + w2 per expert)
    K13, K2 = 2048, 1408
    M13, M2 = 2816, 2048
    idx13 = torch.randint(0, 256, (M13, K13), dtype=torch.uint8, device='cuda')
    idx2 = torch.randint(0, 256, (M2, K2), dtype=torch.uint8, device='cuda')
    a13 = torch.randn((M13*K13+127)//128, dtype=torch.bfloat16, device='cuda').abs()
    a2 = torch.randn((M2*K2+127)//128, dtype=torch.bfloat16, device='cuda').abs()

    def one_expert(x):
        gate_up = fused_lut_gemv(x, idx13, cb, a13)
        mid = M13 // 2
        gate, up = gate_up[:mid], gate_up[mid:]
        activated = torch.sigmoid(gate) * up
        out = fused_lut_gemv(activated, idx2, cb, a2)
        return out

    for _ in range(5): one_expert(inp)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): one_expert(inp)
    torch.cuda.synchronize()
    exp_per = (time.time() - t0) / 100

    print(f"\nOne expert (w13 + w2 fused): {exp_per*1000:.2f}ms")
    print(f"4 experts × 24 layers:       {exp_per*4*24*1000:.0f}ms")
    print(f"Estimated TPOT:              {exp_per*4*24*1000:.0f}ms")
    print(f"Estimated tok/s:             {1/(exp_per*4*24):.1f}")
