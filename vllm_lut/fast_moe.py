#!/usr/bin/env python3
"""
Fast MoE kernel for LUT-quantized weights.

Processes all 4 selected experts in a single call, minimizing Python overhead.
Uses torch.compile to fuse operations.
"""

import torch
import torch.nn.functional as F
from .cuda_lut_gemv import lut_gemv as _gemv


@torch.compile(dynamic=True)
def _fused_expert_compute(x, w13_slices, w2_slices, cb, a13, a2):
    """
    Compute one expert: gate_up = GEMV(x, w13) → silu(gate)*up → GEMV(activated, w2)

    All slices are pre-stacked for torch.compile efficiency.
    x: [hidden] bf16
    w13_slices: [M, K] uint8
    w2_slices: [hidden, ni] uint8
    """
    gate_up = _gemv(x, w13_slices, cb, a13)
    ni = gate_up.shape[0] // 2
    activated = F.silu(gate_up[:ni]) * gate_up[ni:]
    return _gemv(activated, w2_slices, cb, a2)


def fused_lut_moe(x, topk_ids, topk_weights, w13_weight, w13_absmax,
                   w2_weight, w2_absmax, codebook):
    """
    Fused MoE: process all selected experts in batch, minimal Python overhead.

    Args:
        x: [num_tokens, hidden] or [hidden] for decode
        topk_ids: [num_tokens, top_k]
        topk_weights: [num_tokens, top_k]
        All other args are from layer attributes.

    Returns:
        [num_tokens, hidden] output
    """
    num_tokens = x.shape[0] if x.dim() == 2 else 1
    hidden = x.shape[-1]
    unique_ids = torch.unique(topk_ids)
    ni = w13_weight.shape[1] // 2
    cb = codebook.to(x.device)

    out = torch.zeros(num_tokens, hidden, dtype=x.dtype, device=x.device)

    for eid in unique_ids.tolist():
        if num_tokens == 1:
            # Decode: fused GEMV via torch.compile
            gate_up = _fused_expert_compute(
                x, w13_weight[eid], w2_weight[eid], cb,
                w13_absmax[eid], w2_absmax[eid])
            for k in range(topk_ids.shape[-1]):
                m = topk_ids[:, k] == eid
                if not m.any(): continue
                out[m] += gate_up * topk_weights[m, k].unsqueeze(-1)
        else:
            # Prefill: decompress + matmul
            flat13 = w13_weight[eid].reshape(-1).long()
            w13_bf16 = (cb[flat13] * w13_absmax[eid][
                torch.arange(flat13.numel(), device=x.device)//128
            ].clamp(max=w13_absmax.shape[1]-1)).reshape(2*ni, hidden)
            gate_up = F.linear(x, w13_bf16)
            activated = F.silu(gate_up[:, :ni]) * gate_up[:, ni:]
            flat2 = w2_weight[eid].reshape(-1).long()
            w2_bf16 = (cb[flat2] * w2_absmax[eid][
                torch.arange(flat2.numel(), device=x.device)//128
            ].clamp(max=w2_absmax.shape[1]-1)).reshape(hidden, ni)
            expert_out = F.linear(activated, w2_bf16)
            for k in range(topk_ids.shape[-1]):
                m = topk_ids[:, k] == eid
                if not m.any(): continue
                out[m] += expert_out[m] * topk_weights[m, k].unsqueeze(-1)

    return out
