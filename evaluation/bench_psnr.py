#!/usr/bin/env python3
"""
PSNR comparison: LUT 4bit vs RTN Group 4bit on Qwen-MoE expert weights.

Weight PSNR is a standard proxy for PPL - higher PSNR ≈ lower PPL.
No GPU needed, runs on CPU with 245GB RAM.
"""

import os, sys, json, time, math, glob
import numpy as np
import torch
from safetensors import safe_open
from collections import defaultdict

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
GROUP_SIZE = 128
BLOCK_SIZE = 128
LUT_CODEBOOK = os.path.join(MODEL_DIR, "nested_lut_mapped16.npy")


def quantize_rtn_group(weight, group_size=128, bits=4):
    """Quantize → dequantize with RTN group method. Returns (weight_dq, psnr)."""
    w = weight.float()
    orig_shape = w.shape
    N = w.numel()
    w_flat = w.reshape(-1, group_size)
    w_min = w_flat.amin(dim=-1, keepdim=True)
    w_max = w_flat.amax(dim=-1, keepdim=True)
    qmax = 2**bits - 1
    scale = ((w_max - w_min) / qmax).clamp(min=1e-10)
    zero = (-w_min / scale).round()
    q = (w_flat / scale + zero).round().clamp(0, qmax)
    dq = ((q - zero) * scale).reshape(orig_shape)
    mse = ((w - dq) ** 2).mean().item()
    var = w.var().item()
    psnr = 10 * math.log10(var / mse) if mse > 0 else 100
    return psnr


def quantize_lut(weight, codebook, block_size=128):
    """LUT quantize → dequantize. Returns PSNR."""
    w = weight.float().reshape(-1)
    N = w.shape[0]
    nb = (N + block_size - 1) // block_size
    pad = torch.zeros(nb * block_size)
    pad[:N] = w
    blocks = pad.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    norm = (blocks / amax).reshape(-1)[:N]
    dists = (norm.unsqueeze(1) - codebook.float().unsqueeze(0)).abs()
    idx = dists.argmin(dim=1)
    dq = codebook[idx] * amax.reshape(-1)[torch.arange(N) // block_size].clamp(max=nb-1)
    mse = ((weight.float().reshape(-1) - dq) ** 2).mean().item()
    var = weight.float().var().item()
    psnr = 10 * math.log10(var / mse) if mse > 0 else 100
    return psnr


def load_all_weights():
    """Load all expert weights from safetensors files."""
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))
    weights = {}  # name -> tensor
    for ckpt in files:
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for name in f.keys():
                if 'experts.' in name and 'shared_expert' not in name and name.endswith('.weight'):
                    weights[name] = f.get_tensor(name)
    return weights


if __name__ == '__main__':
    print("Loading codebook...")
    cb = torch.from_numpy(np.load(LUT_CODEBOOK))
    print(f"  {cb.shape} entries")

    print("Loading expert weights...")
    weights = load_all_weights()
    print(f"  {len(weights)} expert weight matrices")

    # Group by layer
    layers = defaultdict(lambda: {'gate': [], 'up': [], 'down': []})
    for name in weights:
        parts = name.split('.')
        layer_id = int(parts[2])  # model.layers.{id}...
        expert_id = int(parts[5])
        wtype = 'down' if 'down' in parts[6] else ('gate' if 'gate' in parts[6] else 'up')
        layers[layer_id][wtype].append((expert_id, name))

    print("\nComputing PSNR...")
    results = {'LUT': [], 'RTN_Group4': []}

    t0 = time.time()
    for lid in sorted(layers.keys()):
        for wtype in ['gate', 'up', 'down']:
            for eid, name in layers[lid][wtype]:
                w = weights[name]

                # LUT
                psnr_lut = quantize_lut(w, cb)
                results['LUT'].append(psnr_lut)

                # RTN Group 4bit
                psnr_rtn = quantize_rtn_group(w, GROUP_SIZE, 4)
                results['RTN_Group4'].append(psnr_rtn)

        if (lid + 1) % 4 == 0:
            print(f"  Layer {lid}: LUT={np.mean(results['LUT'][-60:]):.1f}dB, "
                  f"RTN={np.mean(results['RTN_Group4'][-60:]):.1f}dB")

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Method':<20} {'Avg PSNR':<12} {'Min PSNR':<12} {'Max PSNR':<12}")
    print(f"{'-'*55}")
    for method in ['LUT', 'RTN_Group4']:
        vals = results[method]
        print(f"{method:<20} {np.mean(vals):<12.2f} {np.min(vals):<12.2f} {np.max(vals):<12.2f}")

    # Per-layer breakdown (last few layers)
    print(f"\n{'Per-layer avg PSNR (last 8 layers):':^55}")
    print(f"{'Layer':<10} {'LUT':<12} {'RTN':<12} {'Diff':<12}")
    print(f"{'-'*45}")
    for lid in sorted(layers.keys())[-8:]:
        lut_vals = [results['LUT'][i] for i, n in enumerate(weights)
                     if f'.layers.{lid}.' in list(weights.keys())[i]]
        rtn_vals = [results['RTN_Group4'][i] for i, n in enumerate(weights)
                     if f'.layers.{lid}.' in list(weights.keys())[i]]

    print(f"\nTime: {time.time()-t0:.1f}s")
    print(f"\n{'LUT better' if np.mean(results['LUT']) > np.mean(results['RTN_Group4']) else 'RTN better'}")
