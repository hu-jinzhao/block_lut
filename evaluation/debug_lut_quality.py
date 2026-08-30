"""诊断 LUT 量化+恢复的正确性"""
import os, sys
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch

LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"

# 加载 LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32)
lut_f32.sort()
print(f"LUT range: [{lut_f32[0]:.6f}, {lut_f32[-1]:.6f}]")
print(f"LUT entries: {len(lut_f32)}")

# bf16 bit patterns (raw uint16, non-monotonic)
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

# thresholds in monotonic uint16 space
midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)

def to_mono(u16):
    """IEEE 754: sign-magnitude to monotonic uint16"""
    return np.where(u16 & 0x8000, ~u16, u16 ^ np.uint16(0x8000)).astype(np.uint16)

thresholds = to_mono(mid_u16)

def quantize(t):
    """量化 bf16 tensor 到 8-bit LUT indices"""
    flat = t.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
    mono = to_mono(flat)
    return np.searchsorted(thresholds, mono).astype(np.uint8)

def recover(indices):
    """从 LUT indices 恢复 bf16 tensor (返回 bf16 torch tensor)"""
    reco_u16 = lut_u16[indices]  # raw bf16 bit patterns
    # view as int16 then reinterpret bits as bf16 (NOT value conversion!)
    return torch.from_numpy(reco_u16.astype(np.int16)).view(torch.bfloat16)

# Test 1: Random tensors
print("\n=== Test 1: Random bf16 tensor ===")
for _ in range(5):
    t = torch.randn(1000, dtype=torch.bfloat16)
    indices = quantize(t)
    recovered_t = recover(indices)

    orig_f32 = t.float()
    reco_f32 = recovered_t.float()
    mse = ((orig_f32 - reco_f32) ** 2).mean().item()
    psnr = 10 * np.log10(orig_f32.var().item() / mse) if mse > 0 else float('inf')
    max_err = (orig_f32 - reco_f32).abs().max().item()
    print(f"  MSE={mse:.6e}, PSNR={psnr:.1f}dB, max_err={max_err:.6f}")

# Test 2: Actual expert weight from safetensors
from safetensors import safe_open
CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"

print("\n=== Test 2: Real expert weight quantization ===")
with safe_open(os.path.join(CKPT, "model-00001-of-00008.safetensors"), framework="pt", device="cpu") as f:
    keys = f.keys()
    expert_keys = [k for k in keys if "expert" in k and "shared_expert" not in k]
    k = expert_keys[0]
    t = f.get_tensor(k).to(torch.bfloat16)
    print(f"Tensor: {k}, shape={list(t.shape)}")

    indices = quantize(t)
    recovered_t = recover(indices).reshape(t.shape)

    orig_f32 = t.float().numpy().ravel()
    reco_f32 = recovered_t.float().numpy().ravel()
    mse = ((orig_f32 - reco_f32) ** 2).mean()
    psnr = 10 * np.log10(orig_f32.var() / mse) if mse > 0 else float('inf')
    max_err = np.abs(orig_f32 - reco_f32).max()
    print(f"  Python quantize+recover: MSE={mse:.6e}, PSNR={psnr:.1f}dB, max_err={max_err:.6f}")

    # Check: what fraction of values match exactly?
    exact_match = (orig_f32 == reco_f32).mean()
    print(f"  Exact match: {exact_match*100:.1f}%")

# Test 3: Verify LUT table against the C++ upload
print("\n=== Test 3: LUT table integrity ===")
print(f"  lut_u16 dtype: {lut_u16.dtype}, shape: {lut_u16.shape}")
print(f"  First 5 values: {lut_u16[:5]}")
# Verify that lut_u16 values correspond to the sorted bf16 values
reco_check = torch.from_numpy(lut_u16.astype(np.int16)).view(torch.bfloat16)
print(f"  lut_bf16[0] = {lut_bf16[0]:.8f}, lut_bf16[127] = {lut_bf16[127]:.8f}, lut_bf16[255] = {lut_bf16[255]:.8f}")
# Check if sorted
lut_f32_check = reco_check.float().numpy()
print(f"  LUT sorted check: {np.all(np.diff(lut_f32_check) >= 0)}")

# Test 4: Sanity check - generate text with a small random model
print("\n=== Test 4: Quantized weight distribution ===")
unique_idx, counts = np.unique(indices, return_counts=True)
print(f"  Unique indices used: {len(unique_idx)}/256")
print(f"  Top 5 indices: {np.argsort(-counts)[:5]} with counts {sorted(counts, reverse=True)[:5]}")
print(f"  Min count: {counts.min()}, Max count: {counts.max()}")

print("\nDone.")
