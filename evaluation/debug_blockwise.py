"""Block-wise 8-bit 量化 prototype - 对比 PSNR"""
import os, sys
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
import numpy as np
import torch, time

CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/model-00001-of-00008.safetensors"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"

from safetensors import safe_open
weights = {}
with safe_open(CKPT, framework="pt", device="cpu") as f:
    for k in f.keys():
        if "expert" in k and "shared_expert" not in k:
            w = f.get_tensor(k).to(torch.bfloat16)
            weights[k] = w

print(f"Loaded {len(weights)} expert tensors")
test_keys = sorted(weights.keys())[:6]
print(f"Testing {len(test_keys)} tensors")

# ---- Global LUT ----
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)
midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)

def lut_quantize_recover(t):
    flat = t.view(torch.int16).numpy().astype(np.uint16).ravel()
    mono = np.where(flat & 0x8000, ~flat, flat ^ np.uint16(0x8000)).astype(np.uint16)
    idx = np.searchsorted(thresholds, mono).astype(np.uint8)
    reco = torch.from_numpy(lut_u16[idx].astype(np.int16)).view(torch.bfloat16)
    return reco.reshape(t.shape)

# ---- Block-wise (uniform, 8-bit) ----
def blockwise_uniform(t, block_size=128):
    """Uniform block-wise: absmax + 8-bit index. Simple and GPU-friendly."""
    flat = t.ravel()
    n = flat.numel()
    nlevels = 255

    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)

    # quantize to [0, 255]
    normalized = blocks.float() / absmax.unsqueeze(1)  # [-1, 1]
    idx = ((normalized + 1.0) / 2.0 * nlevels + 0.5).clamp(0, nlevels).to(torch.uint8)

    # recover
    dequant = (idx.float() / nlevels * 2.0 - 1.0) * absmax.unsqueeze(1)
    reco = dequant.to(torch.bfloat16).ravel()[:n].reshape(t.shape)

    bits_per_elem = (n_blocks * 16 + n * 8) / n
    return reco, bits_per_elem

# ---- Block-wise (non-uniform, simple pow-based table) ----
def blockwise_nf(t, block_size=128):
    """Non-uniform block-wise: absmax + 8-bit index with denser bins near 0."""
    flat = t.ravel()
    n = flat.numel()

    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)
    # absolute values for quantization (symmetric around 0)
    abs_blocks = blocks.abs().float() / absmax.unsqueeze(1)  # [0, 1]

    # Non-uniform quantization: apply power transform to compress near 0
    # x^p with p < 1 spreads out small values, giving them more levels
    p = 0.6
    transformed = abs_blocks ** p  # [0, 1], small values spread out
    idx = (transformed * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    # Recover: inverse transform
    dequant = (idx.float() / 255) ** (1.0 / p)  # [0, 1]
    reco = (torch.sign(blocks.float()) * dequant * absmax.unsqueeze(1)).to(torch.bfloat16)
    reco = reco.ravel()[:n].reshape(t.shape)

    bits_per_elem = (n_blocks * 16 + n * 8) / n
    return reco, bits_per_elem

# ---- Run ----
print(f"\n{'Method':<28} {'PSNR_avg':>8} {'PSNR_min':>8} {'bits/elem':>10} {'time':>8}")
print("-" * 70)

results = []

# LUT
t0 = time.perf_counter()
psnr_vals = []
for k in test_keys:
    t = weights[k]
    reco = lut_quantize_recover(t)
    o = t.float().numpy().ravel(); r = reco.float().numpy().ravel()
    mse = ((o-r)**2).mean()
    psnr_vals.append(10*np.log10(o.var()/mse) if mse>0 else 99)
results.append(("Global LUT (256 centroids)", np.mean(psnr_vals), min(psnr_vals), 8.0, time.perf_counter()-t0))

# Block-wise with various sizes
for bs in [64, 128, 256]:
    for method, func in [("uniform", blockwise_uniform), ("NF (pow 1.5)", blockwise_nf)]:
        name = f"Block{bs} {method}"
        t0 = time.perf_counter()
        psnr_vals, bits_vals = [], []
        for k in test_keys:
            w = weights[k]
            reco, bpe = func(w, bs)
            o = w.float().numpy().ravel(); r = reco.float().numpy().ravel()
            mse = ((o-r)**2).mean()
            psnr_vals.append(10*np.log10(o.var()/mse) if mse>0 else 99)
            bits_vals.append(bpe)
        results.append((name, np.mean(psnr_vals), min(psnr_vals), np.mean(bits_vals), time.perf_counter()-t0))

for r in results:
    print(f"{r[0]:<28} {r[1]:8.1f} {r[2]:8.1f} {r[3]:10.2f} {r[4]:7.2f}s")

# ---- Deep layer test ----
print("\n--- All layers PSNR scan (Block128 uniform, 30 samples) ---")
# Sample 30 tensors from different layers
all_keys = sorted(weights.keys())
sample_step = max(1, len(all_keys) // 30)
sample_keys = all_keys[::sample_step][:30]

psnr_lut_samples, psnr_bw_samples = [], []
for k in sample_keys:
    w = weights[k]
    reco_lut = lut_quantize_recover(w)
    reco_bw, _ = blockwise_uniform(w, 128)
    o = w.float().numpy().ravel()
    for reco, lst in [(reco_lut, psnr_lut_samples), (reco_bw, psnr_bw_samples)]:
        r = reco.float().numpy().ravel()
        mse = ((o-r)**2).mean()
        lst.append(10*np.log10(o.var()/mse) if mse>0 else 99)

print(f"  Global LUT:    min={min(psnr_lut_samples):.1f}, mean={np.mean(psnr_lut_samples):.1f}, max={max(psnr_lut_samples):.1f} dB")
print(f"  Block128 uni:  min={min(psnr_bw_samples):.1f}, mean={np.mean(psnr_bw_samples):.1f}, max={max(psnr_bw_samples):.1f} dB")
print(f"  Improvement: +{np.mean(psnr_bw_samples)-np.mean(psnr_lut_samples):.1f} dB (min: +{min(psnr_bw_samples)-min(psnr_lut_samples):.1f} dB)")
print("\nDone.")
