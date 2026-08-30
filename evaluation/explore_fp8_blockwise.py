"""
Block-wise FP8 E4M3 量化 vs BLOCKLUT vs int8 uniform — PSNR 对比

FP8 E4M3 格式: 1 sign + 4 exp + 3 mantissa, 天生非均匀量化
实验:
  1. Block128 FP8 (absmax 归一化 → fp8 E4M3)
  2. Pure FP8 (无 block 结构, 直接 per-element fp8)
  3. vs BLOCKLUT (当前最优, 44.36 dB)
  4. vs Block128 int8 uniform (43.5 dB)
"""

import os, math, time
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/models/qwen/blocklut_256.npy"
BLOCK_SIZE = 128


# ═══════════════════════════════════════════════════════════
# FP8 E4M3 codebook
# ═══════════════════════════════════════════════════════════

def generate_fp8_e4m3_codebook():
    """生成 FP8 E4M3 格式的所有可表示值 (排除了 NaN/Inf 编码)."""
    values = []
    for bits in range(256):
        s = (bits >> 7) & 1
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        if exp == 0xF:   # NaN/Inf, 不可用于数据
            continue
        if exp == 0:
            val = (0.0 if mant == 0 else (-1)**s * 2**(-6) * mant / 8.0)
        else:
            val = (-1)**s * 2**(int(exp) - 7) * (1.0 + mant / 8.0)
        values.append(val)
    uniq = np.unique(np.array(values, dtype=np.float32))
    uniq.sort()
    return uniq


def blockwise_fp8_quantize(tensor_flat, codebook, block_size=128):
    """Block128 fp8 E4M3 量化 — 映射到 fp8 完整动态范围 (DeepSeek 做法).

    正确做法: scale = absmax / max_fp8, 将 block 内的值缩放到 fp8 全范围,
    这样量化时可以利用全部 239 个可用码字, 而非仅限于 [-1, 1] 内的 47%.

    Returns (fp8_indices, scales_bf16_as_int16).
    """
    x = tensor_flat.astype(np.float32).copy()
    n = x.size
    bs = block_size
    nb = (n + bs - 1) // bs
    pad = nb * bs - n
    if pad > 0:
        x = np.pad(x, (0, pad))

    max_fp8 = np.max(np.abs(codebook))  # 240 for E4M3
    blocks = x.reshape(nb, bs)
    absmax_vals = np.max(np.abs(blocks), axis=1)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    scales = absmax_vals / max_fp8  # per-block scale

    # Scale values to full fp8 range: scaled ∈ [-max_fp8, max_fp8]
    scaled = blocks / scales[:, np.newaxis]
    flat_scaled = scaled.ravel()

    # Nearest-neighbor in full fp8 codebook (all 239 usable codes)
    idx = np.searchsorted(codebook, flat_scaled)
    idx = np.clip(idx, 1, len(codebook) - 1)
    left = codebook[idx - 1]
    right = codebook[idx]
    idx = np.where(flat_scaled - left <= right - flat_scaled, idx - 1, idx)
    return idx.astype(np.uint8), scales.astype(np.float32)


def blockwise_fp8_dequant(fp8_indices, scales, codebook, block_size, orig_len):
    """fp8 值 × scale 恢复."""
    n = fp8_indices.size
    x = np.zeros(n, dtype=np.float32)
    ids = fp8_indices.astype(np.int32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        fp8_vals = codebook[ids[s:e]]
        x[s:e] = fp8_vals * scales[b]
    return x[:orig_len]


def blockwise_fp8_dequant(fp8_indices, absmax_vals, codebook, block_size, orig_len):
    """Block128 absmax × fp8 反量化."""
    n = fp8_indices.size
    x = np.zeros(n, dtype=np.float32)
    ids = fp8_indices.astype(np.int32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        normalized = codebook[ids[s:e]]
        x[s:e] = normalized * absmax_vals[b]
    return x[:orig_len]


def pure_fp8_quantize(tensor_flat, codebook):
    """纯 per-element FP8 量化 (无 block 结构).

    Returns fp8_indices only (no absmax).
    """
    x = tensor_flat.astype(np.float32).ravel()
    idx = np.searchsorted(codebook, x)
    idx = np.clip(idx, 1, len(codebook) - 1)
    left = codebook[idx - 1]
    right = codebook[idx]
    idx = np.where(x - left <= right - x, idx - 1, idx)
    return idx.astype(np.uint8)


def pure_fp8_dequant(fp8_indices, codebook):
    """纯 per-element FP8 反量化."""
    return codebook[fp8_indices.astype(np.int32)]


# ═══════════════════════════════════════════════════════════
# BLOCKLUT (baseline)
# ═══════════════════════════════════════════════════════════

def blocklut_quantize(tensor_flat, lut, block_size=128):
    """Block128 absmax + LUT-256 量化 (当前集成方案)."""
    x = tensor_flat.astype(np.float32).copy()
    n = x.size
    bs = block_size
    nb = (n + bs - 1) // bs
    pad = nb * bs - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    blocks = x.reshape(nb, bs)
    absmax_vals = np.max(np.abs(blocks), axis=1)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    normalized = blocks / absmax_vals[:, np.newaxis]
    flat_norm = normalized.ravel()
    idx = np.searchsorted(lut, flat_norm)
    idx = np.clip(idx, 1, len(lut) - 1)
    left = lut[idx - 1]
    right = lut[idx]
    idx = np.where(flat_norm - left <= right - flat_norm, idx - 1, idx)
    return idx.astype(np.uint8), absmax_vals.astype(np.float32)


def blocklut_dequant(indices, absmax_vals, lut, block_size, orig_len):
    """等同于 blockwise_fp8_dequant."""
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    ids = indices.astype(np.int32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        x[s:e] = lut[ids[s:e]] * absmax_vals[b]
    return x[:orig_len]


# ═══════════════════════════════════════════════════════════
# Block-wise int8 uniform (baseline)
# ═══════════════════════════════════════════════════════════

def blockwise_int8_quantize(tensor_flat, block_size=128):
    x = tensor_flat.astype(np.float32).copy()
    n = x.size
    bs = block_size
    nb = (n + bs - 1) // bs
    pad = nb * bs - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    blocks = x.reshape(nb, bs)
    absmax_vals = np.max(np.abs(blocks), axis=1)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    normalized = blocks / absmax_vals[:, np.newaxis]
    max_val = 127.5
    q = np.clip(np.round(normalized * max_val), -127, 127).astype(np.int8)
    return q.ravel().view(np.uint8), absmax_vals.astype(np.float32)


def blockwise_int8_dequant(indices, absmax_vals, block_size, orig_len):
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    q = indices.view(np.int8).astype(np.float32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        x[s:e] = q[s:e] * absmax_vals[b] / 127.5
    return x[:orig_len]


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr_var(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    var = np.var(orig)
    return float('inf') if mse == 0 else 10 * math.log10(var / mse)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Checkpoint files: {len(sft_files)}")

    # ---- 1. Generate FP8 codebook ----
    print("\n[1/5] Generating FP8 E4M3 codebook...")
    t0 = time.perf_counter()
    fp8_cb = generate_fp8_e4m3_codebook()
    n_fp8 = len(fp8_cb)
    in_range = np.sum(np.abs(fp8_cb) <= 1.0)
    print(f"  Total usable codes: {n_fp8} (excl. NaN/Inf)")
    print(f"  Codes in [-1, 1]: {in_range} ({in_range/n_fp8*100:.0f}%)")
    print(f"  Range: [{fp8_cb[0]:.6f}, {fp8_cb[-1]:.2f}]")
    print(f"  Step near 0: {fp8_cb[n_fp8//2+1] - fp8_cb[n_fp8//2]:.6f}")
    print(f"  Step near 1: {fp8_cb[-1] - fp8_cb[-2]:.4f}")
    print(f"  Codebook gen time: {time.perf_counter()-t0:.3f}s")

    # ---- 2. Load BLOCKLUT ----
    print("\n[2/5] Loading BLOCKLUT codebook...")
    lut = np.load(LUT_PATH).astype(np.float32)
    lut.sort()
    print(f"  LUT entries: {len(lut)}, range=[{lut[0]:.4f}, {lut[-1]:.4f}]")

    # ---- 3. Collect expert tensors ----
    print("\n[3/5] Collecting expert tensors...")
    t0 = time.perf_counter()
    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))
    print(f"  Total expert tensors: {len(all_tensors)}")
    print(f"  Load time: {time.perf_counter()-t0:.1f}s")

    # Sample ~200 tensors across all layers
    sample_step = max(1, len(all_tensors) // 200)
    sampled = all_tensors[::sample_step][:200]
    print(f"  Sampling {len(sampled)} tensors for evaluation")

    # ---- 4. Evaluate ----
    print("\n[4/5] Evaluating PSNR...")
    methods = {
        'Block128 int8 (uniform)': {
            'psnr': [], 'bits': 8.0 + 16/BLOCK_SIZE,
            'quant': blockwise_int8_quantize,
            'dequant': blockwise_int8_dequant,
        },
        'BLOCKLUT (K-means 256)': {
            'psnr': [], 'bits': 8.0 + 16/BLOCK_SIZE,
            'quant': lambda x: blocklut_quantize(x, lut, BLOCK_SIZE),
            'dequant': lambda *a: blocklut_dequant(*a, lut),
        },
        'Block128 FP8 E4M3': {
            'psnr': [], 'bits': 8.0 + 16/BLOCK_SIZE,
            'quant': lambda x: blockwise_fp8_quantize(x, fp8_cb, BLOCK_SIZE),
            'dequant': lambda *a: blockwise_fp8_dequant(*a, fp8_cb),
        },
        'Pure FP8 E4M3 (no block)': {
            'psnr': [], 'bits': 8.0,
            'quant': lambda x: (pure_fp8_quantize(x, fp8_cb), None),
            'dequant': lambda indices, absmax_vals, orig_len: pure_fp8_dequant(indices, fp8_cb),
        },
    }

    t0 = time.perf_counter()
    for _name, tensor in tqdm(sampled, desc="  Evaluating"):
        W = tensor.to(torch.float32).numpy().ravel()
        orig_len = int(W.size)

        # Block128 int8
        indices, am = blockwise_int8_quantize(W, BLOCK_SIZE)
        reco = blockwise_int8_dequant(indices, am, BLOCK_SIZE, orig_len)
        methods['Block128 int8 (uniform)']['psnr'].append(psnr_var(W, reco))

        # BLOCKLUT
        indices, am = blocklut_quantize(W, lut, BLOCK_SIZE)
        reco = blocklut_dequant(indices, am, lut, BLOCK_SIZE, orig_len)
        methods['BLOCKLUT (K-means 256)']['psnr'].append(psnr_var(W, reco))

        # Block128 FP8
        indices, am = blockwise_fp8_quantize(W, fp8_cb, BLOCK_SIZE)
        reco = blockwise_fp8_dequant(indices, am, fp8_cb, BLOCK_SIZE, orig_len)
        methods['Block128 FP8 E4M3']['psnr'].append(psnr_var(W, reco))

        # Pure FP8
        indices, am = pure_fp8_quantize(W, fp8_cb), None
        reco = pure_fp8_dequant(indices, fp8_cb)
        methods['Pure FP8 E4M3 (no block)']['psnr'].append(psnr_var(W, reco))

    elapsed = time.perf_counter() - t0
    print(f"\n  Evaluate time: {elapsed:.1f}s ({elapsed/len(sampled):.2f}s/tensor)")

    # ---- 5. Results ----
    print("\n[5/5] Results")
    print(f"\n{'Method':<35} {'bits/elem':>10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10}")
    print("-" * 85)

    baseline_psnr = np.mean(methods['Block128 int8 (uniform)']['psnr'])
    for name, data in methods.items():
        valid = data['psnr']
        mean_p = np.mean(valid)
        min_p = np.min(valid)
        max_p = np.max(valid)
        delta = mean_p - baseline_psnr
        delta_str = f"  (vs int8: {delta:+.2f} dB)"
        print(f"{name:<35} {data['bits']:>10.3f} {mean_p:>10.2f} {min_p:>10.2f} {max_p:>10.2f}{delta_str}")

    # ---- Analysis ----
    print(f"\n{'='*85}")
    print("ANALYSIS")
    print(f"{'='*85}")

    fp8_block = np.mean(methods['Block128 FP8 E4M3']['psnr'])
    fp8_pure = np.mean(methods['Pure FP8 E4M3 (no block)']['psnr'])
    blut = np.mean(methods['BLOCKLUT (K-means 256)']['psnr'])
    int8 = baseline_psnr

    print(f"\n  Block128 int8:   {int8:.2f} dB  (baseline uniform)")
    print(f"  BLOCKLUT:        {blut:.2f} dB  (K-means optimized, current best)")
    print(f"  Block128 FP8:    {fp8_block:.2f} dB  (absmax + fp8 E4M3)")
    print(f"  Pure FP8:        {fp8_pure:.2f} dB  (per-element fp8, 8.0 bits/elem)")

    best = max(blut, fp8_block, fp8_pure)
    best_name = ("BLOCKLUT" if best == blut else
                 "Block128 FP8" if best == fp8_block else "Pure FP8")
    print(f"\n  → Winner: {best_name} at {best:.2f} dB")

    # Characterize FP8 quantization quality
    print(f"\n  FP8 E4M3 codebook characterization:")
    # Count codes in different value ranges
    cb = fp8_cb
    pos_cb = cb[cb >= 0]
    ranges = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    for lo, hi in ranges:
        cnt = np.sum((pos_cb >= lo) & (pos_cb < hi))
        avg_step = (hi - lo) / max(cnt, 1)
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:3d} codes, avg step ~{avg_step:.5f}")

    # Compare with int8 uniform step in normalized space
    int8_step = 1.0 / 127.5
    print(f"    int8 uniform step (normalized): {int8_step:.5f}")

    # FP8 recovery speed advantage note
    print(f"\n  Recovery kernel comparison (theoretical):")
    print(f"    BLOCKLUT: SMEM load(256) + sync + random lookup + bf16 mul")
    print(f"    FP8 E4M3: bit-shift exp+mantissa → bf16 (2-3 ALU ops)")
    print(f"    → FP8 recover should be 3-5x faster than BLOCKLUT")


if __name__ == "__main__":
    main()
