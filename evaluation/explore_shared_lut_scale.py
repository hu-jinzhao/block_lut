"""
方向 2: 跨 Expert 共享 LUT + Per-Block 缩放因子 (向量化版本)

value = LUT[idx] * scale[block] * absmax
"""

import os, math, sys, time
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from sklearn.cluster import KMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BS = 128


def collect_normalized(sft_files):
    """快速收集：只扫 2 个文件，所有层混在一起。"""
    np.random.seed(42)
    all_vals = []
    for path in tqdm(sft_files[:2], desc="Collecting"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                t = f.get_tensor(k).to(torch.float32).numpy().ravel()
                n = len(t)
                nb = (n + BS - 1) // BS
                pad = nb * BS - n
                if pad:
                    t = np.pad(t, (0, pad))
                sample_nb = min(nb, 20)
                idx = np.random.choice(nb, sample_nb, replace=False)
                for b in idx:
                    s, e = b * BS, (b + 1) * BS
                    block = t[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    all_vals.append(block / amax)
    return np.concatenate(all_vals)


def build_lut(values, n_centroids):
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    km = KMeans(n_clusters=n_centroids, random_state=42, n_init=3, max_iter=100, tol=1e-5)
    km.fit(values.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def find_nearest_vec(values, centroids):
    """向量化最近邻。values: (N,) or (B, BS), centroids: (K,)"""
    idx = np.searchsorted(centroids, values)
    idx = np.clip(idx, 1, len(centroids) - 1)
    left = np.abs(values - centroids[idx - 1])
    right = np.abs(values - centroids[idx])
    return np.where(left <= right, idx - 1, idx)


def learn_scales(lut, values, n_scales):
    """从数据中学习最优 scale 集合。"""
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    n_vals = (len(values) // BS) * BS
    blocks = values[:n_vals].reshape(-1, BS)

    # Assign with scale=1.0
    idx = find_nearest_vec(blocks, lut)  # (n_blocks, BS)
    lut_vals = lut[idx]
    num = np.sum(lut_vals * blocks, axis=1)
    den = np.sum(lut_vals ** 2, axis=1) + 1e-12
    opt_scales = np.clip(num / den, 0.2, 3.0)

    km = KMeans(n_clusters=n_scales, random_state=42, n_init=3, max_iter=100, tol=1e-5)
    km.fit(opt_scales.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def quantize_scaled_lut_vec(tensor_flat, lut, scales):
    """全向量化：scaled LUT 量化整个 tensor。"""
    n = len(tensor_flat)
    nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad:
        tensor_flat = np.pad(tensor_flat, (0, pad))

    blocks = tensor_flat.reshape(nb, BS)  # (nb, BS)
    absmax_vals = np.max(np.abs(blocks), axis=1)  # (nb,)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    norm_blocks = blocks / absmax_vals[:, np.newaxis]  # (nb, BS)

    n_scales = len(scales)
    scaled_luts = lut[np.newaxis, :] * scales[:, np.newaxis]  # (M, K)

    best_indices = np.zeros((nb, BS), dtype=np.uint8)
    best_scales = np.zeros(nb, dtype=np.uint8)

    # For each scale, compute MSE, keep best per block
    for si in range(n_scales):
        sl = scaled_luts[si]  # (K,)
        norm_clipped = np.clip(norm_blocks, sl[0], sl[-1])
        idx = find_nearest_vec(norm_clipped, sl)  # (nb, BS)
        recon = sl[idx]
        mse = np.mean((norm_blocks - recon) ** 2, axis=1)  # (nb,)

        if si == 0:
            best_mse = mse
            best_indices = idx
            best_scales[:] = si
        else:
            improve = mse < best_mse
            best_mse[improve] = mse[improve]
            best_indices[improve] = idx[improve]
            best_scales[improve] = si

    return best_indices.ravel(), best_scales, absmax_vals


def dequantize_scaled_lut_vec(indices, scale_idx, absmax_vals, lut, scales, orig_len):
    nb = len(scale_idx)
    indices = indices.reshape(nb, BS)
    norm = lut[indices.astype(np.int32)] * scales[scale_idx][:, np.newaxis]  # (nb, BS)
    x = (norm * absmax_vals[:, np.newaxis]).ravel()
    return x[:orig_len]


# ---- Single LUT (vectorized) ----
def quantize_blocklut_vec(tensor_flat, lut):
    n = len(tensor_flat)
    nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad:
        tensor_flat = np.pad(tensor_flat, (0, pad))
    blocks = tensor_flat.reshape(nb, BS)
    absmax_vals = np.max(np.abs(blocks), axis=1)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    norm = blocks / absmax_vals[:, np.newaxis]
    idx = find_nearest_vec(norm, lut)
    return idx.ravel(), absmax_vals


def dequantize_blocklut_vec(indices, absmax_vals, lut, orig_len):
    nb = len(absmax_vals)
    indices = indices.reshape(nb, BS)
    x = (lut[indices.astype(np.int32)] * absmax_vals[:, np.newaxis]).ravel()
    return x[:orig_len]


# ---- Uniform (vectorized) ----
def quantize_uniform_vec(x, nbits):
    max_val = 2 ** (nbits - 1) - 1
    n = len(x)
    nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad:
        x = np.pad(x, (0, pad))
    blocks = x.reshape(nb, BS)
    amax = np.max(np.abs(blocks), axis=1)
    amax = np.maximum(amax, 1e-12)
    q = np.clip(np.round(blocks / amax[:, np.newaxis] * max_val), -max_val - 1, max_val).astype(np.int32)
    return (q + max_val + 1).astype(np.uint8).ravel(), amax


def dequantize_uniform_vec(indices, amax, nbits, orig_len):
    max_val = 2 ** (nbits - 1) - 1
    nb = len(amax)
    indices = indices.reshape(nb, BS)
    q = indices.astype(np.float32) - max_val - 1
    x = (q * amax[:, np.newaxis] / max_val).ravel()
    return x[:orig_len]


def psnr(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    var = np.var(orig)
    return float('inf') if mse == 0 else 10 * math.log10(var / mse)


def main():
    sft_files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
                        if f.startswith("model-") and f.endswith(".safetensors")])
    print(f"Model files: {len(sft_files)}")

    # ---- Collect ----
    print("\n[1/4] Collecting normalized values...")
    t0 = time.time()
    train_vals = collect_normalized(sft_files)
    print(f"  {len(train_vals):,} values, std={train_vals.std():.4f}, time={time.time()-t0:.1f}s")

    # ---- Build LUTs & scales ----
    print("\n[2/4] Building LUTs & learning scales...")
    configs = [
        # (lut_size, n_scales, label)
        (16, 4, "LUT16×4scale (4+2bit)"),
        (16, 2, "LUT16×2scale (4+1bit)"),
        (32, 4, "LUT32×4scale (5+2bit)"),
        (32, 2, "LUT32×2scale (5+1bit)"),
        (64, 4, "LUT64×4scale (6+2bit)"),
        (64, 2, "LUT64×2scale (6+1bit)"),
    ]

    shared_luts = {}
    scale_sets = {}

    for lut_size, n_scales, label in configs:
        lut = build_lut(train_vals, lut_size)
        scales = learn_scales(lut, train_vals, n_scales)
        shared_luts[label] = lut
        scale_sets[label] = scales
        ib = int(np.log2(lut_size))
        sb = int(np.log2(n_scales))
        bpe = ib + sb / BS + 16 / BS
        print(f"  {label}: {bpe:.3f} b/e, lut=[{lut[0]:.3f},{lut[-1]:.3f}], scales={scales}")

    # Single LUT baselines
    single_luts = {}
    for k in [16, 32, 64, 128, 256]:
        label = f"Single LUT-{k}"
        single_luts[label] = build_lut(train_vals, k)
        print(f"  {label}: lut=[{single_luts[label][0]:.3f},{single_luts[label][-1]:.3f}]")

    # ---- Gather tensors ----
    print("\n[3/4] Gathering tensors...")
    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))

    step = max(1, len(all_tensors) // 50)
    sampled = all_tensors[::step][:50]
    print(f"  {len(sampled)} tensors selected")

    # ---- Evaluate ----
    print("\n[4/4] Evaluating...")

    methods = []
    # Uniform: (name, type, param, bpe_override)
    for nb in [5, 6, 7, 8]:
        methods.append((f"Uniform {nb}-bit", "uniform", nb, nb + 16/BS))
    # Single LUT
    for k in [16, 32, 64, 128, 256]:
        label = f"Single LUT-{k}"
        methods.append((label, "single", label, np.log2(k) + 16/BS))
    # Scaled LUT
    for lut_size, n_scales, label in configs:
        ib = int(np.log2(lut_size))
        sb = int(np.log2(n_scales))
        methods.append((label, "scaled", label, ib + sb/BS + 16/BS))

    results = {}

    for name, W_raw in tqdm(sampled, desc="  Evaluating"):
        W = W_raw.to(torch.float32).numpy().ravel()
        orig_len = len(W)

        for mname, mtype, param, bpe in methods:
            if mtype == "uniform":
                idx, am = quantize_uniform_vec(W, param)
                recon = dequantize_uniform_vec(idx, am, param, orig_len)
            elif mtype == "single":
                idx, am = quantize_blocklut_vec(W, single_luts[param])
                recon = dequantize_blocklut_vec(idx, am, single_luts[param], orig_len)
            else:  # scaled
                li, si, am = quantize_scaled_lut_vec(W, shared_luts[param], scale_sets[param])
                recon = dequantize_scaled_lut_vec(li, si, am, shared_luts[param], scale_sets[param], orig_len)

            if mname not in results:
                results[mname] = {'psnr': [], 'bits': bpe}
            results[mname]['psnr'].append(psnr(W, recon))

    # ---- Print ----
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"{'Method':<42} {'bits/elem':>10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10}")
    print("-" * 85)

    for name, data in sorted(results.items(), key=lambda x: x[1]['bits']):
        v = data['psnr']
        print(f"{name:<42} {data['bits']:>10.3f} {np.mean(v):>10.2f} {np.min(v):>10.2f} {np.max(v):>10.2f}")

    # ---- Head-to-head ----
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD: Scaled vs Single LUT at same bit rate")
    print("=" * 90)

    uni8 = np.mean(results["Uniform 8-bit"]['psnr'])

    groups = [
        ("~5.1 bits", ["Uniform 5-bit", "Single LUT-32",
                        "LUT16×2scale (4+1bit)"]),
        ("~6.1 bits", ["Uniform 6-bit", "Single LUT-64",
                        "LUT16×4scale (4+2bit)", "LUT32×2scale (5+1bit)"]),
        ("~7.1 bits", ["Uniform 7-bit", "Single LUT-128",
                        "LUT32×4scale (5+2bit)", "LUT64×2scale (6+1bit)"]),
        ("~8.1 bits", ["Uniform 8-bit", "Single LUT-256",
                        "LUT64×4scale (6+2bit)"]),
    ]

    for gname, names in groups:
        print(f"\n--- {gname} ---")
        best = 0
        for n in names:
            if n in results:
                p = np.mean(results[n]['psnr'])
                m = " ← BEST" if p > best else ""
                if p > best:
                    best = p
                print(f"  {n:<40} {results[n]['bits']:.3f} b/e  {p:.2f} dB  (vs uni8: {p-uni8:+.2f} dB){m}")

    print("\nDone.")


if __name__ == "__main__":
    main()
