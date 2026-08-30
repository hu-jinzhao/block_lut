"""
加权 K-means LUT v2: 约束范围版本 + 8-bit baseline

关键修复: 强制最外层质心固定在 ±1.0，内部质心用加权分配。
这样既集中质心于高密度区域，又不截断尾部。

对比: uniform / std-KM / weighted-KM / constrained-weighted-KM
Bit: 4/5/6/8-bit
"""

import os, math, sys, time
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from sklearn.cluster import KMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BS = 128


def collect_normalized(sft_files, max_files=2):
    np.random.seed(42)
    all_vals = []
    for path in tqdm(sft_files[:max_files], desc="Collecting"):
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


def std_kmeans(values, n_centroids):
    """标准 K-means (baseline)"""
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    km = KMeans(n_clusters=n_centroids, random_state=42, n_init=3, max_iter=100, tol=1e-5)
    km.fit(values.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def weighted_kmeans(values, n_centroids, weight_func):
    """加权 K-means (sample replication)"""
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    w = weight_func(values)
    w = np.maximum(w, 1e-10)
    probs = w / w.sum()
    idx = np.random.choice(len(values), size=200000, replace=True, p=probs)
    km = KMeans(n_clusters=n_centroids, random_state=42, n_init=3, max_iter=100, tol=1e-5)
    km.fit(values[idx].reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def constrained_weighted_kmeans(values, n_centroids, weight_func):
    """
    约束范围加权 K-means:
    - 固定最外层 2 个质心 = ±1.0
    - 对内部 n-2 个质心做加权 K-means (只在内部数据上训练)
    - 确保 LUT 覆盖全范围 [-1, 1]
    """
    if n_centroids < 3:
        return weighted_kmeans(values, n_centroids, weight_func)

    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)

    n_interior = n_centroids - 2

    # 对内部数据做加权 K-means (样本本身就是内部的, 不需要额外筛选)
    w = weight_func(values)
    w = np.maximum(w, 1e-10)
    probs = w / w.sum()
    idx = np.random.choice(len(values), size=200000, replace=True, p=probs)
    weighted_sample = values[idx]

    km = KMeans(n_clusters=n_interior, random_state=42, n_init=3, max_iter=100, tol=1e-5)
    km.fit(weighted_sample.reshape(-1, 1).astype(np.float32))
    interior = np.sort(km.cluster_centers_.ravel()).astype(np.float32)

    # 拼合: [-1.0] + interior + [1.0]
    return np.concatenate([[-1.0], interior, [1.0]]).astype(np.float32)


# Weight functions
def make_uniform():
    return lambda x: np.ones_like(x)

def make_gaussian(sigma=0.36):
    return lambda x: np.exp(-x**2 / (2 * sigma**2))

def make_laplace(tau=0.25):
    return lambda x: np.exp(-np.abs(x) / tau)


def find_nearest_vec(values, centroids):
    idx = np.searchsorted(centroids, values)
    idx = np.clip(idx, 1, len(centroids) - 1)
    left = np.abs(values - centroids[idx - 1])
    right = np.abs(values - centroids[idx])
    return np.where(left <= right, idx - 1, idx)


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
    q = np.clip(np.round(blocks / amax[:, np.newaxis] * max_val), -max_val-1, max_val).astype(np.int32)
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

    # ---- Collect ----
    print("[1/3] Collecting normalized values...")
    t0 = time.time()
    train_vals = collect_normalized(sft_files)
    sigma = train_vals.std()
    print(f"  {len(train_vals):,} values, std={sigma:.4f}, time={time.time()-t0:.1f}s")

    # ---- Build LUTs ----
    print("\n[2/3] Building LUTs...")

    # Methods to test: (name_prefix, builder_func, extra_args)
    builders = [
        ("std-KM", std_kmeans, {}),
        ("w-gauss-KM", weighted_kmeans, {"weight_func": make_gaussian(sigma)}),
        ("w-laplace-KM", weighted_kmeans, {"weight_func": make_laplace(0.25)}),
        ("c-gauss-KM", constrained_weighted_kmeans, {"weight_func": make_gaussian(sigma)}),
        ("c-laplace-KM", constrained_weighted_kmeans, {"weight_func": make_laplace(0.25)}),
    ]

    all_luts = {}
    for k in [16, 32, 64, 256]:
        for prefix, builder, extra in builders:
            name = f"{prefix}-{k}"
            t0 = time.time()
            all_luts[name] = builder(train_vals, k, **extra)
            lut = all_luts[name]
            near_0_1 = np.sum(np.abs(lut) < 0.1)
            print(f"  {name:<28}: range=[{lut[0]:.4f},{lut[-1]:.4f}], "
                  f"|x|<0.1: {near_0_1}/{k}, time={time.time()-t0:.1f}s")

    # ---- Gather tensors ----
    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))
    step = max(1, len(all_tensors) // 50)
    sampled = all_tensors[::step][:50]
    print(f"\n[3/3] Evaluating {len(sampled)} tensors...")

    # ---- Evaluate ----
    results = {}

    # Uniform baselines
    for nb in [4, 5, 6, 7, 8]:
        name = f"Uniform {nb}-bit"
        results[name] = {'psnr': [], 'bits': nb + 16/BS}

    # All LUT methods
    for lut_name in all_luts:
        k = int(lut_name.split("-")[-1])
        results[lut_name] = {'psnr': [], 'bits': np.log2(k) + 16/BS}

    for name, W_raw in tqdm(sampled, desc="  Evaluating"):
        W = W_raw.to(torch.float32).numpy().ravel()
        orig_len = len(W)

        for nb in [4, 5, 6, 7, 8]:
            idx, am = quantize_uniform_vec(W, nb)
            recon = dequantize_uniform_vec(idx, am, nb, orig_len)
            results[f"Uniform {nb}-bit"]['psnr'].append(psnr(W, recon))

        for lut_name, lut in all_luts.items():
            idx, am = quantize_blocklut_vec(W, lut)
            recon = dequantize_blocklut_vec(idx, am, lut, orig_len)
            results[lut_name]['psnr'].append(psnr(W, recon))

    # ---- Print full table ----
    print("\n" + "=" * 95)
    print("FULL RESULTS (sorted by bits/elem)")
    print("=" * 95)
    print(f"{'Method':<35} {'bits/elem':>10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10}")
    print("-" * 78)

    for name, data in sorted(results.items(), key=lambda x: x[1]['bits']):
        v = data['psnr']
        print(f"{name:<35} {data['bits']:>10.3f} {np.mean(v):>10.2f} {np.min(v):>10.2f} {np.max(v):>10.2f}")

    # ---- Head-to-head by bit level ----
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD by bit level")
    print("=" * 95)

    for nbits, k in [(4, 16), (5, 32), (6, 64), (8, 256)]:
        print(f"\n--- {nbits}-bit ({k} centroids, {nbits+16/BS:.3f} b/e) ---")
        uni_name = f"Uniform {nbits}-bit"
        uni_p = np.mean(results[uni_name]['psnr'])

        # Find all methods at this k
        methods_at_k = [(name, np.mean(results[name]['psnr']))
                        for name in results if f"KM-{k}" in name and "KM-" in name
                        and name.split("-")[-1] == str(k)]

        # Also include uniform
        all_at_k = [(uni_name, uni_p)] + sorted(methods_at_k, key=lambda x: -x[1])

        std_p = None
        for name, p in all_at_k:
            if name.startswith("std-KM"):
                std_p = p
                break

        best_p = max(p for _, p in all_at_k)
        for name, p in all_at_k:
            delta_uni = p - uni_p
            delta_std = "" if std_p is None else f"(vs std: {p-std_p:+.2f})"
            marker = " ← BEST" if p >= best_p - 0.005 else ""
            print(f"  {name:<35} {p:.2f} dB  (vs uni: {delta_uni:+.2f}) {delta_std}{marker}")

    # ---- Analysis ----
    print("\n" + "=" * 95)
    print("ANALYSIS")
    print("=" * 95)

    # Constrained vs unconstrained weighted vs std at each bit
    for k, nbits in [(16, 4), (32, 5), (64, 6), (256, 8)]:
        std_p = np.mean(results[f"std-KM-{k}"]['psnr'])
        uni_p = np.mean(results[f"Uniform {nbits}-bit"]['psnr'])

        print(f"\n  {nbits}-bit ({k} centroids):")
        print(f"    Uniform:              {uni_p:.2f} dB")
        print(f"    std-KM:               {std_p:.2f} dB  (vs uni: {std_p-uni_p:+.2f})")

        for prefix, label in [("w-gauss-KM", "weighted-gauss"),
                               ("w-laplace-KM", "weighted-laplace"),
                               ("c-gauss-KM", "constrained-gauss"),
                               ("c-laplace-KM", "constrained-laplace")]:
            name = f"{prefix}-{k}"
            p = np.mean(results[name]['psnr'])
            lut = all_luts[name]
            near_01 = np.sum(np.abs(lut) < 0.1)
            print(f"    {label:<22} {p:.2f} dB  (vs std: {p-std_p:+.2f}, vs uni: {p-uni_p:+.2f})  "
                  f"near0: {near_01}/{k} ({100*near_01/k:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
