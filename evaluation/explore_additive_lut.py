"""
Additive Dual-LUT (加性双LUT) prototype.

核心思路:
  value ≈ (LUT_coarse[idx_coarse] + LUT_fine[idx_fine]) * absmax

有效表达空间: K1 × K2 个值，但只需存储 K1 + K2 个质心参数。
与单级 K-means LUT 对比：同 bit 预算下参数效率更高，
与 AQLM 的差异：纯 K-means 逐级残差，无需反向传播训练。

实验配置:
  - 6-bit index: (3+3)=64 eff, (4+2)=64 eff
  - 7-bit index: (4+3)=128 eff
  - 对比: uniform 同 bit, 单级 LUT 同 bit, BLOCKLUT-256 (8-bit baseline)
"""

import os, math, sys, time
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from sklearn.cluster import KMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BS = 128


def collect_normalized_values(sft_files, block_size=BS, max_samples_per_tensor=30000):
    """收集所有 expert tensor 的 block128 absmax 归一化值用于 K-means 训练。"""
    all_normalized = []
    for path in tqdm(sft_files, desc="Collecting normalized values"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                tensor = f.get_tensor(k).to(torch.float32).numpy().ravel()
                n = tensor.size
                nb = (n + block_size - 1) // block_size
                pad = nb * block_size - n
                if pad > 0:
                    tensor = np.pad(tensor, (0, pad))

                sample_blocks = min(nb, max(80, max_samples_per_tensor // block_size))
                block_indices = np.random.choice(nb, sample_blocks, replace=False)

                for b in block_indices:
                    s, e = b * block_size, (b + 1) * block_size
                    block = tensor[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    all_normalized.append(block / amax)

    return np.concatenate(all_normalized)


def build_lut(values, n_centroids, random_state=42):
    """K-means LUT, 返回排序后的质心。"""
    if len(values) > 500000:
        idx = np.random.choice(len(values), 500000, replace=False)
        values = values[idx]
    values = values.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=n_centroids, random_state=random_state,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(values)
    centroids = np.sort(km.cluster_centers_.ravel()).astype(np.float32)
    return centroids


def build_additive_lut(values, k1, k2, random_state=42):
    """
    构建加性双 LUT。

    方法: 贪心逐级残差 + 交替优化

    1. K-means K1 → coarse LUT
    2. 计算 residual = x - coarse_centroid
    3. K-means K2 on residuals → fine LUT
    4. 交替优化 3 轮:
       a. 固定 fine LUT, 对每个点搜索最优 (c,f) 组合 → 更新 coarse LUT
       b. 固定 coarse LUT, 对每个点搜索最优 (c,f) 组合 → 更新 fine LUT

    返回 (coarse_lut, fine_lut) 均排序
    """
    if len(values) > 500000:
        idx = np.random.choice(len(values), 500000, replace=False)
        values = values[idx]

    x = values.reshape(-1, 1).astype(np.float32)
    n = x.shape[0]

    # Step 1: Coarse LUT
    km = KMeans(n_clusters=k1, random_state=random_state,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(x)
    coarse = np.sort(km.cluster_centers_.ravel()).astype(np.float32)

    # Step 2: Fine LUT on residuals
    coarse_idx = find_nearest(x.ravel(), coarse)
    residuals = x.ravel() - coarse[coarse_idx]
    residuals = residuals.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=k2, random_state=random_state + 1,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(residuals)
    fine = np.sort(km.cluster_centers_.ravel()).astype(np.float32)

    # Step 3-5: Alternating refinement
    for iteration in range(3):
        # 3a: Fix fine, optimize coarse
        # For each point, find (c,f) pair that minimizes error, then update coarse
        # Given the small codebook sizes, we can brute-force search
        coarse_new = np.zeros(k1, dtype=np.float32)
        coarse_counts = np.zeros(k1, dtype=np.int32)

        for i in range(0, n, 4096):  # Batch to manage memory
            batch = x[i:i + 4096].ravel()

            # Shape: (batch, k1, k2) → compute error for all combos
            c = coarse[np.newaxis, :, np.newaxis]  # (1, k1, 1)
            f = fine[np.newaxis, np.newaxis, :]     # (1, 1, k2)

            all_vals = c + f              # (1, k1, k2)
            err = np.abs(batch[:, np.newaxis, np.newaxis] - all_vals)  # (batch, k1, k2)

            # Find best (c, f) for each point
            flat_idx = np.argmin(err.reshape(len(batch), -1), axis=1)
            best_c = flat_idx // k2
            best_f = flat_idx % k2

            # Accumulate: for coarse update, subtract fine contribution
            target_c = batch - fine[best_f]
            for j in range(k1):
                mask = best_c == j
                if mask.any():
                    coarse_new[j] += target_c[mask].sum()
                    coarse_counts[j] += mask.sum()

        for j in range(k1):
            if coarse_counts[j] > 0:
                coarse[j] = coarse_new[j] / coarse_counts[j]
        coarse = np.sort(coarse)

        # 3b: Fix coarse, optimize fine
        fine_new = np.zeros(k2, dtype=np.float32)
        fine_counts = np.zeros(k2, dtype=np.int32)

        for i in range(0, n, 4096):
            batch = x[i:i + 4096].ravel()

            c = coarse[np.newaxis, :, np.newaxis]
            f = fine[np.newaxis, np.newaxis, :]
            all_vals = c + f
            err = np.abs(batch[:, np.newaxis, np.newaxis] - all_vals)

            flat_idx = np.argmin(err.reshape(len(batch), -1), axis=1)
            best_c = flat_idx // k2
            best_f = flat_idx % k2

            target_f = batch - coarse[best_c]
            for j in range(k2):
                mask = best_f == j
                if mask.any():
                    fine_new[j] += target_f[mask].sum()
                    fine_counts[j] += mask.sum()

        for j in range(k2):
            if fine_counts[j] > 0:
                fine[j] = fine_new[j] / fine_counts[j]
        fine = np.sort(fine)

    return coarse, fine


def find_nearest(values, centroids):
    """向量化最近邻搜索。"""
    idx = np.searchsorted(centroids, values)
    idx = np.clip(idx, 1, len(centroids) - 1)
    left = np.abs(values - centroids[idx - 1])
    right = np.abs(values - centroids[idx])
    idx = np.where(left <= right, idx - 1, idx)
    return idx


def find_nearest_additive(values, coarse_lut, fine_lut):
    """找最佳 coarse+fine 组合。对每个值搜索 K1×K2 空间。"""
    k1, k2 = len(coarse_lut), len(fine_lut)
    # All possible sums: (k1, k2)
    all_sums = coarse_lut[:, np.newaxis] + fine_lut[np.newaxis, :]  # (k1, k2)
    all_sums_flat = all_sums.ravel()  # (k1*k2,)

    idx = np.searchsorted(all_sums_flat, values)
    idx = np.clip(idx, 1, len(all_sums_flat) - 1)
    left = np.abs(values - all_sums_flat[idx - 1])
    right = np.abs(values - all_sums_flat[idx])
    best_flat = np.where(left <= right, idx - 1, idx)

    ci = best_flat // k2
    fi = best_flat % k2
    return ci, fi


def quantize_blocklut(tensor_flat, lut, block_size=BS):
    """单级 LUT 量化 (现有 BLOCKLUT 方案)。"""
    n = tensor_flat.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        tensor_flat = np.pad(tensor_flat, (0, pad))

    indices = np.zeros(nb * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)

    for b in range(nb):
        s, e = b * block_size, (b + 1) * block_size
        block = tensor_flat[s:e]
        amax = np.max(np.abs(block))
        if amax < 1e-12:
            amax = 1e-12
        absmax_vals[b] = amax
        normalized = block / amax
        idx = find_nearest(normalized, lut)
        indices[s:e] = idx.astype(np.uint8)

    return indices, absmax_vals


def dequantize_blocklut(indices, absmax_vals, lut, block_size, orig_len):
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        normalized = lut[indices[s:e].astype(np.int32)]
        x[s:e] = normalized * absmax_vals[b]
    return x[:orig_len]


def quantize_additive_lut(tensor_flat, coarse_lut, fine_lut, block_size=BS):
    """加性双 LUT 量化。"""
    n = tensor_flat.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        tensor_flat = np.pad(tensor_flat, (0, pad))

    k1, k2 = len(coarse_lut), len(fine_lut)
    indices_c = np.zeros(nb * block_size, dtype=np.uint8)
    indices_f = np.zeros(nb * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)

    for b in range(nb):
        s, e = b * block_size, (b + 1) * block_size
        block = tensor_flat[s:e]
        amax = np.max(np.abs(block))
        if amax < 1e-12:
            amax = 1e-12
        absmax_vals[b] = amax
        normalized = block / amax
        ci, fi = find_nearest_additive(normalized, coarse_lut, fine_lut)
        indices_c[s:e] = ci.astype(np.uint8)
        indices_f[s:e] = fi.astype(np.uint8)

    return indices_c, indices_f, absmax_vals


def dequantize_additive_lut(indices_c, indices_f, absmax_vals,
                            coarse_lut, fine_lut, block_size, orig_len):
    n = indices_c.size
    x = np.zeros(n, dtype=np.float32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        normalized = (coarse_lut[indices_c[s:e].astype(np.int32)] +
                      fine_lut[indices_f[s:e].astype(np.int32)])
        x[s:e] = normalized * absmax_vals[b]
    return x[:orig_len]


def blockwise_uniform(x, block_size, nbits):
    """Block128 uniform 量化。"""
    max_val = 2 ** (nbits - 1) - 1
    n = x.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    indices = np.zeros(nb * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s, e = b * block_size, (b + 1) * block_size
        block = x[s:e]
        amax = np.max(np.abs(block)) or 1e-12
        absmax_vals[b] = amax
        q = np.clip(np.round(block / amax * max_val), -max_val - 1, max_val).astype(np.int32)
        indices[s:e] = (q + max_val + 1).astype(np.uint8)  # offset to [0, 2*max_val+1]
    return indices, absmax_vals


def dequantize_uniform(indices, absmax_vals, block_size, nbits, orig_len):
    max_val = 2 ** (nbits - 1) - 1
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b * block_size, min((b + 1) * block_size, n)
        q = indices[s:e].astype(np.float32) - max_val - 1  # de-offset
        x[s:e] = q * absmax_vals[b] / max_val
    return x[:orig_len]


def compute_psnr(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    var = np.var(orig)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(var / mse)


def main():
    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Model files: {len(sft_files)}")

    # ---- Collect normalized values ----
    print("\n" + "=" * 100)
    print("STEP 1: Collecting absmax-normalized values for LUT training")
    print("=" * 100)
    np.random.seed(42)
    normalized_values = collect_normalized_values(sft_files, BS)
    print(f"  Collected: {len(normalized_values):,} values")
    print(f"  Range: [{normalized_values.min():.4f}, {normalized_values.max():.4f}]")
    print(f"  Std: {normalized_values.std():.4f}")
    near_zero = np.mean(np.abs(normalized_values) < 0.1)
    print(f"  Near zero (|x|<0.1): {near_zero*100:.1f}%")

    # ---- Build LUTs ----
    print("\n" + "=" * 100)
    print("STEP 2: Building LUTs (single + additive)")
    print("=" * 100)

    # Single LUT baselines
    single_configs = {
        'LUT-64 (6-bit)': 64,
        'LUT-128 (7-bit)': 128,
        'LUT-256 (8-bit, BLOCKLUT baseline)': 256,
    }

    single_luts = {}
    for name, k in single_configs.items():
        t0 = time.time()
        single_luts[name] = build_lut(normalized_values, k)
        print(f"  {name}: {time.time()-t0:.1f}s")

    # Additive LUTs
    additive_configs = [
        (8, 8, "Additive 3+3bit (64 eff)"),     # 6-bit, 8×8=64
        (16, 4, "Additive 4+2bit (64 eff)"),     # 6-bit, 16×4=64
        (16, 8, "Additive 4+3bit (128 eff)"),    # 7-bit, 16×8=128
        (32, 4, "Additive 5+2bit (128 eff)"),    # 7-bit, 32×4=128
        (4, 8, "Additive 2+3bit (32 eff)"),      # 5-bit, 4×8=32
        (8, 4, "Additive 3+2bit (32 eff)"),      # 5-bit, 8×4=32
    ]

    additive_luts = {}
    for k1, k2, name in additive_configs:
        t0 = time.time()
        coarse, fine = build_additive_lut(normalized_values, k1, k2)
        additive_luts[name] = (coarse, fine)
        print(f"  {name}: {time.time()-t0:.1f}s, "
              f"coarse=[{coarse[0]:.4f},{coarse[-1]:.4f}], "
              f"fine=[{fine[0]:.4f},{fine[-1]:.4f}]")

    # ---- Evaluate on sample tensors ----
    print("\n" + "=" * 100)
    print("STEP 3: Evaluating PSNR on sampled tensors")
    print("=" * 100)

    # Gather all expert tensors
    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))

    sample_step = max(1, len(all_tensors) // 200)
    sampled = all_tensors[::sample_step][:200]
    print(f"  Evaluating {len(sampled)} tensors")

    # Define all methods to test
    # (name, bits_index, bits_absmax, method_key)
    methods = [
        # Baselines
        ("Uniform 8-bit", 8, 16 / BS, "uniform8"),
        ("Uniform 7-bit", 7, 16 / BS, "uniform7"),
        ("Uniform 6-bit", 6, 16 / BS, "uniform6"),
        # Single LUT
        ("LUT-256 (8-bit, BLOCKLUT baseline)", 8, 16 / BS, "lut256"),
        ("LUT-128 (7-bit)", 7, 16 / BS, "lut128"),
        ("LUT-64 (6-bit)", 6, 16 / BS, "lut64"),
        # Additive LUTs
        ("Additive 3+3bit (64 eff)", 6, 16 / BS, "add_3+3"),
        ("Additive 4+2bit (64 eff)", 6, 16 / BS, "add_4+2"),
        ("Additive 4+3bit (128 eff)", 7, 16 / BS, "add_4+3"),
        ("Additive 5+2bit (128 eff)", 7, 16 / BS, "add_5+2"),
        ("Additive 2+3bit (32 eff)", 5, 16 / BS, "add_2+3"),
        ("Additive 3+2bit (32 eff)", 5, 16 / BS, "add_3+2"),
    ]

    results = {m[0]: {'psnr': [], 'bits': m[1] + m[2]} for m in methods}

    for tensor_name, tensor in tqdm(sampled, desc="  Evaluating"):
        W = tensor.to(torch.float32).numpy().ravel()
        orig_len = W.size

        # Uniform baselines
        for nbits, key in [(8, "uniform8"), (7, "uniform7"), (6, "uniform6")]:
            idx, am = blockwise_uniform(W, BS, nbits)
            recon = dequantize_uniform(idx, am, BS, nbits, orig_len)
            results[f"Uniform {nbits}-bit"]['psnr'].append(compute_psnr(W, recon))

        # Single LUT
        for lut_name, key in [("LUT-256 (8-bit, BLOCKLUT baseline)", "lut256"),
                               ("LUT-128 (7-bit)", "lut128"),
                               ("LUT-64 (6-bit)", "lut64")]:
            idx, am = quantize_blocklut(W, single_luts[lut_name], BS)
            recon = dequantize_blocklut(idx, am, single_luts[lut_name], BS, orig_len)
            results[lut_name]['psnr'].append(compute_psnr(W, recon))

        # Additive LUT
        for (k1, k2, add_name), method_key in [
            ((8, 8, "Additive 3+3bit (64 eff)"), "add_3+3"),
            ((16, 4, "Additive 4+2bit (64 eff)"), "add_4+2"),
            ((16, 8, "Additive 4+3bit (128 eff)"), "add_4+3"),
            ((32, 4, "Additive 5+2bit (128 eff)"), "add_5+2"),
            ((4, 8, "Additive 2+3bit (32 eff)"), "add_2+3"),
            ((8, 4, "Additive 3+2bit (32 eff)"), "add_3+2"),
        ]:
            coarse, fine = additive_luts[add_name]
            ic, if_, am = quantize_additive_lut(W, coarse, fine, BS)
            recon = dequantize_additive_lut(ic, if_, am, coarse, fine, BS, orig_len)
            results[add_name]['psnr'].append(compute_psnr(W, recon))

    # ---- Print Results ----
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    # Sort by bit rate then PSNR
    sorted_methods = sorted(results.items(), key=lambda x: (x[1]['bits'], -np.mean([v for v in x[1]['psnr'] if v > 0])))

    print(f"\n{'Method':<38} {'bits/elem':>10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10}")
    print("-" * 90)

    for name, data in sorted_methods:
        valid = [v for v in data['psnr'] if v > 0]
        if valid:
            print(f"{name:<38} {data['bits']:>10.3f} "
                  f"{np.mean(valid):>10.2f} {np.min(valid):>10.2f} {np.max(valid):>10.2f}")

    # ---- Head-to-head at iso-bit ----
    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD AT ISO-BIT")
    print("=" * 100)

    uni8_mean = np.mean([v for v in results["Uniform 8-bit"]['psnr'] if v > 0])

    comparisons = [
        # (target_bit_group, [(name, ...)])
        ("5-bit (~5.125 bits/elem)", [
            "Uniform 6-bit", "LUT-64 (6-bit)",
            "Additive 2+3bit (32 eff)", "Additive 3+2bit (32 eff)",
        ]),
        ("6-bit (~6.125 bits/elem)", [
            "Uniform 7-bit", "LUT-128 (7-bit)",
            "Additive 3+3bit (64 eff)", "Additive 4+2bit (64 eff)",
        ]),
        ("8-bit (~8.125 bits/elem, baseline)", [
            "Uniform 8-bit", "LUT-256 (8-bit, BLOCKLUT baseline)",
            "Additive 4+3bit (128 eff)", "Additive 5+2bit (128 eff)",
        ]),
    ]

    for group_name, names in comparisons:
        print(f"\n--- {group_name} ---")
        best_psnr = 0
        best_name = ""
        for name in names:
            if name in results:
                psnr = np.mean([v for v in results[name]['psnr'] if v > 0])
                delta = psnr - uni8_mean
                marker = " ← BEST" if psnr > best_psnr else ""
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_name = name
                print(f"  {name:<35} {results[name]['bits']:.3f} b/e  {psnr:.2f} dB  (vs uni8: {delta:+.2f} dB){marker}")

    # ---- Analysis ----
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # Key question: can additive LUT beat single LUT at the same bit rate?
    print("\n  Key comparison — Additive vs Single LUT at same effective codebook size:")

    # 6-bit: 64 effective
    lut64 = np.mean([v for v in results["LUT-64 (6-bit)"]['psnr'] if v > 0])
    add33 = np.mean([v for v in results["Additive 3+3bit (64 eff)"]['psnr'] if v > 0])
    add42 = np.mean([v for v in results["Additive 4+2bit (64 eff)"]['psnr'] if v > 0])
    uni7 = np.mean([v for v in results["Uniform 7-bit"]['psnr'] if v > 0])
    print(f"  6-bit budget (64 effective values):")
    print(f"    Uniform 7-bit:                  {uni7:.2f} dB")
    print(f"    Single LUT-64:                  {lut64:.2f} dB")
    print(f"    Additive 3+3 (8×8=64):          {add33:.2f} dB (delta: {add33-lut64:+.2f} vs single)")
    print(f"    Additive 4+2 (16×4=64):         {add42:.2f} dB (delta: {add42-lut64:+.2f} vs single)")

    # 7-bit: 128 effective
    lut128 = np.mean([v for v in results["LUT-128 (7-bit)"]['psnr'] if v > 0])
    add43 = np.mean([v for v in results["Additive 4+3bit (128 eff)"]['psnr'] if v > 0])
    add52 = np.mean([v for v in results["Additive 5+2bit (128 eff)"]['psnr'] if v > 0])
    print(f"\n  7-bit budget (128 effective values):")
    print(f"    Single LUT-128:                 {lut128:.2f} dB")
    print(f"    Additive 4+3 (16×8=128):        {add43:.2f} dB (delta: {add43-lut128:+.2f} vs single)")
    print(f"    Additive 5+2 (32×4=128):        {add52:.2f} dB (delta: {add52-lut128:+.2f} vs single)")

    # 5-bit: 32 effective
    uni6 = np.mean([v for v in results["Uniform 6-bit"]['psnr'] if v > 0])
    add23 = np.mean([v for v in results["Additive 2+3bit (32 eff)"]['psnr'] if v > 0])
    add32 = np.mean([v for v in results["Additive 3+2bit (32 eff)"]['psnr'] if v > 0])
    print(f"\n  5-bit budget (32 effective values):")
    print(f"    Uniform 6-bit:                  {uni6:.2f} dB")
    print(f"    Additive 2+3 (4×8=32):          {add23:.2f} dB (delta: {add23-uni6:+.2f} vs uniform)")
    print(f"    Additive 3+2 (8×4=32):          {add32:.2f} dB (delta: {add32-uni6:+.2f} vs uniform)")

    # Render the verdict
    print(f"\n  Baseline BLOCKLUT-256 (8-bit): ~45 dB (per memory)")
    print(f"  省 1 bit 到 7-bit → PSNR 下降了 {uni8_mean-uni7:.1f} dB (uniform)")

    if add33 > lut64 + 0.5:
        print(f"\n  ✓ Additive LUT beats single LUT at 6-bit by {add33-lut64:.1f} dB — 值得继续探索!")
    else:
        print(f"\n  ✗ Additive LUT 在 6-bit 未能显著超越 single LUT (+{add33-lut64:.1f} dB)")
        print(f"    原因: 加性组合的 64 个有效值分布受限于 coarse+fine 的线性结构")
        print(f"    粗 LUT 和细 LUT 的质心范围限制了有效组合的多样性")

    print("\nDone.")


if __name__ == "__main__":
    main()
