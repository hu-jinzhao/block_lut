"""
Block-wise absmax + LUT 非均匀量化

改进思路:
旧 LUT: 全局 256-entry K-means → 31.3 dB (无局部自适应)
Block128 int8: 每 block absmax + 均匀量化 → 43.5 dB (局部自适应, 但量化级别均匀)
新方案: 每 block absmax + LUT 非均匀量化 → ? dB (局部自适应 + 非均匀级别)

实验:
1. 收集所有 block 的 absmax 归一化值
2. 对归一化值跑 K-means 得到 256-entry LUT
3. 测量 PSNR，对比均匀 int8

同时测试:
- Per-layer LUT (每层单独 codebook)
- 不同 LUT 大小 (128, 256, 512 entries)
"""

import os, math, sys, time
from collections import Counter
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from sklearn.cluster import KMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"

def collect_normalized_values(sft_files, block_size=128, max_samples_per_tensor=50000):
    """
    对所有 expert tensor 做 block128 absmax 归一化,
    收集归一化后的值用于 K-means 训练。

    采样策略: 每个 tensor 随机采样最多 max_samples_per_tensor 个归一化值
    """
    all_normalized = []

    for path in tqdm(sft_files, desc="Collecting normalized values"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                tensor = f.get_tensor(k).to(torch.float32).numpy().ravel()

                # Block-wise absmax normalization
                n = tensor.size
                bs = block_size
                nb = (n + bs - 1) // bs
                pad = nb * bs - n
                if pad > 0:
                    tensor = np.pad(tensor, (0, pad))

                # Sample blocks to avoid storing everything
                sample_blocks = min(nb, max(100, max_samples_per_tensor // bs))
                block_indices = np.random.choice(nb, sample_blocks, replace=False)

                for b in block_indices:
                    s, e = b * bs, (b+1) * bs
                    block = tensor[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    normalized = block / amax  # range [-1, 1]
                    all_normalized.append(normalized)

    return np.concatenate(all_normalized)


def collect_normalized_by_layer(sft_files, block_size=128):
    """Collect normalized values grouped by layer."""
    layer_values = {}  # layer -> [normalized values]

    for path in tqdm(sft_files, desc="Collecting by layer"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                # Extract layer
                parts = k.split(".")
                layer = int(parts[2])

                tensor = f.get_tensor(k).to(torch.float32).numpy().ravel()
                n = tensor.size
                bs = block_size
                nb = (n + bs - 1) // bs
                pad = nb * bs - n
                if pad > 0:
                    tensor = np.pad(tensor, (0, pad))

                sample_blocks = min(nb, 50)
                block_indices = np.random.choice(nb, sample_blocks, replace=False)

                vals = []
                for b in block_indices:
                    s, e = b * bs, (b+1) * bs
                    block = tensor[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    vals.append(block / amax)

                if vals:
                    if layer not in layer_values:
                        layer_values[layer] = []
                    layer_values[layer].append(np.concatenate(vals))

    return {l: np.concatenate(v) for l, v in layer_values.items()}


def build_lut(values, n_centroids, random_state=42):
    """K-means LUT on given values."""
    # Subsample for speed if too many values
    if len(values) > 500000:
        idx = np.random.choice(len(values), 500000, replace=False)
        values = values[idx]

    values = values.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=n_centroids, random_state=random_state,
                    n_init=3, max_iter=100, tol=1e-5)
    kmeans.fit(values)
    centroids = np.sort(kmeans.cluster_centers_.ravel())
    return centroids.astype(np.float32)


def quantize_with_lut(tensor_flat, lut, block_size=128):
    """Block-wise absmax + LUT quantization.

    For each block:
    1. Compute absmax
    2. Normalize by absmax
    3. Map each normalized value to nearest LUT centroid
    4. Store absmax (bf16) + LUT index (uint8 for 256 entries)
    """
    n = tensor_flat.size
    bs = block_size
    nb = (n + bs - 1) // bs
    pad = nb * bs - n
    if pad > 0:
        tensor_flat = np.pad(tensor_flat, (0, pad), mode='constant')

    indices = np.zeros(nb * bs, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)

    for b in range(nb):
        s, e = b * bs, (b+1) * bs
        block = tensor_flat[s:e]
        amax = np.max(np.abs(block))
        if amax < 1e-12:
            amax = 1e-12
        absmax_vals[b] = amax
        normalized = block / amax

        # Find nearest LUT centroid for each element
        # Vectorized: for each value, find index of nearest centroid
        idx = np.searchsorted(lut, normalized)
        idx = np.clip(idx, 1, len(lut) - 1)
        # Compare with left neighbor
        left_dist = np.abs(normalized - lut[idx - 1])
        right_dist = np.abs(normalized - lut[idx])
        idx = np.where(left_dist <= right_dist, idx - 1, idx)
        indices[s:e] = idx.astype(np.uint8)

    return indices, absmax_vals


def dequantize_with_lut(indices, absmax_vals, lut, block_size, orig_len):
    """Reverse of quantize_with_lut."""
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    nb = (n + block_size - 1) // block_size

    for b in range(nb):
        s, e = b * block_size, min((b+1) * block_size, n)
        idx = indices[s:e].astype(np.int32)
        normalized = lut[idx]
        x[s:e] = normalized * absmax_vals[b]

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

    print("=" * 100)
    print("BLOCK-WISE ABSMAX + LUT NON-UNIFORM QUANTIZATION")
    print("=" * 100)

    # Collect normalized values for LUT training
    print("\n[1/4] Collecting absmax-normalized values for LUT training...")
    np.random.seed(42)
    normalized_values = collect_normalized_values(sft_files, block_size=128)
    print(f"  Collected {len(normalized_values):,} normalized values")
    print(f"  Range: [{normalized_values.min():.4f}, {normalized_values.max():.4f}]")
    print(f"  Std: {normalized_values.std():.4f}")

    # Histogram of normalized values
    hist, bins = np.histogram(normalized_values, bins=50, range=(-1, 1))
    peak_bin = bins[hist.argmax()]
    near_zero_frac = np.mean(np.abs(normalized_values) < 0.1)
    print(f"  Near zero (|x|<0.1): {near_zero_frac*100:.1f}%")
    print(f"  Peak bin center: {peak_bin:.2f}")

    # Build LUTs of different sizes
    print("\n[2/4] Building LUTs...")
    lut_sizes = [128, 256, 512]
    luts = {}
    for n in lut_sizes:
        t0 = time.time()
        luts[n] = build_lut(normalized_values, n)
        bits = int(np.log2(n))
        print(f"  LUT-{n} ({bits}bit): {time.time()-t0:.1f}s, "
              f"range=[{luts[n][0]:.4f}, {luts[n][-1]:.4f}]")

    # Show centroid spacing near 0 vs tails
    lut256 = luts[256]
    # Spacing in middle 50%
    mid = lut256[64:192]
    tail = np.concatenate([lut256[:32], lut256[224:]])
    print(f"\n  LUT-256 centroid spacing:")
    print(f"    Middle 50% avg spacing: {np.mean(np.diff(mid)):.6f}")
    print(f"    Tail avg spacing: {np.mean(np.diff(tail)):.6f}")
    print(f"    Ratio tail/mid: {np.mean(np.diff(tail))/np.mean(np.diff(mid)):.2f}x")

    # Also build per-layer LUTs
    print("\n[2b/4] Building per-layer LUTs...")
    layer_values = collect_normalized_by_layer(sft_files)
    per_layer_luts = {}
    for layer, vals in sorted(layer_values.items()):
        per_layer_luts[layer] = build_lut(vals, 256)

    # Evaluate on sample tensors
    print("\n[3/4] Evaluating PSNR on sample tensors...")

    # Pick 200 tensors across layers
    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))

    sample_step = max(1, len(all_tensors) // 200)
    sampled = all_tensors[::sample_step][:200]
    print(f"  Evaluating {len(sampled)} tensors")

    bs = 128
    results = {
        'uniform_int8': {'psnr': [], 'bits': 8.0 + 16/bs},
        'lut128': {'psnr': [], 'bits': 7.0 + 16/bs},
        'lut256': {'psnr': [], 'bits': 8.0 + 16/bs},
        'lut512': {'psnr': [], 'bits': 9.0 + 16/bs},
        'per_layer_lut256': {'psnr': [], 'bits': 8.0 + 16/bs},
    }

    for name, tensor in tqdm(sampled, desc="  Evaluating"):
        W = tensor.to(torch.float32).numpy().ravel()
        orig_len = W.size
        parts = name.split(".")
        layer = int(parts[2])

        # Uniform int8
        indices, am = blockwise_quantize_uniform(W, bs, 8)
        recon = dequantize_uniform(indices, am, bs, 8, orig_len)
        results['uniform_int8']['psnr'].append(compute_psnr(W, recon))

        # LUT variants
        for lut_name, lut_size in [('lut128', 128), ('lut256', 256), ('lut512', 512)]:
            if lut_size == 512 and len(W) > 10000000:
                # Skip large tensors for 512 to save time
                results[lut_name]['psnr'].append(0)
                continue
            indices, am = quantize_with_lut(W, luts[lut_size], bs)
            recon = dequantize_with_lut(indices, am, luts[lut_size], bs, orig_len)
            results[lut_name]['psnr'].append(compute_psnr(W, recon))

        # Per-layer LUT
        indices, am = quantize_with_lut(W, per_layer_luts[layer], bs)
        recon = dequantize_with_lut(indices, am, per_layer_luts[layer], bs, orig_len)
        results['per_layer_lut256']['psnr'].append(compute_psnr(W, recon))

    # Summary
    print("\n[4/4] Results")
    print(f"\n{'Method':<25} {'bits/elem':>10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10} {'vs uniform':>12}")
    print("-" * 80)

    uni_mean = np.mean(results['uniform_int8']['psnr'])
    for name, data in results.items():
        valid = [x for x in data['psnr'] if x > 0]
        if valid:
            mean_p = np.mean(valid)
            min_p = np.min(valid)
            max_p = np.max(valid)
            delta = mean_p - uni_mean
            print(f"{name:<25} {data['bits']:>10.3f} {mean_p:>10.2f} {min_p:>10.2f} {max_p:>10.2f} {delta:>+11.2f}")

    # Best combo analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # At iso-quality: what bit width of LUT matches uniform int8 PSNR?
    uni_psnr = np.mean(results['uniform_int8']['psnr'])
    lut256_psnr = np.mean([x for x in results['lut256']['psnr'] if x > 0])
    lut128_psnr = np.mean([x for x in results['lut128']['psnr'] if x > 0])

    print(f"\n  Uniform int8 (8.125 bits): {uni_psnr:.2f} dB")
    print(f"  LUT-256 (8.125 bits):     {lut256_psnr:.2f} dB (delta: {lut256_psnr-uni_psnr:+.2f} dB)")
    print(f"  LUT-128 (7.125 bits):     {lut128_psnr:.2f} dB (delta: {lut128_psnr-uni_psnr:+.2f} dB)")
    print(f"  Per-layer LUT-256 (8.125): {np.mean(results['per_layer_lut256']['psnr']):.2f} dB")

    # Can we use LUT-128 (7-bit) to save 1 bit while maintaining PSNR close to uniform int8?
    if abs(lut128_psnr - uni_psnr) < 3:
        print(f"\n  → LUT-128 at 7.125 bits approaches uniform int8 PSNR, saving ~1 bit/elem!")
    elif lut256_psnr > uni_psnr:
        improvement = lut256_psnr - uni_psnr
        print(f"\n  → LUT-256 improves PSNR by {improvement:.1f} dB at same bit rate!")
    else:
        print(f"\n  → LUT does not significantly improve over uniform int8")
        print(f"  → The absmax-normalized distribution is too close to uniform for K-means to help")


# Reuse block-wise uniform functions for comparison
def blockwise_quantize_uniform(x, block_size, nbits):
    max_val = 2**(nbits-1) - 1
    n = x.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    indices = np.zeros(nb * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s, e = b*block_size, (b+1)*block_size
        block = x[s:e]
        amax = np.max(np.abs(block)) or 1e-12
        absmax_vals[b] = amax
        scale = amax / max_val
        q = np.clip(np.round(block/scale), -max_val-1, max_val).astype(np.int8)
        indices[s:e] = q.view(np.uint8)
    return indices, absmax_vals

def dequantize_uniform(indices, absmax_vals, block_size, nbits, orig_len):
    max_val = 2**(nbits-1) - 1
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    nb = (n + block_size - 1) // block_size
    for b in range(nb):
        s, e = b*block_size, min((b+1)*block_size, n)
        q = indices[s:e].view(np.int8).astype(np.float32)
        x[s:e] = q * absmax_vals[b] / max_val
    return x[:orig_len]


if __name__ == "__main__":
    main()
