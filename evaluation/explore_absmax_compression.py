"""
Absmax 压缩分析

Block-wise 量化中每个 block 存一个 bf16 absmax，占 16/128 = 0.125 bits/elem。
如果相邻 block 的 absmax 高度相关，可以用 delta 编码压缩。

分析:
1. absmax 值的分布特征
2. 相邻 block absmax 的相关性
3. Delta 编码后的熵
4. Huffman/LZ4HC 压缩 absmax 的 bits/elem 估算
"""

import os, math, sys
from collections import Counter
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
import lz4.block as lz4_block
import heapq

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"

# ---------------------------------------------------------------------------
def huffman_estimate(data: np.ndarray) -> float:
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    if total == 0:
        return 0
    heap = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, id(sym), [sym, f]))
    if len(heap) <= 1:
        return 1.0
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, id(n1), [n1, n2]))

    def get_lengths(node, depth=0, lengths=None):
        if lengths is None:
            lengths = {}
        if isinstance(node, list):
            if len(node) == 2 and not isinstance(node[0], list):
                lengths[node[0]] = max(depth, 1)
            else:
                get_lengths(node[0], depth + 1, lengths)
                get_lengths(node[1], depth + 1, lengths)
        return lengths
    lengths = get_lengths(heap[0][2])
    return sum(freq[sym] * lengths.get(sym, 1) for sym in freq) / total

def shannon_entropy(data: np.ndarray) -> float:
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    return -sum(c/total * math.log2(c/total) for c in freq.values())

# ---------------------------------------------------------------------------

def blockwise_absmax(tensor_2d: np.ndarray, block_size: int = 128) -> np.ndarray:
    """Extract absmax values for each block in a 2D tensor (row-major flatten)."""
    x = tensor_2d.ravel()
    n = x.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        x = np.pad(x, (0, pad), mode='constant', constant_values=0)

    absmax_vals = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s = b * block_size
        e = s + block_size
        amax = np.max(np.abs(x[s:e]))
        if amax == 0:
            amax = 1e-12
        absmax_vals[b] = amax
    return absmax_vals  # shape: (num_blocks,)

def analyze_absmax():
    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])

    # Load all expert tensors
    tensors = []
    for path in tqdm(sft_files, desc="Loading"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    tensors.append((k, f.get_tensor(k)))

    print(f"Loaded {len(tensors)} expert tensors")

    # Sample 100 tensors across all layers
    step = max(1, len(tensors) // 100)
    sampled = tensors[::step][:100]
    print(f"Sampled {len(sampled)} tensors\n")

    block_sizes = [64, 128, 256]

    print("=" * 100)
    print("ABSMAX COMPRESSION ANALYSIS")
    print("=" * 100)

    for bs in block_sizes:
        print(f"\n--- Block size = {bs} ---")
        print(f"  Baseline absmax cost: 16/{bs} = {16/bs:.4f} bits/elem")

        all_absmax = []
        all_log_absmax = []
        absmax_stats = []

        for name, tensor in tqdm(sampled, desc=f"  BS={bs}"):
            x = tensor.to(torch.float32).numpy()
            am = blockwise_absmax(x, bs)

            # Basic stats
            log_am = np.log2(am + 1e-30)

            # Neighbor correlation (1D flattened view)
            if len(am) > 1:
                # Correlation with lag-1 neighbor
                corr = np.corrcoef(am[:-1], am[1:])[0, 1]
                # Delta between consecutive absmax
                delta = np.diff(am)
                # Relative delta
                rel_delta = delta / (am[:-1] + 1e-12)

                absmax_stats.append({
                    'corr': corr,
                    'am_mean': np.mean(am),
                    'am_std': np.std(am),
                    'delta_std': np.std(delta),
                    'delta_range': np.max(np.abs(delta)),
                    'log_std': np.std(log_am),
                    'rel_delta_std': np.std(rel_delta),
                })

            all_absmax.append(am)
            all_log_absmax.append(log_am)

        # Aggregate stats
        corrs = [s['corr'] for s in absmax_stats]
        delta_stds = [s['delta_std'] for s in absmax_stats]
        am_means = [s['am_mean'] for s in absmax_stats]
        am_stds = [s['am_std'] for s in absmax_stats]
        log_stds = [s['log_std'] for s in absmax_stats]
        rel_delta_stds = [s['rel_delta_std'] for s in absmax_stats]

        print(f"\n  Neighbor correlation (absmax[t] vs absmax[t+1]):")
        print(f"    Mean corr: {np.mean(corrs):.4f}, Median: {np.median(corrs):.4f}")
        print(f"    Min: {np.min(corrs):.4f}, Max: {np.max(corrs):.4f}")

        print(f"\n  Absmax value statistics:")
        print(f"    Mean absmax: {np.mean(am_means):.6f}, Std: {np.mean(am_stds):.6f}")
        print(f"    Log-absmax std: {np.mean(log_stds):.4f} (in log2 space)")

        print(f"\n  Delta statistics (absmax[t+1] - absmax[t]):")
        print(f"    Delta std: {np.mean(delta_stds):.6f}")
        print(f"    Relative delta std: {np.mean(rel_delta_stds):.4f}")
        ratio = np.mean(delta_stds) / (np.mean(am_stds) + 1e-12)
        print(f"    Delta std / Absmax std ratio: {ratio:.4f}")

        # --- Quantization analysis ---
        # Strategy 1: Store absmax as fp16/bf16 → 16 bits per block
        # Strategy 2: Quantize absmax to int8 (uniform over range)
        # Strategy 3: Quantize absmax in log space to int8
        # Strategy 4: Delta encoding: store first absmax as fp16, then quantize deltas

        # Strategy 2: int8 uniform quantization of absmax
        all_am_concat = np.concatenate(all_absmax)
        am_range = [np.min(all_am_concat), np.max(all_am_concat)]
        am_scale = (am_range[1] - am_range[0]) / 255
        am_quant = np.clip(np.round((all_am_concat - am_range[0]) / am_scale), 0, 255).astype(np.uint8)
        am_huff = huffman_estimate(am_quant)
        am_ent = shannon_entropy(am_quant)
        print(f"\n  Strategy 2: int8 uniform absmax → Huffman:")
        print(f"    Range: [{am_range[0]:.4f}, {am_range[1]:.4f}]")
        print(f"    Entropy: {am_ent:.3f} bits, Huffman: {am_huff:.3f} bits per absmax")
        print(f"    Cost: {am_huff/bs:.4f} bits/elem")

        # Strategy 3: int8 in log space
        log_am_concat = np.concatenate(all_log_absmax)
        log_range = [np.min(log_am_concat), np.max(log_am_concat)]
        log_scale = (log_range[1] - log_range[0]) / 255
        log_am_quant = np.clip(np.round((log_am_concat - log_range[0]) / log_scale), 0, 255).astype(np.uint8)
        log_huff = huffman_estimate(log_am_quant)
        log_ent = shannon_entropy(log_am_quant)
        print(f"\n  Strategy 3: int8 log-absmax → Huffman:")
        print(f"    Log range: [{log_range[0]:.2f}, {log_range[1]:.2f}]")
        print(f"    Log std: {np.mean(log_stds):.4f}")
        print(f"    Entropy: {log_ent:.3f} bits, Huffman: {log_huff:.3f} bits per absmax")
        print(f"    Cost: {log_huff/bs:.4f} bits/elem")

        # Strategy 4: Delta encoding of absmax
        # For each tensor: store first absmax as bf16, then quantize deltas
        all_deltas = []
        first_vals = []
        for am in all_absmax:
            if len(am) > 1:
                first_vals.append(am[0])
                deltas = np.diff(am)
                all_deltas.append(deltas)

        all_deltas_concat = np.concatenate(all_deltas)
        delta_range = [np.min(all_deltas_concat), np.max(all_deltas_concat)]
        delta_scale = (delta_range[1] - delta_range[0]) / 255
        delta_quant = np.clip(np.round((all_deltas_concat - delta_range[0]) / delta_scale), 0, 255).astype(np.uint8)
        delta_huff = huffman_estimate(delta_quant)
        delta_ent = shannon_entropy(delta_quant)

        # Delta in log space
        all_log_deltas = []
        for log_am in all_log_absmax:
            if len(log_am) > 1:
                all_log_deltas.append(np.diff(log_am))
        log_deltas_concat = np.concatenate(all_log_deltas)
        log_delta_range = [np.min(log_deltas_concat), np.max(log_deltas_concat)]
        log_delta_scale = (log_delta_range[1] - log_delta_range[0]) / 255
        log_delta_quant = np.clip(np.round((log_deltas_concat - log_delta_range[0]) / log_delta_scale), 0, 255).astype(np.uint8)
        log_delta_huff = huffman_estimate(log_delta_quant)
        log_delta_ent = shannon_entropy(log_delta_quant)

        print(f"\n  Strategy 4a: Delta absmax (int8) → Huffman:")
        print(f"    Delta range: [{delta_range[0]:.6f}, {delta_range[1]:.6f}]")
        print(f"    Entropy: {delta_ent:.3f} bits, Huffman: {delta_huff:.3f} bits per delta")
        # Overhead: bf16 first value per tensor (negligible for large tensors)
        first_overhead = 16 / np.mean([am.size * bs for am in all_absmax])
        print(f"    Cost: {delta_huff/bs:.4f} bits/elem + first-bf16 overhead {first_overhead:.6f} bits/elem")
        print(f"    Total: {delta_huff/bs + first_overhead:.4f} bits/elem")

        print(f"\n  Strategy 4b: Delta log-absmax (int8) → Huffman:")
        print(f"    Log-delta range: [{log_delta_range[0]:.4f}, {log_delta_range[1]:.4f}]")
        print(f"    Entropy: {log_delta_ent:.3f} bits, Huffman: {log_delta_huff:.3f} bits per delta")
        print(f"    Cost: {log_delta_huff/bs:.4f} bits/elem + first-bf16 overhead {first_overhead:.6f} bits/elem")
        print(f"    Total: {log_delta_huff/bs + first_overhead:.4f} bits/elem")

        # Strategy 5: LZ4HC on raw absmax bytes
        all_am_bytes = np.concatenate(all_absmax).astype(np.float16).tobytes()
        lz4hc_compressed = lz4_block.compress(all_am_bytes, mode='high_compression', compression=9)
        lz4hc_bits_per_am = len(lz4hc_compressed) * 8 / len(all_am_concat)
        print(f"\n  Strategy 5: LZ4HC on fp16 absmax stream:")
        print(f"    Original: {len(all_am_bytes)*8} bits → Compressed: {len(lz4hc_compressed)*8} bits")
        print(f"    LZ4HC: {lz4hc_bits_per_am:.3f} bits per absmax")
        print(f"    Cost: {lz4hc_bits_per_am/bs:.4f} bits/elem")

        # Strategy 6: LZ4HC on delta bytes
        all_delta_bytes = all_deltas_concat.astype(np.float16).tobytes()
        lz4hc_delta_comp = lz4_block.compress(all_delta_bytes, mode='high_compression', compression=9)
        lz4hc_delta_bits = len(lz4hc_delta_comp) * 8 / len(all_deltas_concat)
        print(f"\n  Strategy 6: LZ4HC on fp16 delta stream:")
        print(f"    LZ4HC: {lz4hc_delta_bits:.3f} bits per delta")
        print(f"    Cost: {lz4hc_delta_bits/bs:.4f} bits/elem + first-bf16 overhead")

    # --- Summary table ---
    print("\n" + "=" * 100)
    print("SUMMARY: Absmax compression strategies (block=128)")
    print("=" * 100)

    # Recompute for bs=128 specifically
    bs = 128
    baseline = 16 / bs
    print(f"\n  Baseline (bf16 per block):          {baseline:.4f} bits/elem")
    print(f"  Int8 Huffman absmax:                {am_huff/bs:.4f} bits/elem (saves {(baseline-am_huff/bs):.4f})")
    print(f"  Int8 Huffman log-absmax:            {log_huff/bs:.4f} bits/elem (saves {(baseline-log_huff/bs):.4f})")
    print(f"  Int8 Delta absmax + Huffman:        {delta_huff/bs + first_overhead:.4f} bits/elem (saves {(baseline-delta_huff/bs-first_overhead):.4f})")
    print(f"  Int8 Delta log-absmax + Huffman:    {log_delta_huff/bs + first_overhead:.4f} bits/elem (saves {(baseline-log_delta_huff/bs-first_overhead):.4f})")
    print(f"  LZ4HC fp16 absmax stream:           {lz4hc_bits_per_am/bs:.4f} bits/elem (saves {(baseline-lz4hc_bits_per_am/bs):.4f})")
    print(f"  LZ4HC fp16 delta stream + first:    {lz4hc_delta_bits/bs + first_overhead:.4f} bits/elem (saves {(baseline-lz4hc_delta_bits/bs-first_overhead):.4f})")

    # Also test: what if we just store absmax at int8 precision (losing some quality)?
    print(f"\n  --- Impact of absmax quantization on weight PSNR ---")
    # For one sample tensor, compare: bf16 absmax vs int8 absmax
    name, tensor = sampled[0]
    x = tensor.to(torch.float32).numpy().ravel()
    n = x.size
    bs = 128
    nb = (n + bs - 1) // bs
    pad = nb * bs - n
    if pad > 0:
        x_pad = np.pad(x, (0, pad))
    else:
        x_pad = x

    # bf16 absmax reference
    am_bf16 = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s, e = b * bs, min((b+1)*bs, n)
        am_bf16[b] = max(np.max(np.abs(x_pad[s:e])), 1e-12)

    # int8 uniform quantized absmax
    am_min, am_max = np.min(am_bf16), np.max(am_bf16)
    am_q_scale = (am_max - am_min) / 255
    am_q_idx = np.clip(np.round((am_bf16 - am_min) / am_q_scale), 0, 255).astype(np.uint8)
    am_q = am_min + am_q_idx.astype(np.float32) * am_q_scale

    # Reconstruct with both absmax versions
    rec_bf16 = np.zeros_like(x_pad)
    rec_int8_am = np.zeros_like(x_pad)
    for b in range(nb):
        s, e = b * bs, min((b+1)*bs, n)
        block = x_pad[s:e]
        # bf16 absmax
        scale_b = am_bf16[b] / 127.5
        q_b = np.clip(np.round(block / scale_b), -128, 127)
        rec_bf16[s:e] = q_b * scale_b
        # int8 absmax
        scale_i = am_q[b] / 127.5
        q_i = np.clip(np.round(block / scale_i), -128, 127)
        rec_int8_am[s:e] = q_i * scale_i

    mse_bf16 = np.mean((x_pad[:n] - rec_bf16[:n])**2)
    mse_int8 = np.mean((x_pad[:n] - rec_int8_am[:n])**2)
    var_orig = np.var(x_pad[:n])
    psnr_bf16 = 10 * math.log10(var_orig / mse_bf16) if mse_bf16 > 0 else float('inf')
    psnr_int8 = 10 * math.log10(var_orig / mse_int8) if mse_int8 > 0 else float('inf')

    print(f"  bf16 absmax → block128 int8 weights: PSNR={psnr_bf16:.2f} dB")
    print(f"  int8 absmax → block128 int8 weights: PSNR={psnr_int8:.2f} dB")
    print(f"  PSNR loss from absmax quantization:  {psnr_bf16-psnr_int8:.2f} dB")

    # Also test int8 log-space absmax
    log_am = np.log2(am_bf16 + 1e-30)
    log_min, log_max = np.min(log_am), np.max(log_am)
    log_q_scale = (log_max - log_min) / 255
    log_q_idx = np.clip(np.round((log_am - log_min) / log_q_scale), 0, 255).astype(np.uint8)
    log_q = 2.0 ** (log_min + log_q_idx.astype(np.float32) * log_q_scale)

    rec_log_am = np.zeros_like(x_pad)
    for b in range(nb):
        s, e = b * bs, min((b+1)*bs, n)
        block = x_pad[s:e]
        scale_l = log_q[b] / 127.5
        q_l = np.clip(np.round(block / scale_l), -128, 127)
        rec_log_am[s:e] = q_l * scale_l

    mse_log = np.mean((x_pad[:n] - rec_log_am[:n])**2)
    psnr_log = 10 * math.log10(var_orig / mse_log) if mse_log > 0 else float('inf')
    print(f"  int8 log-absmax → block128 int8 weights: PSNR={psnr_log:.2f} dB")
    print(f"  PSNR loss from log-absmax quantization:  {psnr_bf16-psnr_log:.2f} dB")

    # --- Final recommendation ---
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    best_bits = min(
        (am_huff/bs, "int8 Huffman absmax"),
        (log_huff/bs, "int8 Huffman log-absmax"),
        (delta_huff/bs + first_overhead, "int8 delta + Huffman"),
        (log_delta_huff/bs + first_overhead, "int8 delta log + Huffman"),
        (lz4hc_bits_per_am/bs, "LZ4HC fp16 absmax"),
        (lz4hc_delta_bits/bs + first_overhead, "LZ4HC delta")
    )
    savings = baseline - best_bits[0]
    print(f"\n  Best: {best_bits[1]} at {best_bits[0]:.4f} bits/elem")
    print(f"  Saving vs baseline bf16: {savings:.4f} bits/elem ({savings/baseline*100:.1f}%)")
    print(f"  Combined with block128 int8 (8.0 bits): {8.0 + best_bits[0]:.4f} bits/elem total")

if __name__ == "__main__":
    analyze_absmax()
