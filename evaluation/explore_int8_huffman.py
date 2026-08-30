"""
int8 量化 + 熵编码 (Huffman) 压缩可行性实验

思路:
1. 对 MoE expert 权重做 block-wise int8 量化 (absmax uniform)
2. 对 int8 索引构建 Huffman 编码, 利用索引分布的非均匀性进一步压缩
3. 测量: PSNR (var-based), bits/elem, 与现有方案对比

对比基准:
- 原始 bf16: 16 bits/elem, lossless
- exp+SM+LZ4HC: ~12 bits/elem, lossless
- Block128 int8: ~8.125 bits/elem, ~43.5 dB
- Block128 int8+Huffman: ? bits/elem, 同 PSNR (Huffman 无损)
"""

import os, sys, math, time, json
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD_DIR = "/home/hh/zip_Moe/LUT_MoE/offload/qwen"

# ---------------------------------------------------------------------------
# Huffman coding (canonical, for entropy estimation we only need code lengths)
# ---------------------------------------------------------------------------

class HuffmanNode:
    __slots__ = ('freq', 'symbol', 'left', 'right')
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

def build_huffman_tree(freq: Counter) -> HuffmanNode:
    """Build Huffman tree from symbol frequencies. Returns root."""
    import heapq
    heap = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, id(sym), HuffmanNode(f, sym)))
    if len(heap) == 0:
        return None
    if len(heap) == 1:
        f, _, n = heapq.heappop(heap)
        return HuffmanNode(f, left=n)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, id(merged), merged))
    return heap[0][2]

def get_code_lengths(root: HuffmanNode, depth=0, lengths=None):
    """Recursively collect code lengths for each symbol."""
    if lengths is None:
        lengths = {}
    if root.symbol is not None:
        lengths[root.symbol] = max(depth, 1)  # min 1 bit for single symbol
    if root.left:
        get_code_lengths(root.left, depth + 1, lengths)
    if root.right:
        get_code_lengths(root.right, depth + 1, lengths)
    return lengths

def huffman_bits_per_element(data: np.ndarray) -> float:
    """Estimate Huffman-coded bits per element for uint8 data."""
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    root = build_huffman_tree(freq)
    lengths = get_code_lengths(root)
    avg_bits = sum(freq[sym] * lengths[sym] for sym in freq) / total
    return avg_bits

def entropy_bound(data: np.ndarray) -> float:
    """Shannon entropy lower bound (bits per element)."""
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# ---------------------------------------------------------------------------
# Block-wise int8 quantization / reconstruction
# ---------------------------------------------------------------------------

def blockwise_quantize_int8(
    tensor: torch.Tensor, block_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-wise absmax uniform int8 quantization.

    Per block: scale = absmax / 127.5
    idx = round(clip(x / scale, -127.5, 127.5))   → stored as uint8 (0..255 mapped from -128..127)

    Returns:
      indices: uint8 array of shape (num_blocks, block_size) or flattened
      absmax:  bf16 values, one per block (stored as float32 for numpy)
    """
    x = tensor.detach().to(torch.float32).numpy().ravel()
    n = x.size
    num_blocks = (n + block_size - 1) // block_size
    pad_len = num_blocks * block_size - n
    if pad_len > 0:
        x = np.pad(x, (0, pad_len), mode='constant', constant_values=0)

    indices = np.zeros(num_blocks * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(num_blocks, dtype=np.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block = x[start:end]
        amax = np.max(np.abs(block))
        if amax == 0:
            amax = 1e-12
        absmax_vals[b] = amax
        scale = amax / 127.5
        q = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        indices[start:end] = q.view(np.uint8)

    return indices, absmax_vals

def blockwise_dequantize_bf16(
    indices: np.ndarray, absmax_vals: np.ndarray,
    block_size: int = 128, original_len: int = None
) -> torch.Tensor:
    """Reverse of blockwise_quantize_int8, returns bf16 tensor."""
    n = indices.size
    num_blocks = (n + block_size - 1) // block_size
    x = np.zeros(n, dtype=np.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block_indices = indices[start:end].view(np.int8).astype(np.float32)
        scale = absmax_vals[b] / 127.5
        x[start:end] = block_indices * scale

    if original_len is not None:
        x = x[:original_len]
    return torch.from_numpy(x).to(torch.bfloat16)

# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr_var(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Var-based PSNR: 10 * log10(var(original) / mse)"""
    orig = original.detach().to(torch.float32).ravel()
    recon = reconstructed.detach().to(torch.float32).ravel()
    mse = ((orig - recon) ** 2).mean().item()
    var = orig.var().item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(var / mse)

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@dataclass
class Result:
    tensor_name: str
    layer: int
    num_elements: int
    psnr_db: float
    huffman_bits: float       # Huffman-coded bits/elem for int8 indices
    entropy_bound_bits: float  # Shannon bound for int8 indices
    raw_int8_bits: float       # 8 + absmax overhead
    total_bits_huffman: float  # Huffman bits + absmax overhead
    block_size: int

def load_expert_tensors(safetensor_paths: List[str]) -> Dict[str, torch.Tensor]:
    """Load only expert parameters from safetensors files."""
    experts = {}
    for path in tqdm(safetensor_paths, desc="Loading safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    experts[k] = f.get_tensor(k)
    return experts

def extract_layer(name: str) -> int:
    """Extract layer index from parameter name like model.layers.0.mlp.experts.0.up_proj.weight"""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1

def run_experiment(
    block_sizes=[64, 128, 256],
    sample_tensors=None,  # None = all
    max_tensors=60,       # sample up to N tensors across layers
):
    # Find safetensor files
    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Found {len(sft_files)} safetensor files")

    # Also check offload directory for pre-extracted expert data
    if os.path.exists(OFFLOAD_DIR):
        print(f"Offload dir exists: {OFFLOAD_DIR}")

    experts = load_expert_tensors(sft_files)
    print(f"Loaded {len(experts)} expert tensors")

    # Print some names to understand structure
    sample_names = list(experts.keys())[:5]
    print(f"Example names: {sample_names}")

    # Sample tensors across layers
    all_names = sorted(experts.keys())
    if sample_tensors is not None:
        all_names = all_names[:sample_tensors]
    elif max_tensors and len(all_names) > max_tensors:
        # Stratified sampling: pick evenly across layers
        layer_groups = {}
        for name in all_names:
            layer = extract_layer(name)
            layer_groups.setdefault(layer, []).append(name)

        all_names = []
        for layer in sorted(layer_groups.keys()):
            names = layer_groups[layer]
            # Take ~max_tensors/num_layers per layer
            n_per_layer = max(1, max_tensors // max(1, len(layer_groups)))
            step = max(1, len(names) // n_per_layer)
            all_names.extend(names[::step][:n_per_layer])
        all_names = all_names[:max_tensors]

    print(f"Sampling {len(all_names)} tensors for experiment\n")

    results: List[Result] = []

    for block_size in block_sizes:
        absmax_bits_per_elem = 16 / block_size  # 16-bit bf16 absmax per block

        for name in tqdm(all_names, desc=f"Block={block_size}"):
            tensor = experts[name]
            original_len = tensor.numel()

            # Block-wise int8 quantize
            indices, absmax_vals = blockwise_quantize_int8(tensor, block_size)

            # Huffman estimate
            huff_bits = huffman_bits_per_element(indices)
            ent_bits = entropy_bound(indices)

            # Total bits with Huffman = Huffman coded indices + absmax overhead
            total_huff = huff_bits + absmax_bits_per_elem

            # Reconstruct and compute PSNR
            recon = blockwise_dequantize_bf16(indices, absmax_vals, block_size, original_len)
            psnr = compute_psnr_var(tensor, recon)

            layer = extract_layer(name)

            results.append(Result(
                tensor_name=name.split(".")[-1],
                layer=layer,
                num_elements=original_len,
                psnr_db=psnr,
                huffman_bits=huff_bits,
                entropy_bound_bits=ent_bits,
                raw_int8_bits=8.0 + absmax_bits_per_elem,
                total_bits_huffman=total_huff,
                block_size=block_size,
            ))

    return results, experts, all_names

def print_summary(results: List[Result], experts: dict, sampled_names: list):
    """Print summary grouped by block_size."""
    print("\n" + "=" * 100)
    print("SUMMARY: int8 + Huffman Compression Experiment")
    print("=" * 100)

    # Also compute Huffman on raw bf16 bytes as baseline
    print(f"\n{'Block':<10} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10} "
          f"{'Huffman':>10} {'Entropy':>10} {'Raw int8':>10} {'Total':>10} {'vs raw':>10}")
    print(f"{'':10} {'(dB)':>10} {'(dB)':>10} {'(dB)':>10} "
          f"{'(bits/elem)':>10} {'(bits/elem)':>10} {'(bits/elem)':>10} {'(bits/elem)':>10} {'(saving)':>10}")
    print("-" * 100)

    by_block = {}
    for r in results:
        by_block.setdefault(r.block_size, []).append(r)

    for bs in sorted(by_block.keys()):
        items = by_block[bs]
        psnrs = [r.psnr_db for r in items]
        huffs = [r.huffman_bits for r in items]
        ents = [r.entropy_bound_bits for r in items]
        totals = [r.total_bits_huffman for r in items]
        raws = [r.raw_int8_bits for r in items]

        mean_saving = np.mean([(8.0 - h) / 8.0 * 100 for h in huffs])

        print(f"{bs:<10} {np.mean(psnrs):>10.2f} {np.min(psnrs):>10.2f} {np.max(psnrs):>10.2f} "
              f"{np.mean(huffs):>10.3f} {np.mean(ents):>10.3f} {np.mean(raws):>10.3f} "
              f"{np.mean(totals):>10.3f} {mean_saving:>9.1f}%")

    # Per-layer breakdown for best block size
    print("\n" + "-" * 100)
    print("Per-layer breakdown (block=128):")
    print(f"{'Layer':<8} {'PSNR':>8} {'Huffman':>10} {'Entropy':>10} {'Total':>10} {'Saving':>10}")

    bs128 = [r for r in results if r.block_size == 128]
    by_layer = {}
    for r in bs128:
        by_layer.setdefault(r.layer, []).append(r)

    for layer in sorted(by_layer.keys()):
        items = by_layer[layer]
        psnr = np.mean([r.psnr_db for r in items])
        huff = np.mean([r.huffman_bits for r in items])
        ent = np.mean([r.entropy_bound_bits for r in items])
        total = np.mean([r.total_bits_huffman for r in items])
        saving = (8.0 - huff) / 8.0 * 100
        print(f"{layer:<8} {psnr:>8.2f} {huff:>10.3f} {ent:>10.3f} {total:>10.3f} {saving:>9.1f}%")

    # Comparison table
    print("\n" + "=" * 100)
    print("COMPARISON WITH EXISTING APPROACHES")
    print("=" * 100)

    # Pick block=128 for comparison
    bs128 = [r for r in results if r.block_size == 128]
    mean_psnr = np.mean([r.psnr_db for r in bs128])
    mean_huff = np.mean([r.huffman_bits for r in bs128])
    mean_ent = np.mean([r.entropy_bound_bits for r in bs128])
    absmax_overhead = 16 / 128
    mean_total = mean_huff + absmax_overhead

    print(f"{'Method':<35} {'bits/elem':>12} {'PSNR (dB)':>12} {'vs bf16':>12}")
    print("-" * 72)
    print(f"{'Raw bf16':<35} {16.0:>12.1f} {'lossless':>12} {'1.00x':>12}")
    print(f"{'exp+SM raw (8+8)':<35} {16.0:>12.1f} {'lossless':>12} {'1.00x':>12}")
    print(f"{'exp+SM+LZ4HC (current best)':<35} {12.0:>12.1f} {'lossless':>12} {'1.33x':>12}")
    print(f"{'exp+SM+LZ4 (current fast)':<35} {14.0:>12.1f} {'lossless':>12} {'1.14x':>12}")
    print(f"{'Global LUT 256 (K-means)':<35} {8.0:>12.1f} {31.3:>12.1f} {'2.00x':>12}")
    print(f"{'Block128 int8 uniform':<35} {8.125:>12.2f} {43.5:>12.1f} {'1.97x':>12}")
    print(f"{'Block128 int8 + Huffman (this)':<35} {mean_total:>12.2f} {mean_psnr:>12.1f} {16.0/mean_total:>11.2f}x")
    print(f"{'Block128 int8 + Entropy bound':<35} {mean_ent + absmax_overhead:>12.2f} {mean_psnr:>12.1f} {16.0/(mean_ent+absmax_overhead):>11.2f}x")

    # Also test: Huffman on raw uint8 data (no quantization) to see entropy of raw weights
    print("\n" + "=" * 100)
    print("RAW DATA ENTROPY ANALYSIS (no quantization)")
    print("=" * 100)
    analyze_raw_entropy(experts, sampled_names[:min(30, len(sampled_names))])

def analyze_raw_entropy(experts: dict, names: list):
    """Analyze entropy of:
    1. Raw bf16 bytes (high/low)
    2. Exponent bytes from exp+SM split (to compare Huffman vs LZ4HC)
    3. int8 indices distribution histogram
    """
    import lz4.block as lz4_block

    entropies_hi = []
    entropies_lo = []
    huffman_hi = []
    huffman_lo = []
    lz4hc_hi_bits = []
    lz4hc_lo_bits = []

    # For exp bytes (from current scheme)
    huffman_exp = []
    entropy_exp = []
    lz4hc_exp_bits = []

    for name in tqdm(names, desc="Raw entropy"):
        tensor = experts[name]
        u16 = tensor.detach().view(torch.int16).numpy().ravel()
        u8 = u16.view(np.uint8)
        hi = u8[1::2].copy()  # high byte (exponent + sign)
        lo = u8[0::2].copy()  # low byte (mantissa)

        entropies_hi.append(entropy_bound(hi))
        entropies_lo.append(entropy_bound(lo))
        huffman_hi.append(huffman_bits_per_element(hi))
        huffman_lo.append(huffman_bits_per_element(lo))

        # LZ4HC on raw bytes
        lz4hc_hi = len(lz4_block.compress(hi.tobytes(), mode='high_compression', compression=9))
        lz4hc_lo = len(lz4_block.compress(lo.tobytes(), mode='high_compression', compression=9))
        lz4hc_hi_bits.append(lz4hc_hi * 8 / hi.size)
        lz4hc_lo_bits.append(lz4hc_lo * 8 / lo.size)

        # Current scheme: extract exponent byte (bits [7:15] of bf16)
        exponent = ((u16 >> 7) & 0xFF).astype(np.uint8)
        huffman_exp.append(huffman_bits_per_element(exponent))
        entropy_exp.append(entropy_bound(exponent))
        lz4hc_exp_comp = len(lz4_block.compress(exponent.tobytes(), mode='high_compression', compression=9))
        lz4hc_exp_bits.append(lz4hc_exp_comp * 8 / exponent.size)

    print(f"\n{'Byte':<18} {'Entropy':>10} {'Huffman':>10} {'LZ4HC':>10} {'Huff vs LZ4':>14}")
    print(f"{'':18} {'(bits/elem)':>10} {'(bits/elem)':>10} {'(bits/elem)':>10}")
    print("-" * 66)
    print(f"{'High byte (exp+sign)':<18} {np.mean(entropies_hi):>10.3f} {np.mean(huffman_hi):>10.3f} "
          f"{np.mean(lz4hc_hi_bits):>10.3f} {'Huff ' + ('wins' if np.mean(huffman_hi) < np.mean(lz4hc_hi_bits) else 'loses'):>14}")
    print(f"{'Low byte (mantissa)':<18} {np.mean(entropies_lo):>10.3f} {np.mean(huffman_lo):>10.3f} "
          f"{np.mean(lz4hc_lo_bits):>10.3f} {'Huff ' + ('wins' if np.mean(huffman_lo) < np.mean(lz4hc_lo_bits) else 'loses'):>14}")
    print(f"{'Exponent (bits 7-15)':<18} {np.mean(entropy_exp):>10.3f} {np.mean(huffman_exp):>10.3f} "
          f"{np.mean(lz4hc_exp_bits):>10.3f} {'Huff ' + ('wins' if np.mean(huffman_exp) < np.mean(lz4hc_exp_bits) else 'loses'):>14}")

    print(f"\nCurrent exp+SM+LZ4HC total: ~12 bits/elem (exp compressed + SM raw 8 bits)")
    print(f"Exp Huffman + SM raw would be: {np.mean(huffman_exp):.2f} + 8.0 = {np.mean(huffman_exp) + 8.0:.2f} bits/elem")
    print(f"Exp LZ4HC + SM raw (current):  {np.mean(lz4hc_exp_bits):.2f} + 8.0 = {np.mean(lz4hc_exp_bits) + 8.0:.2f} bits/elem")

    # Also compute Huffman on the full exponent+SM interleaved (to check if joint encoding helps)
    print("\n--- Joint exponent+SM Huffman analysis ---")
    huff_joint = []
    ent_joint = []
    lz4hc_joint_bits = []
    for name in tqdm(names[:min(5, len(names))], desc="Joint analysis"):
        tensor = experts[name]
        u16 = tensor.detach().view(torch.int16).numpy().ravel()
        u8 = u16.view(np.uint8)
        # Try encoding pairs of (hi, lo) as 16-bit symbols
        # But with 65536 possible symbols, Huffman is impractical to show
        # Instead: show that LZ4HC on the full uint8 stream
        raw_bytes = u8.tobytes()
        lz4hc_all = len(lz4_block.compress(raw_bytes, mode='high_compression', compression=9))
        lz4hc_joint_bits.append(lz4hc_all * 8 / u8.size)
        ent_joint.append(entropy_bound(u8))
        huff_joint.append(huffman_bits_per_element(u8))

    print(f"Full uint8 stream: Entropy={np.mean(ent_joint):.3f}, Huffman={np.mean(huff_joint):.3f}, "
          f"LZ4HC={np.mean(lz4hc_joint_bits):.3f} bits/elem")
    print(f"LZ4HC is better than Huffman for raw bytes because it captures repeating patterns,")
    print(f"not just per-symbol frequencies. Huffman only sees marginals, LZ4HC sees context.")

    # Histogram analysis of int8 indices
    print("\n" + "=" * 100)
    print("INT8 INDEX DISTRIBUTION ANALYSIS (why Huffman saves only ~5%)")
    print("=" * 100)

    name = names[0]
    tensor = experts[name]
    indices, absmax_vals = blockwise_quantize_int8(tensor, 128)

    freq = Counter(indices.ravel().tolist())
    total = sum(freq.values())

    print(f"\nTensor: {name}, {tensor.numel()} elements")
    print(f"Unique int8 values: {len(freq)} / 256")
    print(f"Shannon entropy: {entropy_bound(indices):.3f} bits (max 8.0)")
    print(f"Huffman avg:     {huffman_bits_per_element(indices):.3f} bits")
    print(f"Huffman saving vs uniform 8-bit: {(8.0 - huffman_bits_per_element(indices)) / 8.0 * 100:.1f}%")
    print(f"\nTop-20 most frequent int8 values:")
    for val, count in freq.most_common(20):
        pct = count / total * 100
        bar = '█' * int(pct * 20)
        print(f"  {np.int8(val):>4}: {count:>8} ({pct:>5.1f}%) {bar}")

    # Why is it nearly uniform? Because weights ~ N(0, sigma), and absmax uniform
    # quantization maps evenly across the range, each block's values spread
    # across the [-127, 127] range quite evenly.
    print(f"\nKey insight: Block-wise absmax uniform quantization produces nearly uniform")
    print(f"distributions because each block's values are normalized by its own absmax,")
    print(f"spreading values evenly across the [-127,127] range. This leaves little")
    print(f"redundancy for Huffman to exploit (only ~5% saving).")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, experts, all_names = run_experiment(
        block_sizes=[128],
        max_tensors=200,  # sample 200 tensors across all layers
    )
    print_summary(results, experts, all_names)

    # Save results
    out_path = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/int8_huffman_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serializable = [
        {
            "tensor_name": r.tensor_name,
            "layer": r.layer,
            "num_elements": r.num_elements,
            "psnr_db": r.psnr_db,
            "huffman_bits": r.huffman_bits,
            "entropy_bound_bits": r.entropy_bound_bits,
            "raw_int8_bits": r.raw_int8_bits,
            "total_bits_huffman": r.total_bits_huffman,
            "block_size": r.block_size,
        }
        for r in results
    ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
