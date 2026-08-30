"""
Expert 矩阵结构规律分析

探索方向:
1. SVD 低秩特性 — 每个 expert 矩阵的奇异值衰减速度
2. 层内 expert 相似性 — 同层 60 个 expert 的 pairwise cosine similarity
3. Mean+Delta 编码 — 层均值 + per-expert 残差，残差方差/熵 vs 原始
4. 跨层模式 — 不同层 expert 是否共享子空间
5. 块内结构 — 权重矩阵内部的 block 模式

对每个方向，估算潜在压缩收益。
"""

import os, sys, math, time, json
from collections import Counter
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
import lz4.block as lz4_block

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"

# ============================================================================
# Data loading
# ============================================================================

def load_all_experts(sft_files):
    """Load expert tensors organized by (layer, expert, type)."""
    # Structure: experts[layer][expert_idx][proj_type] = tensor
    experts = {}
    for path in tqdm(sft_files, desc="Loading safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                # model.layers.L.mlp.experts.E.TYPE.weight
                parts = k.split(".")
                layer = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6]  # gate_proj, up_proj, down_proj
                if layer not in experts:
                    experts[layer] = {}
                if expert_idx not in experts[layer]:
                    experts[layer][expert_idx] = {}
                experts[layer][expert_idx][proj_type] = f.get_tensor(k)
    return experts

def get_layer_tensor_matrix(experts, layer, proj_type):
    """
    Stack all experts of a given layer+type into a matrix.
    Shape: (n_experts, n_rows, n_cols)
    """
    n_experts = len(experts[layer])
    first = experts[layer][0][proj_type]
    shape = first.shape
    stack = torch.stack([experts[layer][e][proj_type].to(torch.float32)
                         for e in range(n_experts)])
    return stack  # (E, R, C)

# ============================================================================
# 1. SVD Low-Rank Analysis
# ============================================================================

def analyze_svd_spectrum(tensor, name, percentiles=[50, 80, 90, 95, 99]):
    """Analyze SVD spectrum of a single expert matrix."""
    x = tensor.to(torch.float32).numpy()
    U, S, Vt = np.linalg.svd(x, full_matrices=False)

    total_energy = np.sum(S ** 2)
    cumsum = np.cumsum(S ** 2) / total_energy

    result = {"name": name, "shape": list(x.shape), "total_singular_values": len(S)}
    result["top5_singular"] = S[:5].tolist()
    result["condition_num"] = float(S[0] / S[-1])

    for pct in percentiles:
        rank = int(np.searchsorted(cumsum, pct / 100.0)) + 1
        result[f"rank_{pct}pct"] = rank
        result[f"rank_{pct}pct_ratio"] = rank / len(S)

    result["effective_rank"] = int(np.sum(S) ** 2 / np.sum(S ** 2))  # stable rank

    return result

def svd_analysis(experts):
    """Run SVD analysis on sampled expert matrices."""
    print("\n" + "=" * 100)
    print("1. SVD LOW-RANK ANALYSIS")
    print("=" * 100)

    # Sample experts across layers and types
    results = []
    layers = sorted(experts.keys())
    sample_layers = layers[::4]  # Every 4th layer
    proj_types = ["gate_proj", "up_proj", "down_proj"]

    for layer in sample_layers:
        for ptype in proj_types:
            # Use expert 0 as representative
            tensor = experts[layer][0][ptype]
            name = f"L{layer}_{ptype}"
            r = analyze_svd_spectrum(tensor, name)
            results.append(r)

    # Summary
    print(f"\n{'Tensor':<30} {'Shape':<20} {'Rank50%':>8} {'Rank90%':>8} {'Rank95%':>8} {'Rank99%':>8} {'EffRank':>8}")
    print("-" * 110)
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        print(f"{r['name']:<30} {shape_str:<20} "
              f"{r['rank_50pct']:>8} ({r['rank_50pct_ratio']*100:>4.1f}%) "
              f"{r['rank_90pct']:>8} ({r['rank_90pct_ratio']*100:>4.1f}%) "
              f"{r['rank_95pct']:>8} ({r['rank_95pct_ratio']*100:>4.1f}%) "
              f"{r['rank_99pct']:>8} ({r['rank_99pct_ratio']*100:>4.1f}%) "
              f"{r['effective_rank']:>8}")

    # Compression estimate from low-rank
    # If rank_r captures 99% energy: store U(R×r), S(r), Vt(r×C) instead of R×C
    # Compression = R*C / (R*r + r + r*C) = R*C / (r*(R+C+1))
    print(f"\n--- Low-Rank Compression Potential (99% energy) ---")
    for r in results[:3]:  # Show first few
        R, C = r['shape']
        rank99 = r['rank_99pct']
        original = R * C
        compressed = rank99 * (R + C + 1)
        ratio = original / compressed
        print(f"  {r['name']}: rank99={rank99}, compression ratio={ratio:.2f}x "
              f"({compressed*32/original:.1f} bits/elem if float32, vs 16 bf16)")

# ============================================================================
# 2. Expert Similarity within Layer
# ============================================================================

def analyze_expert_similarity(experts, sample_layers=None):
    """Compute pairwise cosine similarity between experts within each layer."""
    print("\n" + "=" * 100)
    print("2. EXPERT SIMILARITY WITHIN LAYER")
    print("=" * 100)

    layers = sorted(experts.keys())
    if sample_layers is None:
        sample_layers = layers[::3]

    for layer in sample_layers:
        n_experts = len(experts[layer])
        print(f"\n--- Layer {layer} ({n_experts} experts) ---")

        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            # Stack all experts as flattened vectors
            vecs = []
            for e in range(n_experts):
                v = experts[layer][e][ptype].to(torch.float32).ravel()
                vecs.append(v)
            vecs = torch.stack(vecs)  # (E, N)

            # Cosine similarity matrix
            norms = torch.norm(vecs, dim=1, keepdim=True)
            normalized = vecs / (norms + 1e-12)
            sim = normalized @ normalized.T  # (E, E)

            # Statistics (excluding diagonal)
            mask = ~torch.eye(n_experts, dtype=torch.bool)
            off_diag = sim[mask]

            print(f"  {ptype:<15}: mean_sim={off_diag.mean():.4f}, "
                  f"min={off_diag.min():.4f}, max={off_diag.max():.4f}, "
                  f"std={off_diag.std():.4f}")

# ============================================================================
# 3. Mean + Delta Encoding
# ============================================================================

def huffman_estimate(data: np.ndarray) -> float:
    """Estimate Huffman bits per element."""
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    # Build Huffman tree
    import heapq
    heap = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, id(sym), [sym, f]))
    if len(heap) == 0:
        return 0
    if len(heap) == 1:
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
    avg_bits = sum(freq[sym] * lengths.get(sym, 1) for sym in freq) / total
    return avg_bits

def shannon_entropy(data: np.ndarray) -> float:
    freq = Counter(data.ravel().tolist())
    total = sum(freq.values())
    return -sum(c/total * math.log2(c/total) for c in freq.values())

def analyze_mean_delta(experts, sample_layers=None):
    """
    Mean + Delta encoding:
    - Compute layer-wise mean expert (per projection type)
    - Delta = Expert - Mean
    - Measure: delta variance vs original, delta entropy, compression potential
    """
    print("\n" + "=" * 100)
    print("3. MEAN + DELTA ENCODING ANALYSIS")
    print("=" * 100)

    layers = sorted(experts.keys())
    if sample_layers is None:
        sample_layers = layers[::2]

    print(f"\n{'Layer':<8} {'Proj':<15} {'OrigStd':>10} {'DeltaStd':>10} {'StdRatio':>10} "
          f"{'OrigEnt':>10} {'DeltaEnt':>10} {'EntRatio':>10} {'block128PSNR':>12} {'HuffDelta':>12}")
    print("-" * 130)

    all_std_ratios = []
    all_ent_ratios = []
    all_delta_psnrs = []
    all_huffman_bits = []

    for layer in sample_layers:
        n_experts = len(experts[layer])

        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            stack = torch.stack([experts[layer][e][ptype].to(torch.float32)
                                for e in range(n_experts)])  # (E, R, C)

            # Layer mean
            mean_expert = stack.mean(dim=0)  # (R, C)

            # Deltas
            deltas = stack - mean_expert.unsqueeze(0)  # (E, R, C)

            # Variance comparison
            orig_std = stack.std().item()
            delta_std = deltas.std().item()
            std_ratio = delta_std / orig_std

            # Entropy of raw values (quantized to int8)
            # For original: use block-wise int8 against global mean
            # For delta: use block-wise int8 against delta range

            # Block128 int8 PSNR for delta
            def block128_psnr(tensor):
                x = tensor.to(torch.float32).numpy().ravel()
                n = x.size
                bs = 128
                nb = (n + bs - 1) // bs
                pad = nb * bs - n
                if pad > 0:
                    x = np.pad(x, (0, pad))

                rec = np.zeros_like(x)
                for b in range(nb):
                    s = b * bs
                    e = s + bs
                    block = x[s:e]
                    amax = np.max(np.abs(block))
                    if amax == 0:
                        amax = 1e-12
                    scale = amax / 127.5
                    q = np.clip(np.round(block / scale), -128, 127).astype(np.float32)
                    rec[s:e] = q * scale

                x_orig = x[:n]
                x_rec = rec[:n]
                mse = np.mean((x_orig - x_rec) ** 2)
                var = np.var(x_orig)
                if mse == 0:
                    return float('inf')
                return 10 * math.log10(var / mse)

            # PSNR of block128 int8 on delta only (we'd store mean separately)
            # Mean is stored losslessly, so only delta quality matters
            if deltas.numel() > 0 and deltas.abs().max() > 0:
                delta_psnr = block128_psnr(deltas)
            else:
                delta_psnr = float('inf')

            # Estimate Huffman bits for int8-quantized delta
            delta_np = deltas.numpy().ravel()
            # Block128 quantize delta
            bs = 128
            n = delta_np.size
            nb = (n + bs - 1) // bs
            pad = nb * bs - n
            if pad > 0:
                delta_np = np.pad(delta_np, (0, pad))
            indices = np.zeros(nb * bs, dtype=np.uint8)
            for b in range(nb):
                s = b * bs
                e = s + bs
                block = delta_np[s:e]
                amax = np.max(np.abs(block))
                if amax == 0:
                    amax = 1e-12
                scale = amax / 127.5
                q = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
                indices[s:e] = q.view(np.uint8)
            huff_bits = huffman_estimate(indices)

            # Entropy of raw delta values (binned)
            # Normalize delta to [0, 255] range for entropy estimate
            d_absmax = np.max(np.abs(delta_np))
            if d_absmax > 0:
                delta_quant_8bit = np.clip(np.round(delta_np / d_absmax * 127.5 + 127.5), 0, 255).astype(np.uint8)
                delta_ent = shannon_entropy(delta_quant_8bit)
            else:
                delta_ent = 0

            # Original entropy
            orig_np = stack.numpy().ravel()
            o_absmax = np.max(np.abs(orig_np))
            orig_quant_8bit = np.clip(np.round(orig_np / o_absmax * 127.5 + 127.5), 0, 255).astype(np.uint8)
            orig_ent = shannon_entropy(orig_quant_8bit)

            ent_ratio = delta_ent / orig_ent if orig_ent > 0 else 0

            print(f"{layer:<8} {ptype:<15} {orig_std:>10.4f} {delta_std:>10.4f} {std_ratio:>10.4f} "
                  f"{orig_ent:>10.3f} {delta_ent:>10.3f} {ent_ratio:>10.4f} {delta_psnr:>12.1f} {huff_bits:>12.3f}")

            all_std_ratios.append(std_ratio)
            all_ent_ratios.append(ent_ratio)
            all_delta_psnrs.append(delta_psnr)
            all_huffman_bits.append(huff_bits)

    # Storage analysis
    print(f"\n--- Storage Analysis for Mean+Delta Scheme ---")
    # Mean stored losslessly: 1 copy per layer × 3 types = mean_cost
    # Deltas stored as block128 int8: 8.125 bits/elem per expert
    # Total = mean/N_experts + 8.125 + absmax
    # But mean is tiny compared to 60 experts → negligible overhead
    mean_std_ratio = np.mean(all_std_ratios)
    mean_delta_psnr = np.mean([x for x in all_delta_psnrs if x != float('inf')])
    mean_huff = np.mean(all_huffman_bits)

    print(f"  Mean std(delta)/std(original): {mean_std_ratio:.4f}")
    print(f"  Delta has {mean_std_ratio*100:.1f}% of original std → needs fewer bits for same PSNR")
    print(f"  Block128 int8 PSNR on delta (not mean): {mean_delta_psnr:.1f} dB")
    print(f"  Huffman bits for block128 delta indices: {mean_huff:.3f}")

    # If we store delta with block128 int8:
    # bits = absmax(16) / 128 + Huffman(delta_indices)
    # The key question: can we use fewer bits for delta than for original?
    # Delta has smaller absmax → same 8-bit quantization has smaller step → higher PSNR
    # OR: we can use fewer bits (e.g. int6 or int4) for same PSNR as int8 on original
    absmax_bits = 16 / 128  # 0.125
    delta_bits = mean_huff + absmax_bits
    print(f"  Total delta storage: {delta_bits:.3f} bits/elem (Huffman indices + absmax)")
    print(f"  + mean overhead: {16*3/60:.3f} bits/elem (3 means shared by 60 experts)")
    print(f"  Total: {delta_bits + 16*3/60:.3f} bits/elem for mean+delta scheme")

# ============================================================================
# 4. Cross-Layer Expert Analysis
# ============================================================================

def analyze_cross_layer(experts):
    """Check if experts from different layers share structure."""
    print("\n" + "=" * 100)
    print("4. CROSS-LAYER EXPERT ANALYSIS")
    print("=" * 100)

    layers = sorted(experts.keys())

    # For each projection type, compute layer-mean expert and check cross-layer similarity
    for ptype in ["gate_proj", "up_proj", "down_proj"]:
        layer_means = []
        for layer in layers:
            n_experts = len(experts[layer])
            stack = torch.stack([experts[layer][e][ptype].to(torch.float32)
                                for e in range(n_experts)])
            layer_means.append(stack.mean(dim=0).ravel())
        layer_means = torch.stack(layer_means)  # (L, N)

        # Cross-layer cosine similarity
        norms = torch.norm(layer_means, dim=1, keepdim=True)
        normalized = layer_means / (norms + 1e-12)
        sim = normalized @ normalized.T

        off_diag_mask = ~torch.eye(len(layers), dtype=torch.bool)
        off_diag = sim[off_diag_mask]

        print(f"\n  {ptype}: Cross-layer mean-expert cosine similarity")
        print(f"    Mean: {off_diag.mean():.4f}, Std: {off_diag.std():.4f}")
        print(f"    Min: {off_diag.min():.4f} (L{divmod(off_diag.argmin().item(), len(layers)-1)[0]}-L{divmod(off_diag.argmin().item(), len(layers)-1)[1]+1})")
        print(f"    Max: {off_diag.max():.4f}")

        # Adjacent layer similarity (layer i vs layer i+1)
        adj_sims = []
        for i in range(len(layers) - 1):
            adj_sims.append(sim[i, i+1].item())
        print(f"    Adjacent layers mean sim: {np.mean(adj_sims):.4f}")

# ============================================================================
# 5. Weight Distribution Analysis
# ============================================================================

def analyze_weight_distribution(experts):
    """Analyze the distribution shape of expert weights."""
    print("\n" + "=" * 100)
    print("5. WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 100)

    layers = sorted(experts.keys())
    sample_layers = layers[::6]

    for layer in sample_layers:
        print(f"\n--- Layer {layer} ---")
        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            # Compute statistics across all experts in layer
            all_vals = []
            for e in range(len(experts[layer])):
                v = experts[layer][e][ptype].to(torch.float32).ravel()
                all_vals.append(v)
            all_vals = torch.cat(all_vals)

            # Kurtosis (excess): measures tail heaviness vs Gaussian
            mean = all_vals.mean()
            var = all_vals.var()
            std = var.sqrt()
            z = (all_vals - mean) / std
            kurtosis = (z ** 4).mean().item() - 3  # excess kurtosis

            # Sparsity: fraction of values near zero
            threshold = 0.01 * std.item()
            near_zero = (all_vals.abs() < threshold).float().mean().item()

            # Range
            absmax = all_vals.abs().max().item()

            print(f"  {ptype:<15}: mean={mean:.6f}, std={std:.4f}, "
                  f"absmax={absmax:.4f}, kurtosis={kurtosis:.2f}, "
                  f"near_zero(<0.01σ)={near_zero*100:.1f}%")

# ============================================================================
# 6. Expert Subspace Sharing (PCA on expert vectors)
# ============================================================================

def analyze_expert_subspace(experts, sample_layers=None):
    """
    Treat each expert as a point in weight space.
    PCA on the expert vectors to see how many principal directions capture expert variation.
    """
    print("\n" + "=" * 100)
    print("6. EXPERT SUBSPACE DIMENSIONALITY (PCA on expert vectors)")
    print("=" * 100)

    layers = sorted(experts.keys())
    if sample_layers is None:
        sample_layers = layers[::3]

    for layer in sample_layers:
        n_experts = len(experts[layer])
        print(f"\n--- Layer {layer} ---")

        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            # Expert vectors
            vecs = []
            for e in range(n_experts):
                v = experts[layer][e][ptype].to(torch.float32).ravel()
                vecs.append(v)
            vecs = torch.stack(vecs)  # (E, N)

            # Center by subtracting mean
            mean_vec = vecs.mean(dim=0)
            centered = vecs - mean_vec  # (E, N)

            # SVD of centered expert matrix
            U, S, Vt = np.linalg.svd(centered.numpy(), full_matrices=False)

            total_var = np.sum(S ** 2)
            cumsum = np.cumsum(S ** 2) / total_var

            # How many PCs for 95%, 99% of expert variation?
            pc95 = int(np.searchsorted(cumsum, 0.95)) + 1
            pc99 = int(np.searchsorted(cumsum, 0.99)) + 1

            print(f"  {ptype:<15}: n_experts={n_experts}, "
                  f"PCs for 95%={pc95} ({pc95/n_experts*100:.0f}%), "
                  f"99%={pc99} ({pc99/n_experts*100:.0f}%), "
                  f"top5={S[:5].astype(int).tolist()}")

            # Implication: if experts span K-dimensional subspace,
            # we can store: basis (K, N) + coefficients (E, K)
            # Storage: K*N + E*K vs E*N
            # Compression: E*N / (K*N + E*K)
            if pc95 < n_experts * 0.5:
                N = vecs.shape[1]
                E = n_experts
                original = E * N
                compressed = pc95 * N + E * pc95
                ratio = original / compressed
                print(f"    → PCA-{pc95} compression: {ratio:.1f}x "
                      f"({compressed/original*16:.1f} bits/elem for bf16 basis+coeffs)")
                print(f"       Basis: {pc95}×{N} + Coeffs: {E}×{pc95}")

# ============================================================================
# 7. Block128 quantized Delta PSNR distribution
# ============================================================================

def analyze_delta_quantization_quality(experts):
    """
    For the mean+delta scheme, quantify:
    - What PSNR do we get if we quantize delta at different bit widths?
    - At what bit width does delta PSNR match the origin int8 PSNR?
    """
    print("\n" + "=" * 100)
    print("7. DELTA QUANTIZATION: BIT-WIDTH SWEEP")
    print("=" * 100)

    layers = sorted(experts.keys())
    sample_layers = layers[::4]

    bit_widths = [4, 5, 6, 7, 8]
    results = {bw: {"psnr": [], "bits_total": []} for bw in bit_widths}

    for layer in sample_layers:
        n_experts = len(experts[layer])

        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            stack = torch.stack([experts[layer][e][ptype].to(torch.float32)
                                for e in range(n_experts)])
            mean_expert = stack.mean(dim=0)
            deltas = stack - mean_expert.unsqueeze(0)

            delta_np = deltas.numpy().ravel()

            for bw in bit_widths:
                max_val = 2 ** (bw - 1) - 1  # e.g., 127 for int8

                # Block128 quantization at this bit width
                bs = 128
                n = delta_np.size
                nb = (n + bs - 1) // bs
                pad = nb * bs - n
                x = np.pad(delta_np, (0, pad)) if pad > 0 else delta_np.copy()

                rec = np.zeros_like(x)
                for b in range(nb):
                    s = b * bs
                    e = s + bs
                    block = x[s:e]
                    amax = np.max(np.abs(block))
                    if amax == 0:
                        amax = 1e-12
                    scale = amax / max_val
                    q = np.clip(np.round(block / scale), -max_val-1, max_val).astype(np.float32)
                    rec[s:e] = q * scale

                x_orig = delta_np
                x_rec = rec[:n]
                mse = np.mean((x_orig - x_rec) ** 2)
                var = np.var(x_orig)
                if mse == 0:
                    psnr = float('inf')
                else:
                    psnr = 10 * math.log10(var / mse)

                # Total bits: absmax (16-bit bf16) per block + bw bits per element
                absmax_bits = 16 / bs
                total_bits = bw + absmax_bits

                results[bw]["psnr"].append(psnr)
                results[bw]["bits_total"].append(total_bits)

    print(f"\n{'BitWidth':<10} {'BitsTotal':>12} {'DeltaPSNR':>12} {'PSNR vs orig int8':>18}")
    print("-" * 60)

    # Original int8 PSNR (from previous experiment ≈ 43.3 dB)
    orig_int8_psnr = 43.3

    for bw in bit_widths:
        mean_psnr = np.mean([x for x in results[bw]["psnr"] if x != float('inf')])
        mean_bits = np.mean(results[bw]["bits_total"])
        improvement = mean_psnr - orig_int8_psnr
        print(f"{bw:<10} {mean_bits:>12.3f} {mean_psnr:>12.1f} {improvement:>+18.1f}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Loading {len(sft_files)} safetensor files...")
    experts = load_all_experts(sft_files)
    print(f"Loaded: {len(experts)} layers, {len(experts[0])} experts/layer")
    print(f"Tensor shapes: gate={experts[0][0]['gate_proj'].shape}, "
          f"up={experts[0][0]['up_proj'].shape}, down={experts[0][0]['down_proj'].shape}")

    # Run all analyses
    svd_analysis(experts)
    analyze_expert_similarity(experts)
    analyze_mean_delta(experts)
    analyze_cross_layer(experts)
    analyze_weight_distribution(experts)
    analyze_expert_subspace(experts)
    analyze_delta_quantization_quality(experts)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
