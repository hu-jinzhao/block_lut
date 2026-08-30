"""Standalone QUIP-style 4-bit quantization evaluation on MoE expert weights.

Compares:
  A. BLOCKLUT-16 (K-means): current approach, ~37.7 dB
  B. QUIP-style (Hadamard incoherence + int4 group128)
  C. Ablation: int4 group128 without rotation
  D. NF4-style group128

Metrics:
  - var-based weight PSNR (comparable to existing experiments)
  - Per-layer output cosine similarity (using calibration inputs)
  - Per-layer output MSE ratio vs lossless

Key: QUIP's claim = incoherence processing reduces OUTPUT distortion,
     even if weight PSNR doesn't improve much.
"""
import os, sys, math, time, json
import numpy as np
import torch
from collections import defaultdict
from safetensors import safe_open
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
INPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/expert_behavior"
OUTPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results"
GROUP_SIZE = 128
N_BITS = 4


# =========================================================================
# Hadamard transform utilities
# =========================================================================

def hadamard_matrix(n):
    """Generate n×n Hadamard matrix (n must be power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError(f"n={n} must be power of 2")
    H = np.ones((1, 1))
    m = 1
    while m < n:
        H = np.block([[H, H], [H, -H]])
        m *= 2
    return H / np.sqrt(n)  # normalized


def randomized_hadamard(dim, seed):
    """Randomized Hadamard transform matrix: diag(S) @ H @ diag(S')
    where S, S' are random sign vectors.
    Returns (matrix, sign1, sign2) for reproducible application.
    """
    rng = np.random.RandomState(seed)
    s1 = rng.choice([-1, 1], size=dim).astype(np.float32)
    s2 = rng.choice([-1, 1], size=dim).astype(np.float32)
    H = hadamard_matrix(dim)
    # D @ H @ D'
    # For column transform: W @ V^T where V = D' @ H
    # For row transform: U @ W where U = D @ H
    return s1, s2, H.astype(np.float32)


def apply_left_hadamard(W, s, H):
    """Apply U @ W where U = diag(s) @ H."""
    # W: (d_out, d_in), H: (d_out, d_out)
    return np.diag(s) @ H @ W


def apply_right_hadamard(W, s, H):
    """Apply W @ V^T where V = diag(s) @ H, so V^T = H^T @ diag(s) = H @ diag(s)."""
    return W @ H @ np.diag(s)


def apply_inv_left_hadamard(W, s, H):
    """Apply U^T @ W."""
    return H.T @ np.diag(s) @ W


def apply_inv_right_hadamard(W, s, H):
    """Apply W @ V."""
    return W @ np.diag(s) @ H.T


# =========================================================================
# Quantization
# =========================================================================

def groupwise_quantize(W, nbits, group_size):
    """Direct group-wise uniform quantization.
    Returns (indices, scales) where indices are uint8 and scales are float32.
    """
    max_val = 2 ** (nbits - 1) - 1
    d_out, d_in = W.shape
    flat = W.ravel().astype(np.float64)  # high precision for accumulation
    n = flat.size
    ng = (n + group_size - 1) // group_size
    pad = ng * group_size - n
    if pad > 0:
        flat = np.pad(flat, (0, pad))

    scales = np.zeros(ng, dtype=np.float32)
    indices = np.zeros(ng * group_size, dtype=np.int8)

    for g in range(ng):
        s = g * group_size
        e = s + group_size
        group = flat[s:e]
        amax = np.max(np.abs(group))
        if amax < 1e-12:
            amax = 1e-12
        scales[g] = amax / max_val
        q = np.clip(np.round(group / scales[g]), -max_val - 1, max_val)
        indices[s:e] = q.astype(np.int8)

    return indices.astype(np.uint8), scales, d_out, d_in


def groupwise_dequantize(indices, scales, d_out, d_in, group_size):
    """Dequantize from group-wise format."""
    max_val = 2 ** (N_BITS - 1) - 1
    ng = len(scales)
    int_vals = indices.view(np.int8).astype(np.float64).reshape(ng, group_size)
    flat = (int_vals * scales[:, np.newaxis]).ravel()
    n_orig = d_out * d_in
    return flat[:n_orig].reshape(d_out, d_in).astype(np.float32)


# =========================================================================
# NF4 quantization
# =========================================================================

# NF4 levels optimized for N(0,1), from QLoRA paper
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
], dtype=np.float32)


def nf4_groupwise_quantize(W, group_size):
    """NF4 quantization on per-group basis.
    NF4 levels are on [-1, 1]. We scale per-group values to this range.
    Returns (indices, scales, offsets) or just (indices, scales) if symmetric.
    """
    d_out, d_in = W.shape
    flat = W.ravel().astype(np.float64)
    n = flat.size
    ng = (n + group_size - 1) // group_size
    pad = ng * group_size - n
    if pad > 0:
        flat = np.pad(flat, (0, pad))

    scales = np.zeros(ng, dtype=np.float32)
    indices = np.zeros(ng * group_size, dtype=np.uint8)

    midpoints = (NF4_LEVELS[:-1] + NF4_LEVELS[1:]) / 2.0

    for g in range(ng):
        s = g * group_size
        e = s + group_size
        group = flat[s:e]
        amax = np.max(np.abs(group))
        if amax < 1e-12:
            amax = 1e-12
        scales[g] = amax
        normed = group / amax
        # Find nearest NF4 level
        idx = np.searchsorted(midpoints, normed)
        indices[s:e] = np.clip(idx, 0, 15).astype(np.uint8)

    return indices, scales, d_out, d_in


def nf4_groupwise_dequantize(indices, scales, d_out, d_in, group_size):
    ng = len(scales)
    nf4_vals = NF4_LEVELS[indices.astype(np.int32)].reshape(ng, group_size)
    flat = (nf4_vals * scales[:, np.newaxis]).ravel()
    n_orig = d_out * d_in
    return flat[:n_orig].reshape(d_out, d_in).astype(np.float32)


# =========================================================================
# BlockLUT (K-means) quantization — simplified for evaluation
# =========================================================================

def build_kmeans_lut(values, n_centroids, random_state=42):
    """Build K-means LUT from normalized values."""
    from sklearn.cluster import KMeans
    if len(values) > 200000:
        rng = np.random.RandomState(random_state)
        values = rng.choice(values, 200000, replace=False)
    km = KMeans(n_clusters=n_centroids, random_state=random_state,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(values.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def blocklut_quantize(W, lut, block_size=128):
    """Block128 absmax normalize → LUT quantize."""
    d_out, d_in = W.shape
    flat = W.ravel().astype(np.float32)
    n = flat.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(nb, block_size)
    absmax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-12)
    normed = (blocks / absmax[:, np.newaxis]).ravel()

    # Nearest centroid
    midpoints = (lut[:-1] + lut[1:]) / 2.0
    idx = np.searchsorted(midpoints, normed).astype(np.uint8)
    return idx, absmax, d_out, d_in


def blocklut_dequantize(idx, absmax, lut, d_out, d_in, block_size=128):
    nb = len(absmax)
    n_orig = d_out * d_in
    recon_norm = lut[idx.astype(np.int32)].reshape(nb, block_size)
    recon = (recon_norm * absmax[:, np.newaxis]).ravel()
    return recon[:n_orig].reshape(d_out, d_in).astype(np.float32)


# =========================================================================
# Metrics
# =========================================================================

def compute_psnr(orig, recon):
    """Var-based PSNR matching existing experiments."""
    var = np.var(orig)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return 99.0
    return float(10 * math.log10(var / mse))


def compute_output_cosine(Y_orig, Y_quant):
    """Average cosine similarity per output vector."""
    y1 = Y_orig.reshape(-1, Y_orig.shape[-1])
    y2 = Y_quant.reshape(-1, Y_quant.shape[-1])
    norms1 = np.linalg.norm(y1, axis=1)
    norms2 = np.linalg.norm(y2, axis=1)
    denom = norms1 * norms2 + 1e-12
    cos_sim = np.sum(y1 * y2, axis=1) / denom
    return float(np.mean(cos_sim)), float(np.min(cos_sim))


# =========================================================================
# Main evaluation
# =========================================================================

def main():
    t_start = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load calibration inputs
    print("Loading calibration inputs...")
    calib = {}
    for layer in range(24):
        path = os.path.join(INPUT_DIR, f"inputs_layer_{layer:02d}.npy")
        if os.path.exists(path):
            calib[layer] = torch.from_numpy(np.load(path).astype(np.float32))
    print(f"  Loaded inputs for {len(calib)} layers")

    # Build weight index
    safetensor_files = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")
    )

    w_index = defaultdict(dict)  # layer -> expert -> proj_type -> (file, key)
    for fp in safetensor_files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                parts = k.split(".")
                layer = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6]
                if expert_idx not in w_index[layer]:
                    w_index[layer][expert_idx] = {}
                w_index[layer][expert_idx][proj_type] = (fp, k)

    layers_moe = sorted(w_index.keys())
    n_experts = max(w_index[layers_moe[0]].keys()) + 1
    print(f"MoE layers: {len(layers_moe)}, experts: {n_experts}")

    # Sample layers for evaluation
    eval_layers = [0, 5, 10, 15, 20, 23]  # same as behavior clustering
    eval_layers = [l for l in eval_layers if l in calib]

    # Build K-means 16-LUT
    print("\nBuilding K-means 16-LUT...")
    lut_samples = []
    for fp in safetensor_files[:2]:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                if len(lut_samples) > 10000000:
                    break
                W = f.get_tensor(k).to(torch.float32).numpy()
                nb = max(1, W.size // GROUP_SIZE)
                pad = nb * GROUP_SIZE - W.size
                if pad:
                    W_flat = np.pad(W.ravel(), (0, pad))
                else:
                    W_flat = W.ravel()
                for b in range(min(nb, 3)):
                    s, e = b * GROUP_SIZE, (b + 1) * GROUP_SIZE
                    block = W_flat[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    lut_samples.append(block / amax)
            if len(lut_samples) > 10000000:
                break

    lut_samples = np.concatenate([s.ravel() for s in lut_samples])
    print(f"  {len(lut_samples):,} normalized values for LUT training")
    n_sample = min(200000, len(lut_samples))
    lut_16 = build_kmeans_lut(np.random.choice(lut_samples, n_sample, replace=False), 15)
    print(f"  LUT-16: range=[{lut_16[0]:.4f}, {lut_16[-1]:.4f}], entries={len(lut_16)}")

    # =====================================================================
    # Evaluate each method on each (layer, expert, projection)
    # =====================================================================
    results = {method: {"psnr": [], "cos_mean": [], "cos_min": [], "mse_ratio": []}
               for method in ["BLOCKLUT-16", "int4_g128", "QUIP-int4", "NF4_g128"]}

    total_tensors = 0

    for layer_idx, layer in enumerate(eval_layers):
        if layer not in calib:
            continue
        X = calib[layer].numpy()  # (N, d_in)
        print(f"\n{'='*70}")
        print(f"Layer {layer}: {X.shape[0]} calibration samples")

        layer_results = {m: {"psnr": [], "cos_mean": [], "cos_min": []} for m in results}

        for e in range(n_experts):
            for ptype in ["gate_proj", "up_proj", "down_proj"]:
                if e not in w_index[layer] or ptype not in w_index[layer][e]:
                    continue
                fp, key = w_index[layer][e][ptype]
                with safe_open(fp, framework="pt", device="cpu") as f:
                    W_orig = f.get_tensor(key).to(torch.float32).numpy()
                d_out, d_in = W_orig.shape

                # --- BLOCKLUT-16 ---
                idx, am, do, di = blocklut_quantize(W_orig, lut_16, GROUP_SIZE)
                W_blut = blocklut_dequantize(idx, am, lut_16, do, di, GROUP_SIZE)
                psnr_blut = compute_psnr(W_orig.ravel(), W_blut.ravel())

                # --- int4 group128 (no rotation, no blocklut) ---
                idx, scales, do, di = groupwise_quantize(W_orig, N_BITS, GROUP_SIZE)
                W_int4 = groupwise_dequantize(idx, scales, do, di, GROUP_SIZE)
                psnr_int4 = compute_psnr(W_orig.ravel(), W_int4.ravel())

                # --- QUIP-style ---
                # Pad to Hadamard-compatible dims
                def next_pow2(x):
                    return 1 << (x - 1).bit_length()

                # For QUIP, apply row-wise Hadamard (mix output channels)
                # Row transform: U @ W where U = D @ H
                do_pad = next_pow2(d_out)
                di_pad = next_pow2(d_in)
                W_padded = np.pad(W_orig.astype(np.float64),
                                  ((0, do_pad - d_out), (0, di_pad - d_in)))

                # Generate randomized Hadamard (seeded by tensor identity)
                seed = hash((layer, e, ptype)) & 0x7FFFFFFF
                s_row, _, H_row = randomized_hadamard(do_pad, seed)
                s_col, _, H_col = randomized_hadamard(di_pad, seed + 1)

                # Apply incoherence: W_tilde = U @ W @ V^T
                W_tilde = apply_left_hadamard(W_padded, s_row, H_row)
                W_tilde = apply_right_hadamard(W_tilde, s_col, H_col)

                # Quantize in transformed space
                idx_q, scales_q, _, _ = groupwise_quantize(W_tilde, N_BITS, GROUP_SIZE)
                W_tilde_q = groupwise_dequantize(idx_q, scales_q, do_pad, di_pad, GROUP_SIZE)

                # Inverse transform
                W_quip = apply_inv_right_hadamard(W_tilde_q, s_col, H_col)
                W_quip = apply_inv_left_hadamard(W_quip, s_row, H_row)

                # Remove padding
                W_quip = W_quip[:d_out, :d_in]
                psnr_quip = compute_psnr(W_orig.ravel(), W_quip.ravel())

                # --- NF4 group128 ---
                idx_nf, scales_nf, do_nf, di_nf = nf4_groupwise_quantize(W_orig, GROUP_SIZE)
                W_nf4 = nf4_groupwise_dequantize(idx_nf, scales_nf, do_nf, di_nf, GROUP_SIZE)
                psnr_nf4 = compute_psnr(W_orig.ravel(), W_nf4.ravel())

                # --- Output-level metrics ---
                # For gate_proj/up_proj: Y = X @ W^T (X is the MoE layer input)
                # For down_proj: skip output-level (different input space: gate*up output)
                if ptype != "down_proj":
                    Y_orig = X @ W_orig.T
                    Y_blut = X @ W_blut.T
                    Y_int4 = X @ W_int4.T
                    Y_quip = X @ W_quip.T
                    Y_nf4 = X @ W_nf4.T

                    for method, Y_q in [("BLOCKLUT-16", Y_blut), ("int4_g128", Y_int4),
                                        ("QUIP-int4", Y_quip), ("NF4_g128", Y_nf4)]:
                        cm, cmin = compute_output_cosine(Y_orig, Y_q)
                        layer_results[method]["psnr"].append(psnr_blut if method == "BLOCKLUT-16"
                                                             else psnr_int4 if method == "int4_g128"
                                                             else psnr_quip if method == "QUIP-int4"
                                                             else psnr_nf4)
                        layer_results[method]["cos_mean"].append(cm)
                        layer_results[method]["cos_min"].append(cmin)
                else:
                    # Weight PSNR only for down_proj
                    for method in layer_results:
                        layer_results[method]["psnr"].append(psnr_blut if method == "BLOCKLUT-16"
                                                             else psnr_int4 if method == "int4_g128"
                                                             else psnr_quip if method == "QUIP-int4"
                                                             else psnr_nf4)
                        layer_results[method]["cos_mean"].append(1.0)
                        layer_results[method]["cos_min"].append(1.0)

                total_tensors += 1

        # Print per-layer summary
        for method in results:
            ps = layer_results[method]["psnr"]
            cm = layer_results[method]["cos_mean"]
            if ps:
                print(f"  {method:<15}: PSNR={np.mean(ps):.2f}, "
                      f"cos_mean={np.mean(cm):.4f}, cos_min={np.min(layer_results[method]['cos_min']):.4f}")

        # Accumulate global
        for method in results:
            results[method]["psnr"].extend(layer_results[method]["psnr"])
            results[method]["cos_mean"].extend(layer_results[method]["cos_mean"])
            results[method]["cos_min"].extend(layer_results[method]["cos_min"])

    # =====================================================================
    # Global summary
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"GLOBAL SUMMARY ({total_tensors} tensors across {len(eval_layers)} layers)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'PSNR':>8} {'cos_mean':>10} {'cos_min95':>10} {'cos_min':>10}")
    print(f"{'-'*55}")

    final = {}
    for method in results:
        ps = results[method]["psnr"]
        cm = results[method]["cos_mean"]
        cmin = results[method]["cos_min"]
        if ps:
            # 5th percentile of min cos (robust worst-case)
            cmin_sorted = sorted(cmin)
            cmin_p5 = cmin_sorted[max(0, int(len(cmin_sorted) * 0.05))]
            print(f"  {method:<15} {np.mean(ps):>8.2f} {np.mean(cm):>10.4f} "
                  f"{cmin_p5:>10.4f} {np.min(cmin):>10.4f}")
            final[method] = {
                "psnr_mean": float(np.mean(ps)),
                "cos_mean": float(np.mean(cm)),
                "cos_min_p5": float(cmin_p5),
                "cos_min_abs": float(np.min(cmin)),
            }

    # Save
    out_path = os.path.join(OUTPUT_DIR, "quip_4bit_eval.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved: {out_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
