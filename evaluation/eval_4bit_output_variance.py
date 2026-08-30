"""Fast per-expert 4-bit output perturbation analysis.

Uses saved calibration inputs. For each expert, computes output with 4-bit
quantized weights vs 8-bit weights, measures output cosine similarity.

Key question: does output perturbation vary significantly across experts?
If yes → KL-based tier assignment is promising.
If no → all experts equally sensitive, no differentiation possible.

Runtime: ~5 min (matrix multiplies only, no model inference).
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


# =========================================================================
# K-means LUT for 4-bit (16 centroids = 15 in K-means index)
# =========================================================================

def build_kmeans_lut(values, n_centroids):
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(42)
    if len(values) > 200000:
        values = rng.choice(values, 200000, replace=False)
    km = KMeans(n_clusters=n_centroids, random_state=42, n_init=3, max_iter=100)
    km.fit(values.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def blocklut_quantize(W, lut, block_size=128):
    flat = W.ravel().astype(np.float32)
    n = flat.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad: flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(nb, block_size)
    absmax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-12)
    normed = (blocks / absmax[:, np.newaxis]).ravel()
    midpoints = (lut[:-1] + lut[1:]) / 2.0
    idx = np.searchsorted(midpoints, normed).astype(np.uint8)
    return idx, absmax


def blocklut_dequantize(idx, absmax, lut, out_shape, block_size=128):
    nb = len(absmax)
    d_out, d_in = out_shape
    n_orig = d_out * d_in
    recon_norm = lut[idx.astype(np.int32)].reshape(nb, block_size)
    recon = (recon_norm * absmax[:, np.newaxis]).ravel()
    return recon[:n_orig].reshape(d_out, d_in).astype(np.float32)


def compute_psnr(orig, recon):
    var = np.var(orig)
    mse = np.mean((orig - recon) ** 2)
    return 99.0 if mse == 0 else float(10 * math.log10(var / mse))


def compute_output_cosine(Y1, Y2):
    """Per-row cosine similarity, returns (mean, std, min)."""
    y1 = Y1.reshape(-1, Y1.shape[-1])
    y2 = Y2.reshape(-1, Y2.shape[-1])
    n1 = np.linalg.norm(y1, axis=1) + 1e-12
    n2 = np.linalg.norm(y2, axis=1) + 1e-12
    cos = np.sum(y1 * y2, axis=1) / (n1 * n2)
    return float(np.mean(cos)), float(np.std(cos)), float(np.min(cos))


def compute_output_mse_ratio(Y_clean, Y_quant):
    """MSE of quantized output / MSE of clean output (relative error)."""
    mse = np.mean((Y_clean - Y_quant) ** 2)
    var = np.var(Y_clean)
    return float(mse / var) if var > 0 else 0.0


# =========================================================================
# Main
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
            calib[layer] = np.load(path).astype(np.float32)
    print(f"  Loaded inputs for {len(calib)} layers")

    # Build weight index
    sft_files = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")
    )
    w_index = defaultdict(dict)
    for fp in sft_files:
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
    eval_layers = [0, 5, 10, 15, 20, 23]
    eval_layers = [l for l in eval_layers if l in calib]

    # =====================================================================
    # Build 4-bit K-means LUT
    # =====================================================================
    print("\nBuilding 4-bit K-means LUT...")
    lut_samples = []
    for fp in sft_files[:2]:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                W = f.get_tensor(k).float().numpy()
                nb = max(1, W.size // GROUP_SIZE)
                for b in range(min(nb, 2)):
                    s, e = b * GROUP_SIZE, (b + 1) * GROUP_SIZE
                    block = W.ravel()[s:e] if e <= W.size else np.pad(W.ravel()[s:], (0, e - W.size))
                    amax = np.max(np.abs(block))
                    if amax < 1e-12: continue
                    lut_samples.append(block / amax)
    lut_samples = np.concatenate([s.ravel() for s in lut_samples])
    n_sample = min(200000, len(lut_samples))
    lut_16 = build_kmeans_lut(np.random.choice(lut_samples, n_sample, replace=False), 15)
    print(f"  LUT-16: {len(lut_16)} centroids, range=[{lut_16[0]:.4f}, {lut_16[-1]:.4f}]")

    # =====================================================================
    # Per-expert output perturbation analysis
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"PER-EXPERT 4-BIT OUTPUT PERTURBATION ANALYSIS")
    print(f"  {len(eval_layers)} layers × {n_experts} experts × 2 projections")
    print(f"{'='*80}")

    all_results = []  # list of {layer, expert, ptype, psnr, cos_mean, cos_min, mse_ratio}

    for layer in eval_layers:
        X = calib[layer].astype(np.float32)  # (N, 2048)
        print(f"\n  Layer {layer}: ", end="", flush=True)

        layer_results = []
        for e in tqdm(range(n_experts), desc=f"  L{layer}", leave=False):
            for ptype in ["gate_proj", "up_proj"]:  # skip down_proj (different input)
                if e not in w_index[layer] or ptype not in w_index[layer][e]:
                    continue
                fp, key = w_index[layer][e][ptype]
                with safe_open(fp, framework="pt", device="cpu") as f:
                    W = f.get_tensor(key).to(torch.float32).numpy()
                d_out, d_in = W.shape

                # 8-bit "clean" output (K-means 256 would be ~60+ dB, essentially lossless)
                # Use original weights as clean reference
                Y_clean = X @ W.T

                # 4-bit quantized output
                idx, am = blocklut_quantize(W, lut_16, GROUP_SIZE)
                W_4bit = blocklut_dequantize(idx, am, lut_16, (d_out, d_in), GROUP_SIZE)
                Y_4bit = X @ W_4bit.T

                # Metrics
                psnr_val = compute_psnr(W.ravel(), W_4bit.ravel())
                cos_mean, cos_std, cos_min = compute_output_cosine(Y_clean, Y_4bit)
                mse_r = compute_output_mse_ratio(Y_clean, Y_4bit)

                layer_results.append({
                    "layer": layer, "expert": e, "ptype": ptype,
                    "psnr": psnr_val, "cos_mean": cos_mean,
                    "cos_min": cos_min, "mse_ratio": mse_r,
                })
                all_results.append(layer_results[-1])

        # Per-layer summary
        psnrs = [r["psnr"] for r in layer_results]
        cos_means = [r["cos_mean"] for r in layer_results]
        cos_mins = [r["cos_min"] for r in layer_results]
        print(f"PSNR={np.mean(psnrs):.1f}±{np.std(psnrs):.1f}, "
              f"cos={np.mean(cos_means):.4f}±{np.std(cos_means):.4f}, "
              f"cos_min={np.min(cos_mins):.4f}")

    # =====================================================================
    # Global analysis: is there useful variance?
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"VARIANCE ANALYSIS: CAN KL-BASED TIERING WORK?")
    print(f"{'='*80}")

    all_psnr = [r["psnr"] for r in all_results]
    all_cos_mean = [r["cos_mean"] for r in all_results]
    all_cos_min = [r["cos_min"] for r in all_results]

    # Split experts into "good" (top 20%) and "bad" (bottom 20%) by output cosine
    n = len(all_results)
    sorted_by_cos = sorted(all_results, key=lambda r: r["cos_mean"])
    n_bottom = max(1, n // 5)
    n_top = n_bottom
    bottom_20 = sorted_by_cos[:n_bottom]
    top_20 = sorted_by_cos[-n_top:]

    print(f"\n  All {n} expert-projection pairs:")
    print(f"    PSNR:        {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"    cos_mean:    {np.mean(all_cos_mean):.4f} ± {np.std(all_cos_mean):.4f}")
    print(f"    cos_min:     {np.mean(all_cos_min):.4f}, absolute min: {np.min(all_cos_min):.4f}")

    print(f"\n  Bottom 20% (worst output quality, n={n_bottom}):")
    print(f"    cos_mean:    {np.mean([r['cos_mean'] for r in bottom_20]):.4f} ± "
          f"{np.std([r['cos_mean'] for r in bottom_20]):.4f}")
    print(f"    PSNR:        {np.mean([r['psnr'] for r in bottom_20]):.2f} ± "
          f"{np.std([r['psnr'] for r in bottom_20]):.2f}")

    print(f"\n  Top 20% (best output quality, n={n_top}):")
    print(f"    cos_mean:    {np.mean([r['cos_mean'] for r in top_20]):.4f} ± "
          f"{np.std([r['cos_mean'] for r in top_20]):.4f}")
    print(f"    PSNR:        {np.mean([r['psnr'] for r in top_20]):.2f} ± "
          f"{np.std([r['psnr'] for r in top_20]):.2f}")

    spread_cos = float(np.std(all_cos_mean))
    spread_psnr = float(np.std(all_psnr))

    print(f"\n  ── VERDICT ──")
    print(f"  Output cosine std: {spread_cos:.4f}")
    print(f"  PSNR std:          {spread_psnr:.2f} dB")

    if spread_cos > 0.02:
        print(f"\n  → Significant output quality variance across experts.")
        print(f"    KL-based tier assignment has potential.")
        print(f"    Top 20% cos={np.mean([r['cos_mean'] for r in top_20]):.4f} vs "
              f"Bottom 20% cos={np.mean([r['cos_mean'] for r in bottom_20]):.4f}")
    elif spread_cos > 0.005:
        print(f"\n  → Weak output quality variance. KL-based tiering MIGHT provide")
        print(f"    marginal benefit, but the spread is small.")
    else:
        print(f"\n  → Output quality is near-uniform across all experts.")
        print(f"    KL-based tier assignment will NOT provide meaningful benefit.")
        print(f"    All experts respond similarly to 4-bit quantization.")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "4bit_output_variance.json")
    summary = {
        "n_total": n,
        "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr)),
                 "min": float(np.min(all_psnr)), "max": float(np.max(all_psnr))},
        "cos_mean": {"mean": float(np.mean(all_cos_mean)), "std": float(np.std(all_cos_mean)),
                     "min": float(np.min(all_cos_mean)), "max": float(np.max(all_cos_mean))},
        "cos_min": {"mean": float(np.mean(all_cos_min)), "min_absolute": float(np.min(all_cos_min))},
        "spread": {"cos_std": spread_cos, "psnr_std": spread_psnr},
        "bottom20_cos_mean": float(np.mean([r["cos_mean"] for r in bottom_20])),
        "top20_cos_mean": float(np.mean([r["cos_mean"] for r in top_20])),
        "per_layer": {},
    }
    for layer in eval_layers:
        lr = [r for r in all_results if r["layer"] == layer]
        if lr:
            summary["per_layer"][str(layer)] = {
                "n": len(lr),
                "psnr_mean": float(np.mean([r["psnr"] for r in lr])),
                "cos_mean": float(np.mean([r["cos_mean"] for r in lr])),
            }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
