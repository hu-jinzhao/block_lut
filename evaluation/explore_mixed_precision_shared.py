"""Mixed-precision BLOCKLUT with SHARED per-tier LUTs + 4-bit cold experts.

Key changes from explore_mixed_precision.py:
  1. Shared LUT per tier (not per-expert) — deployable, K-means only ~5 times total
  2. Cold tier pushed to 4-bit (15 centroids) in new mixed schemes
  3. Two-phase: collect samples → learn shared LUTs → single evaluation pass

All safetensor processing is incremental — at most one tensor in memory at a time.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from collections import defaultdict
from safetensors import safe_open
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import (
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)
from utils.hf_config import parse_expert_id

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BLOCK_SIZE = 128
KMEANS_SAMPLE = 50000
KMEANS_BATCH = 2048
SHARED_LUT_SAMPLE_EXPERTS = 30  # experts per tier for LUT training

# bit allocation: {tier: n_centroids}
# 255=~8bit, 127=~7bit, 63=~6bit, 31=~5bit, 15=~4bit
SCHEMES = {
    "uniform8":  {"hot": 255, "warm": 255, "cold": 255},
    "uniform7":  {"hot": 127, "warm": 127, "cold": 127},
    "uniform6":  {"hot": 63,  "warm": 63,  "cold": 63},
    "uniform5":  {"hot": 31,  "warm": 31,  "cold": 31},
    "uniform4":  {"hot": 15,  "warm": 15,  "cold": 15},
    "mixed_865": {"hot": 255, "warm": 63,  "cold": 31},   # 8/6/5
    "mixed_864": {"hot": 255, "warm": 63,  "cold": 15},   # 8/6/4
    "mixed_854": {"hot": 255, "warm": 31,  "cold": 15},   # 8/5/4
    "mixed_765": {"hot": 127, "warm": 63,  "cold": 31},   # 7/6/5
    "mixed_764": {"hot": 127, "warm": 63,  "cold": 15},   # 7/6/4
}

FREQ_TOP = 0.2
FREQ_MID = 0.5


def effective_bits(centroids):
    return np.log2(centroids + 1) + 16.0 / BLOCK_SIZE


def block_normalize(tensor_f32):
    """Block128 absmax normalize. Returns (normalized_np, absmax_np, orig_shape).

    normalized_np is a flat float32 numpy array (already padded to block boundary).
    absmax_np is float32 numpy array of shape (n_blocks,).
    These are reused across all LUT quantizations — computed once per tensor.
    """
    flat = tensor_f32.ravel()
    n = flat.numel()
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = n_blocks * BLOCK_SIZE
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n)])
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normalized = (blocks / absmax.unsqueeze(1)).ravel().numpy().astype(np.float32)
    return normalized, absmax.numpy().astype(np.float32), tensor_f32.shape


def learn_shared_lut(samples_list, n_centroids):
    """Learn a shared LUT from pooled block-normalized samples across experts."""
    all_data = np.concatenate([s.ravel() for s in samples_list])
    if len(all_data) > KMEANS_SAMPLE:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_data), KMEANS_SAMPLE, replace=False)
        data = all_data[idx].reshape(-1, 1)
    else:
        data = all_data.reshape(-1, 1)

    km = MiniBatchKMeans(n_clusters=n_centroids, random_state=42,
                         batch_size=KMEANS_BATCH, n_init=1, max_iter=30)
    km.fit(data)
    centroids = km.cluster_centers_.ravel().astype(np.float32)
    centroids.sort()
    return centroids


def quantize_from_normalized(normalized_np, absmax_np, lut_f32, orig_shape):
    """Quantize from pre-computed block-normalized data using LUT.

    normalized_np: flat float32 [n_padded]
    absmax_np: float32 [n_blocks]
    lut_f32: sorted centroids float32 [n_centroids]
    orig_shape: original tensor shape
    Returns reconstructed bf16 tensor.
    """
    midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
    idx = np.searchsorted(midpoints, normalized_np).astype(np.uint8)
    n_blocks = len(absmax_np)
    reco_norm = lut_f32[idx].reshape(n_blocks, BLOCK_SIZE)
    reco = torch.from_numpy(
        (reco_norm * absmax_np[:, None]).astype(np.float32)
    ).to(torch.bfloat16)
    n_orig = int(np.prod(orig_shape))
    return reco.ravel()[:n_orig].reshape(orig_shape)


def load_frequencies(json_path):
    with open(json_path) as f:
        data = json.load(f)
    freq = {}
    for e in data.get("per_expert", []):
        freq[(e["layer"], e["expert"])] = e.get("token_count", 0)
    return freq


def assign_tiers(freq, num_layers, num_experts):
    all_experts = []
    for lid in range(num_layers):
        for eid in range(num_experts):
            all_experts.append(((lid, eid), freq.get((lid, eid), 0)))
    all_experts.sort(key=lambda x: x[1], reverse=True)
    n = len(all_experts)
    n_hot = max(1, int(n * FREQ_TOP))
    n_warm = max(1, int(n * FREQ_MID))

    tiers = {}
    for i, (key, _) in enumerate(all_experts):
        if i < n_hot:
            tiers[key] = "hot"
        elif i < n_warm:
            tiers[key] = "warm"
        else:
            tiers[key] = "cold"
    return tiers


def main():
    parser = argparse.ArgumentParser("Mixed-Precision Shared-LUT BLOCKLUT")
    parser.add_argument("--freq_json", type=str, default="",
                        help="Frequency JSON from freq_weighted_psnr.py")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--schemes", type=str,
                        default="uniform8,uniform6,uniform4,mixed_865,mixed_864",
                        help="Comma-separated scheme names")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    model_type = args.model_type
    checkpoint = args.model_dir
    num_layers = List_num_expert_layers[model_type]
    num_experts = List_num_experts[model_type]
    first_k_dense = List_first_k_dense_replace[model_type]
    scheme_names = [s.strip() for s in args.schemes.split(",")]
    for sn in scheme_names:
        if sn not in SCHEMES:
            raise ValueError(f"Unknown scheme: {sn}")

    # ---- Load frequency data ----
    if args.freq_json:
        freq = load_frequencies(args.freq_json)
        total_freq_tokens = sum(freq.values())
        print(f"Loaded frequencies from {args.freq_json}")
        print(f"  Total tokens: {total_freq_tokens:,}")
        print(f"  Active experts: {sum(1 for v in freq.values() if v > 0)}")
    else:
        print("No frequency JSON — using uniform frequency")
        freq = {}

    tiers = assign_tiers(freq, num_layers, num_experts)
    n_hot = sum(1 for t in tiers.values() if t == "hot")
    n_warm = sum(1 for t in tiers.values() if t == "warm")
    n_cold = sum(1 for t in tiers.values() if t == "cold")
    print(f"Tier assignment: hot={n_hot}, warm={n_warm}, cold={n_cold}")

    # ---- Discover safetensor files ----
    safetensor_files = sorted(
        f for f in os.listdir(checkpoint) if f.endswith(".safetensors")
    )
    print(f"Found {len(safetensor_files)} safetensor files")

    # ---- Load model config ----
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(checkpoint, trust_remote=True)

    # ============================================================
    # Phase 1: Collect block-normalized samples per tier
    # ============================================================
    print(f"\nPhase 1: Collecting samples for shared LUT training "
          f"(up to {SHARED_LUT_SAMPLE_EXPERTS} experts/tier)...")

    tier_samples = {"hot": [], "warm": [], "cold": []}
    tier_expert_count = {"hot": 0, "warm": 0, "cold": 0}
    tier_target = SHARED_LUT_SAMPLE_EXPERTS
    all_done = False

    for sf in safetensor_files:
        if all_done:
            break
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if all_done:
                    break
                if "expert" not in k or "shared_expert" in k:
                    continue
                model_layer, expert_id = parse_expert_id(k, model_config)
                if model_layer is None or expert_id is None:
                    continue
                moe_layer = model_layer - first_k_dense
                tier = tiers.get((moe_layer, expert_id), "cold")

                if tier_expert_count[tier] >= tier_target:
                    continue

                t = fhandle.get_tensor(k).to(torch.float32)
                norm_np, _, _ = block_normalize(t)
                tier_samples[tier].append(norm_np)
                tier_expert_count[tier] += 1
                del t, norm_np

                if all(v >= tier_target for v in tier_expert_count.values()):
                    all_done = True

    for tier in ["hot", "warm", "cold"]:
        print(f"  {tier}: {tier_expert_count[tier]} experts sampled, "
              f"{sum(s.size for s in tier_samples[tier]):,} total values")

    # ============================================================
    # Phase 2: Learn shared LUTs
    # ============================================================
    print("\nPhase 2: Learning shared per-tier LUTs...")

    # Collect unique (tier, n_centroids) pairs across all schemes
    lut_specs = set()
    for sn in scheme_names:
        scheme = SCHEMES[sn]
        for tier in ["hot", "warm", "cold"]:
            lut_specs.add((tier, scheme[tier]))

    shared_luts = {}
    for tier, n_cent in sorted(lut_specs, key=lambda x: x[1], reverse=True):
        if not tier_samples[tier]:
            print(f"  WARNING: no samples for {tier}, skipping LUT-{n_cent}")
            continue
        t0 = time.perf_counter()
        lut = learn_shared_lut(tier_samples[tier], n_cent)
        elapsed = time.perf_counter() - t0
        shared_luts[(tier, n_cent)] = lut
        eff_bits = effective_bits(n_cent)
        print(f"  LUT-{n_cent:>3} ({tier:>4}, ~{eff_bits:.1f} bit): "
              f"{len(lut)} centroids, {elapsed:.1f}s")

    # Free sample data
    del tier_samples

    # ---- Load global LUT for comparison ----
    global_lut_path = os.path.join(checkpoint, "blocklut_256.npy")
    global_lut_f32 = None
    if os.path.exists(global_lut_path):
        global_lut_f32 = np.sort(np.load(global_lut_path).astype(np.float32))
        print(f"\nLoaded global LUT: {len(global_lut_f32)} entries")

    # ============================================================
    # Phase 3: Single-pass evaluation using shared LUTs
    # ============================================================
    print("\nPhase 3: Evaluating all schemes with shared LUTs...")

    stats = {sn: {"sse": defaultdict(float), "nelem": defaultdict(int),
                  "maxval": defaultdict(float)}
             for sn in scheme_names}
    if global_lut_f32 is not None:
        stats["global"] = {"sse": defaultdict(float), "nelem": defaultdict(int),
                           "maxval": defaultdict(float)}

    # Count tensors for progress bar
    n_expert_keys = 0
    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if "expert" in k and "shared_expert" not in k:
                    ml, _ = parse_expert_id(k, model_config)
                    if ml is not None:
                        n_expert_keys += 1

    print(f"Processing {n_expert_keys} expert tensors...")
    pbar = tqdm(total=n_expert_keys, desc="Evaluating")

    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                model_layer, expert_id = parse_expert_id(k, model_config)
                if model_layer is None or expert_id is None:
                    continue
                moe_layer = model_layer - first_k_dense
                expert_key = (moe_layer, expert_id)
                tier = tiers.get(expert_key, "cold")

                original = fhandle.get_tensor(k).to(torch.float32)
                n_elem = int(original.numel())
                max_abs = float(original.abs().max())

                # Block-normalize once, reuse across all schemes
                norm_np, absmax_np, orig_shape = block_normalize(original)
                orig_np = original.numpy().ravel()
                del original

                # Global LUT baseline
                if global_lut_f32 is not None:
                    recon_g = quantize_from_normalized(norm_np, absmax_np, global_lut_f32, orig_shape)
                    sse_g = float(np.sum((orig_np - recon_g.float().numpy().ravel()) ** 2))
                    stats["global"]["sse"][expert_key] += sse_g
                    stats["global"]["nelem"][expert_key] += n_elem
                    stats["global"]["maxval"][expert_key] = max(
                        stats["global"]["maxval"][expert_key], max_abs)
                    del recon_g

                # Evaluate each scheme using shared LUT
                for sn in scheme_names:
                    scheme = SCHEMES[sn]
                    n_centroids = scheme[tier]
                    lut_key = (tier, n_centroids)
                    if lut_key not in shared_luts:
                        continue
                    lut = shared_luts[lut_key]

                    recon = quantize_from_normalized(norm_np, absmax_np, lut, orig_shape)
                    sse = float(np.sum((orig_np - recon.float().numpy().ravel()) ** 2))
                    stats[sn]["sse"][expert_key] += sse
                    stats[sn]["nelem"][expert_key] += n_elem
                    stats[sn]["maxval"][expert_key] = max(
                        stats[sn]["maxval"][expert_key], max_abs)
                    del recon

                del orig_np, norm_np, absmax_np
                pbar.update(1)
    pbar.close()

    # ============================================================
    # Results
    # ============================================================
    print(f"\n{'='*100}")
    print(f"{'Scheme':<16} {'Freq-w PSNR':>12} {'Unw PSNR':>10} {'Avg bits':>10}  "
          f"{'Hot PSNR':>10} {'Warm PSNR':>10} {'Cold PSNR':>10}")
    print(f"{'-'*100}")

    results = {}
    all_names = scheme_names + (["global"] if global_lut_f32 is not None else [])

    for sn in all_names:
        st = stats[sn]
        if not st["sse"]:
            continue

        per_expert_psnr = {}
        for expert_key in st["sse"]:
            sse_sum = st["sse"][expert_key]
            nelem = st["nelem"][expert_key]
            maxval = st["maxval"][expert_key]
            avg_mse = sse_sum / nelem if nelem > 0 else 0
            if avg_mse > 0:
                per_expert_psnr[expert_key] = float(20 * np.log10(maxval / np.sqrt(avg_mse)))
            else:
                per_expert_psnr[expert_key] = 99.0

        # Frequency-weighted PSNR
        weighted_sse = 0.0
        total_weight = 0.0
        global_maxval = 0.0
        for expert_key in st["sse"]:
            w = freq.get(expert_key, 1)
            sse_sum = st["sse"][expert_key]
            nelem = st["nelem"][expert_key]
            avg_mse = sse_sum / nelem if nelem > 0 else 0
            weighted_sse += w * avg_mse
            total_weight += w
            global_maxval = max(global_maxval, st["maxval"][expert_key])

        if total_weight > 0 and weighted_sse > 0:
            freq_weighted = float(20 * np.log10(global_maxval / np.sqrt(weighted_sse / total_weight)))
        else:
            freq_weighted = 99.0

        unweighted = np.mean(list(per_expert_psnr.values())) if per_expert_psnr else 0.0

        # Average bits
        total_elems = sum(st["nelem"].values())
        avg_bits = 0.0
        for expert_key, nelem in st["nelem"].items():
            tier = tiers.get(expert_key, "cold")
            if sn == "global":
                n_cent = 255
            else:
                n_cent = SCHEMES[sn][tier]
            avg_bits += effective_bits(n_cent) * nelem
        if total_elems > 0:
            avg_bits /= total_elems

        # Per-tier PSNR
        tier_psnr = {"hot": [], "warm": [], "cold": []}
        for expert_key, psnr_val in per_expert_psnr.items():
            tier_psnr[tiers.get(expert_key, "cold")].append(psnr_val)

        hot_str = f"{np.mean(tier_psnr['hot']):.2f}" if tier_psnr["hot"] else "N/A"
        warm_str = f"{np.mean(tier_psnr['warm']):.2f}" if tier_psnr["warm"] else "N/A"
        cold_str = f"{np.mean(tier_psnr['cold']):.2f}" if tier_psnr["cold"] else "N/A"

        print(f"{sn:<16} {freq_weighted:>12.2f} {unweighted:>10.2f} {avg_bits:>10.2f}  "
              f"{hot_str:>10} {warm_str:>10} {cold_str:>10}")

        results[sn] = {
            "freq_weighted_psnr": freq_weighted,
            "unweighted_psnr": unweighted,
            "avg_bits_per_elem": avg_bits,
            "tier_psnr": {t: float(np.mean(v)) if v else None
                          for t, v in tier_psnr.items()},
            "n_experts": len(per_expert_psnr),
        }

    # ---- Summary ----
    print(f"\n{'='*100}")
    print("Key comparisons:")
    print("  mixed_864 vs mixed_865: 4-bit vs 5-bit cold — does 4-bit cold hurt?")
    print("  mixed_864 vs uniform6:  similar avg bits (~6.0) — does hot=8bit help?")
    print("  mixed_864 vs uniform8:  25% bandwidth savings at what quality cost?")
    print(f"{'='*100}")

    # ---- Save ----
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        output = {
            "model_type": model_type,
            "tier_split": {"hot_pct": FREQ_TOP, "warm_pct": FREQ_MID - FREQ_TOP,
                           "cold_pct": 1 - FREQ_MID},
            "shared_lut": True,
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
