"""PSNR-sensitivity-based mixed-precision BLOCKLUT evaluation.

Strategy: measure per-expert quantization sensitivity via uniform 6-bit PSNR,
then assign tiers by sensitivity (NOT by activation frequency):

  Low PSNR  → "hot"  (sensitive to quantization, needs more bits → 8bit)
  Mid PSNR  → "warm" (medium sensitivity → 6bit)
  High PSNR → "cold" (robust to quantization, can use fewer bits → 4bit)

Compares against frequency-based tier assignment to see which is better.
"""
import argparse, json, os, sys, time, gc
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
SHARED_LUT_SAMPLE_EXPERTS = 30
SENSITIVITY_LUT_BITS = 6        # use 6-bit for sensitivity measurement (good spread)
SENSITIVITY_LUT_CENTROIDS = 63  # 2^6 - 1

SCHEMES = {
    "uniform8":  {"hot": 255, "warm": 255, "cold": 255},
    "uniform6":  {"hot": 63,  "warm": 63,  "cold": 63},
    "uniform5":  {"hot": 31,  "warm": 31,  "cold": 31},
    "uniform4":  {"hot": 15,  "warm": 15,  "cold": 15},
    # PSNR-based: sensitive experts get 8bit, robust get 4bit
    "psnr_865":  {"hot": 255, "warm": 63,  "cold": 31},
    "psnr_864":  {"hot": 255, "warm": 63,  "cold": 15},
    "psnr_854":  {"hot": 255, "warm": 31,  "cold": 15},
    # Frequency-based for comparison
    "freq_865":  {"hot": 255, "warm": 63,  "cold": 31},
    "freq_864":  {"hot": 255, "warm": 63,  "cold": 15},
}

TIER_TOP = 0.2    # top 20% = hot
TIER_MID = 0.5    # 20%-50% = warm; remainder = cold


def effective_bits(centroids):
    return np.log2(centroids + 1) + 16.0 / BLOCK_SIZE


def block_normalize(tensor_f32):
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


def quantize_from_normalized(normalized_np, absmax_np, lut_f32, orig_shape):
    midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
    idx = np.searchsorted(midpoints, normalized_np).astype(np.uint8)
    n_blocks = len(absmax_np)
    reco_norm = lut_f32[idx].reshape(n_blocks, BLOCK_SIZE)
    reco = torch.from_numpy(
        (reco_norm * absmax_np[:, None]).astype(np.float32)
    ).to(torch.bfloat16)
    n_orig = int(np.prod(orig_shape))
    return reco.ravel()[:n_orig].reshape(orig_shape)


def learn_shared_lut(samples_list, n_centroids):
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


def load_frequencies(json_path):
    with open(json_path) as f:
        data = json.load(f)
    freq = {}
    for e in data.get("per_expert", []):
        freq[(e["layer"], e["expert"])] = e.get("token_count", 0)
    return freq


def assign_tiers_by_ranking(scores, num_layers, num_experts, reverse=False):
    """Assign tiers by ranking scores across all experts.

    By default (reverse=False), LOW scores → hot, HIGH scores → cold.
    For PSNR: low PSNR = sensitive → hot (reverse=False).
    For frequency: high freq → hot (reverse=True).
    """
    all_experts = []
    for lid in range(num_layers):
        for eid in range(num_experts):
            all_experts.append(((lid, eid), scores.get((lid, eid), 0)))
    all_experts.sort(key=lambda x: x[1], reverse=reverse)
    n = len(all_experts)
    n_hot = max(1, int(n * TIER_TOP))
    n_warm = max(1, int(n * TIER_MID))

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
    t_total_start = time.perf_counter()

    parser = argparse.ArgumentParser("PSNR-Sensitivity Mixed-Precision BLOCKLUT")
    parser.add_argument("--freq_json", type=str, default="",
                        help="Frequency JSON for freq-based comparison")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip sensitivity pass (use existing psnr_json)")
    parser.add_argument("--psnr_json", type=str, default="",
                        help="Pre-computed PSNR sensitivity JSON")
    args = parser.parse_args()

    model_type = args.model_type
    checkpoint = args.model_dir
    num_layers = List_num_expert_layers[model_type]
    num_experts = List_num_experts[model_type]
    first_k_dense = List_first_k_dense_replace[model_type]

    # Discover safetensor files
    safetensor_files = sorted(
        f for f in os.listdir(checkpoint) if f.endswith(".safetensors")
    )
    print(f"Model: {model_type}, {num_layers} layers × {num_experts} experts")
    print(f"Found {len(safetensor_files)} safetensor files")

    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(checkpoint, trust_remote=True)

    # Count total expert tensors
    n_expert_keys = 0
    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if "expert" in k and "shared_expert" not in k:
                    ml, _ = parse_expert_id(k, model_config)
                    if ml is not None:
                        n_expert_keys += 1
    print(f"Total expert tensors: {n_expert_keys}")

    # ==================================================================
    # Phase 0: Learn reference 6-bit LUT for sensitivity measurement
    # ==================================================================
    print(f"\n{'='*72}")
    print("Phase 0: Learning reference 6-bit LUT for sensitivity measurement")
    print(f"{'='*72}")

    ref_samples = []
    ref_n_collected = 0
    ref_target = 10  # 10 experts enough for reference LUT
    for sf in safetensor_files:
        if ref_n_collected >= ref_target:
            break
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if ref_n_collected >= ref_target:
                    break
                if "expert" not in k or "shared_expert" in k:
                    continue
                ml, _ = parse_expert_id(k, model_config)
                if ml is None:
                    continue
                t = fhandle.get_tensor(k).to(torch.float32)
                norm_np, _, _ = block_normalize(t)
                ref_samples.append(norm_np)
                ref_n_collected += 1
                del t, norm_np

    t0 = time.perf_counter()
    ref_lut = learn_shared_lut(ref_samples, SENSITIVITY_LUT_CENTROIDS)
    del ref_samples
    gc.collect()
    print(f"  Reference LUT: {len(ref_lut)} centroids, {time.perf_counter() - t0:.1f}s")

    # ==================================================================
    # Phase 1: PSNR sensitivity measurement (one pass, one tensor at a time)
    # ==================================================================
    if args.skip_sensitivity and args.psnr_json:
        print(f"\n{'='*72}")
        print("Phase 1: SKIPPED (loading pre-computed PSNR)")
        print(f"{'='*72}")
        with open(args.psnr_json) as f:
            psnr_data = json.load(f)
        per_expert_psnr = {}
        for e in psnr_data.get("per_expert", []):
            per_expert_psnr[(e["layer"], e["expert"])] = e["psnr_db"]
        print(f"  Loaded PSNR for {len(per_expert_psnr)} experts")
    else:
        print(f"\n{'='*72}")
        print(f"Phase 1: Measuring per-expert PSNR sensitivity (uniform 6-bit)")
        print(f"  Processing {n_expert_keys} tensors, one at a time...")
        print(f"{'='*72}")

        per_expert_mse = {}
        per_expert_nelem = {}
        per_expert_maxval = {}

        pbar = tqdm(total=n_expert_keys, desc="Sensitivity")
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

                    original = fhandle.get_tensor(k).to(torch.float32)
                    n_elem = int(original.numel())
                    max_abs = float(original.abs().max())

                    norm_np, absmax_np, orig_shape = block_normalize(original)
                    orig_np = original.numpy().ravel()
                    del original

                    recon = quantize_from_normalized(norm_np, absmax_np, ref_lut, orig_shape)
                    sse = float(np.sum((orig_np - recon.float().numpy().ravel()) ** 2))

                    per_expert_mse[expert_key] = per_expert_mse.get(expert_key, 0.0) + sse
                    per_expert_nelem[expert_key] = per_expert_nelem.get(expert_key, 0) + n_elem
                    per_expert_maxval[expert_key] = max(
                        per_expert_maxval.get(expert_key, 0.0), max_abs)

                    del orig_np, norm_np, absmax_np, recon
                    pbar.update(1)
        pbar.close()

        # Compute per-expert PSNR
        per_expert_psnr = {}
        for expert_key in per_expert_mse:
            sse_sum = per_expert_mse[expert_key]
            nelem = per_expert_nelem[expert_key]
            maxval = per_expert_maxval[expert_key]
            avg_mse = sse_sum / nelem if nelem > 0 else 0
            if avg_mse > 0:
                per_expert_psnr[expert_key] = float(20 * np.log10(maxval / np.sqrt(avg_mse)))
            else:
                per_expert_psnr[expert_key] = 99.0

        del per_expert_mse, per_expert_nelem, per_expert_maxval
        gc.collect()

        psnr_vals = list(per_expert_psnr.values())
        print(f"  PSNR range: [{min(psnr_vals):.2f}, {max(psnr_vals):.2f}] dB")
        print(f"  PSNR mean:  {np.mean(psnr_vals):.2f} dB")
        print(f"  PSNR std:   {np.std(psnr_vals):.2f} dB")

    # ==================================================================
    # Phase 2: Assign tiers by PSNR sensitivity (and by frequency for comparison)
    # ==================================================================
    print(f"\n{'='*72}")
    print("Phase 2: Assigning expert tiers")
    print(f"{'='*72}")

    # PSNR-based: low PSNR = more sensitive → hot
    psnr_tiers = assign_tiers_by_ranking(per_expert_psnr, num_layers, num_experts,
                                         reverse=False)
    n_hot_psnr = sum(1 for t in psnr_tiers.values() if t == "hot")
    n_warm_psnr = sum(1 for t in psnr_tiers.values() if t == "warm")
    n_cold_psnr = sum(1 for t in psnr_tiers.values() if t == "cold")

    # Per-tier PSNR stats
    hot_psnr = [per_expert_psnr[k] for k, t in psnr_tiers.items() if t == "hot"]
    warm_psnr = [per_expert_psnr[k] for k, t in psnr_tiers.items() if t == "warm"]
    cold_psnr = [per_expert_psnr[k] for k, t in psnr_tiers.items() if t == "cold"]
    print(f"PSNR-based tiers:")
    print(f"  hot  ({n_hot_psnr:>4} experts): PSNR range [{min(hot_psnr):.2f}, {max(hot_psnr):.2f}]")
    print(f"  warm ({n_warm_psnr:>4} experts): PSNR range [{min(warm_psnr):.2f}, {max(warm_psnr):.2f}]")
    print(f"  cold ({n_cold_psnr:>4} experts): PSNR range [{min(cold_psnr):.2f}, {max(cold_psnr):.2f}]")

    # Frequency-based tiers for comparison
    freq_tiers = None
    if args.freq_json:
        freq = load_frequencies(args.freq_json)
        total_freq_tokens = sum(freq.values())
        print(f"\nFrequency data loaded: {total_freq_tokens:,} tokens, "
              f"{sum(1 for v in freq.values() if v > 0)} active experts")
        freq_tiers = assign_tiers_by_ranking(freq, num_layers, num_experts, reverse=True)
        n_hot_freq = sum(1 for t in freq_tiers.values() if t == "hot")
        n_warm_freq = sum(1 for t in freq_tiers.values() if t == "warm")
        n_cold_freq = sum(1 for t in freq_tiers.values() if t == "cold")
        print(f"Frequency-based tiers: hot={n_hot_freq}, warm={n_warm_freq}, cold={n_cold_freq}")

        # Tier overlap analysis
        overlap_hot = sum(1 for k in psnr_tiers if psnr_tiers[k] == "hot" and freq_tiers.get(k) == "hot")
        overlap_cold = sum(1 for k in psnr_tiers if psnr_tiers[k] == "cold" and freq_tiers.get(k) == "cold")
        print(f"PSNR-hot ∩ Freq-hot: {overlap_hot}/{n_hot_psnr}")
        print(f"PSNR-cold ∩ Freq-cold: {overlap_cold}/{n_cold_psnr}")
    else:
        freq = {}
        print("No frequency JSON — skipping freq-based comparison")

    # ==================================================================
    # Phase 3: Collect block-normalized samples for shared LUTs
    # ==================================================================
    print(f"\n{'='*72}")
    print(f"Phase 3: Collecting samples for shared LUTs "
          f"(up to {SHARED_LUT_SAMPLE_EXPERTS} experts/tier)")
    print(f"{'='*72}")

    # Collect all unique (tier_system, tier, n_centroids) combinations
    lut_specs = set()
    # PSNR-based schemes
    for sn in ["psnr_865", "psnr_864", "psnr_854", "uniform6", "uniform5", "uniform4", "uniform8"]:
        if sn in SCHEMES:
            scheme = SCHEMES[sn]
            for tier in ["hot", "warm", "cold"]:
                lut_specs.add(("psnr", tier, scheme[tier]))
    # Freq-based schemes
    if freq_tiers:
        for sn in ["freq_865", "freq_864"]:
            if sn in SCHEMES:
                scheme = SCHEMES[sn]
                for tier in ["hot", "warm", "cold"]:
                    lut_specs.add(("freq", tier, scheme[tier]))

    # Group unique (tier, n_centroids) — LUT doesn't care about psnr vs freq
    unique_lut_specs = set()
    for _, tier, n_cent in lut_specs:
        unique_lut_specs.add((tier, n_cent))

    # Collect samples for each unique (tier, n_centroids) combination
    # We need separate samples for psnr-tiers and freq-tiers
    psnr_tier_samples = {"hot": [], "warm": [], "cold": []}
    freq_tier_samples = {"hot": [], "warm": [], "cold": []}
    psnr_tier_count = {"hot": 0, "warm": 0, "cold": 0}
    freq_tier_count = {"hot": 0, "warm": 0, "cold": 0}
    tier_target = SHARED_LUT_SAMPLE_EXPERTS

    psnr_done = all(v >= tier_target for v in psnr_tier_count.values())
    freq_done = freq_tiers is None or all(v >= tier_target for v in freq_tier_count.values())

    for sf in safetensor_files:
        if psnr_done and freq_done:
            break
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if psnr_done and freq_done:
                    break
                if "expert" not in k or "shared_expert" in k:
                    continue
                model_layer, expert_id = parse_expert_id(k, model_config)
                if model_layer is None or expert_id is None:
                    continue
                moe_layer = model_layer - first_k_dense
                expert_key = (moe_layer, expert_id)

                # PSNR-based sampling
                psnr_tier = psnr_tiers.get(expert_key, "cold")
                if psnr_tier_count[psnr_tier] < tier_target:
                    t = fhandle.get_tensor(k).to(torch.float32)
                    norm_np, _, _ = block_normalize(t)
                    psnr_tier_samples[psnr_tier].append(norm_np)
                    psnr_tier_count[psnr_tier] += 1
                    del t, norm_np

                # Freq-based sampling
                if freq_tiers:
                    freq_tier = freq_tiers.get(expert_key, "cold")
                    if freq_tier_count[freq_tier] < tier_target:
                        t = fhandle.get_tensor(k).to(torch.float32)
                        norm_np, _, _ = block_normalize(t)
                        freq_tier_samples[freq_tier].append(norm_np)
                        freq_tier_count[freq_tier] += 1
                        del t, norm_np

                psnr_done = all(v >= tier_target for v in psnr_tier_count.values())
                freq_done = freq_tiers is None or all(v >= tier_target for v in freq_tier_count.values())

    for tier in ["hot", "warm", "cold"]:
        print(f"  PSNR-{tier}: {psnr_tier_count[tier]} experts sampled, "
              f"{sum(s.size for s in psnr_tier_samples[tier]):,} values")
        if freq_tiers:
            print(f"  Freq-{tier}: {freq_tier_count[tier]} experts sampled, "
                  f"{sum(s.size for s in freq_tier_samples[tier]):,} values")

    # ==================================================================
    # Phase 4: Learn shared LUTs
    # ==================================================================
    print(f"\n{'='*72}")
    print("Phase 4: Learning shared per-tier LUTs")
    print(f"{'='*72}")

    shared_luts = {}
    for tier, n_cent in sorted(unique_lut_specs, key=lambda x: x[1], reverse=True):
        # Try PSNR samples first, fall back to freq samples
        samples = psnr_tier_samples.get(tier, [])
        if not samples and freq_tiers:
            samples = freq_tier_samples.get(tier, [])
        if not samples:
            print(f"  WARNING: no samples for {tier}/{n_cent}, skipping")
            continue
        t0 = time.perf_counter()
        lut = learn_shared_lut(samples, n_cent)
        elapsed = time.perf_counter() - t0
        shared_luts[(tier, n_cent)] = lut
        eff_bits = effective_bits(n_cent)
        print(f"  LUT-{n_cent:>3} ({tier:>4}, ~{eff_bits:.1f} bit): "
              f"{len(lut)} centroids, {elapsed:.1f}s")

    del psnr_tier_samples, freq_tier_samples
    gc.collect()

    # ==================================================================
    # Phase 5: Full evaluation — PSNR-based + freq-based + uniform baselines
    # ==================================================================
    print(f"\n{'='*72}")
    print("Phase 5: Full evaluation (one tensor at a time)")
    print(f"{'='*72}")

    scheme_names = ["uniform8", "uniform6", "uniform5", "uniform4",
                    "psnr_865", "psnr_864", "psnr_854"]
    if freq_tiers:
        scheme_names += ["freq_865", "freq_864"]

    stats = {sn: {"sse": defaultdict(float), "nelem": defaultdict(int),
                  "maxval": defaultdict(float)}
             for sn in scheme_names}

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

                original = fhandle.get_tensor(k).to(torch.float32)
                n_elem = int(original.numel())
                max_abs = float(original.abs().max())

                norm_np, absmax_np, orig_shape = block_normalize(original)
                orig_np = original.numpy().ravel()
                del original

                for sn in scheme_names:
                    scheme = SCHEMES[sn]
                    if sn.startswith("psnr_"):
                        tier = psnr_tiers.get(expert_key, "cold")
                    elif sn.startswith("freq_"):
                        tier = freq_tiers.get(expert_key, "cold") if freq_tiers else "cold"
                    else:
                        tier = "hot"  # uniform schemes use same LUT everywhere

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

    # ==================================================================
    # Results
    # ==================================================================
    print(f"\n{'='*100}")
    header = (f"{'Scheme':<16} {'Freq-w PSNR':>12} {'Unw PSNR':>10} {'Avg bits':>10}  "
              f"{'Hot PSNR':>10} {'Warm PSNR':>10} {'Cold PSNR':>10}  {'Tier':>6}")
    print(header)
    print(f"{'-'*100}")

    results = {}
    for sn in scheme_names:
        st = stats[sn]
        if not st["sse"]:
            continue

        per_expert_psnr_eval = {}
        for expert_key in st["sse"]:
            sse_sum = st["sse"][expert_key]
            nelem = st["nelem"][expert_key]
            maxval = st["maxval"][expert_key]
            avg_mse = sse_sum / nelem if nelem > 0 else 0
            if avg_mse > 0:
                per_expert_psnr_eval[expert_key] = float(20 * np.log10(maxval / np.sqrt(avg_mse)))
            else:
                per_expert_psnr_eval[expert_key] = 99.0

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

        unweighted = float(np.mean(list(per_expert_psnr_eval.values()))) if per_expert_psnr_eval else 0.0

        # Average bits
        total_elems = sum(st["nelem"].values())
        avg_bits = 0.0
        for expert_key, nelem in st["nelem"].items():
            if sn.startswith("psnr_"):
                tier = psnr_tiers.get(expert_key, "cold")
            elif sn.startswith("freq_"):
                tier = freq_tiers.get(expert_key, "cold") if freq_tiers else "cold"
            else:
                tier = "hot"
            n_cent = SCHEMES[sn][tier]
            avg_bits += effective_bits(n_cent) * nelem
        if total_elems > 0:
            avg_bits /= total_elems

        # Per-tier PSNR
        if sn.startswith("psnr_"):
            tier_system = psnr_tiers
            tier_label = "PSNR"
        elif sn.startswith("freq_"):
            tier_system = freq_tiers if freq_tiers else {}
            tier_label = "FREQ"
        else:
            tier_system = {}
            tier_label = "UNIF"

        tier_psnr_eval = {"hot": [], "warm": [], "cold": []}
        for expert_key, psnr_val in per_expert_psnr_eval.items():
            t = tier_system.get(expert_key, "hot") if tier_system else "hot"
            tier_psnr_eval[t].append(psnr_val)

        hot_str = f"{np.mean(tier_psnr_eval['hot']):.2f}" if tier_psnr_eval["hot"] else "N/A"
        warm_str = f"{np.mean(tier_psnr_eval['warm']):.2f}" if tier_psnr_eval["warm"] else "N/A"
        cold_str = f"{np.mean(tier_psnr_eval['cold']):.2f}" if tier_psnr_eval["cold"] else "N/A"

        print(f"{sn:<16} {freq_weighted:>12.2f} {unweighted:>10.2f} {avg_bits:>10.2f}  "
              f"{hot_str:>10} {warm_str:>10} {cold_str:>10}  {tier_label:>6}")

        results[sn] = {
            "freq_weighted_psnr": freq_weighted,
            "unweighted_psnr": unweighted,
            "avg_bits_per_elem": avg_bits,
            "tier_psnr": {t: float(np.mean(v)) if v else None
                          for t, v in tier_psnr_eval.items()},
            "n_experts": len(per_expert_psnr_eval),
            "tier_system": tier_label,
        }

    # ==================================================================
    # Key comparisons
    # ==================================================================
    print(f"\n{'='*100}")
    print("KEY COMPARISONS (PSNR-sensitivity vs Frequency-based):")
    print(f"{'='*100}")

    if "psnr_864" in results and "freq_864" in results:
        pu = results["psnr_864"]["unweighted_psnr"]
        fu = results["freq_864"]["unweighted_psnr"]
        delta = pu - fu
        winner = "PSNR-based" if delta > 0 else "Freq-based"
        print(f"  psnr_864 unweighted PSNR: {pu:.2f} dB")
        print(f"  freq_864 unweighted PSNR: {fu:.2f} dB")
        print(f"  Delta: {delta:+.2f} dB → {winner} WINS for overall quality")

    if "psnr_864" in results and "uniform6" in results:
        pu = results["psnr_864"]["unweighted_psnr"]
        u6 = results["uniform6"]["unweighted_psnr"]
        pb = results["psnr_864"]["avg_bits_per_elem"]
        u6b = results["uniform6"]["avg_bits_per_elem"]
        print(f"\n  psnr_864: {pu:.2f} dB @ {pb:.2f} bits")
        print(f"  uniform6: {u6:.2f} dB @ {u6b:.2f} bits")
        print(f"  psnr_864 saves {u6b-pb:.2f} bits vs uniform6, PSNR delta: {pu-u6:+.2f} dB")

    # Overlap analysis
    if freq_tiers:
        print(f"\n  Tier overlap analysis:")
        for tier_name in ["hot", "warm", "cold"]:
            psnr_set = {k for k, t in psnr_tiers.items() if t == tier_name}
            freq_set = {k for k, t in freq_tiers.items() if t == tier_name}
            overlap = len(psnr_set & freq_set)
            print(f"    {tier_name}: PSNR={len(psnr_set)}, Freq={len(freq_set)}, overlap={overlap} "
                  f"({overlap/max(len(psnr_set),1)*100:.0f}%)")

    elapsed_total = time.perf_counter() - t_total_start
    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")

    # ==================================================================
    # Save output
    # ==================================================================
    if not args.output_json:
        from evaluation.profile_tools import get_current_datatime
        args.output_json = (
            f"/home/hh/zip_Moe/LUT_MoE/evaluation/results/"
            f"psnr_sensitive-{model_type}-{get_current_datatime()}.json"
        )
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    output = {
        "model_type": model_type,
        "tier_split": {"hot_pct": TIER_TOP, "warm_pct": TIER_MID - TIER_TOP,
                       "cold_pct": 1 - TIER_MID},
        "psnr_sensitivity": {
            "mean": float(np.mean(list(per_expert_psnr.values()))),
            "std": float(np.std(list(per_expert_psnr.values()))),
            "min": float(np.min(list(per_expert_psnr.values()))),
            "max": float(np.max(list(per_expert_psnr.values()))),
        },
        "results": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
