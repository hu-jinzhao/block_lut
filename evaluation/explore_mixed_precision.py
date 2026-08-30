"""Mixed-precision per-expert BLOCKLUT evaluation.

Assigns LUT size by activation frequency: hot experts get more bits,
cold experts get fewer bits. Frequency-weighted PSNR vs uniform baselines.

All safetensor processing is incremental — at most one tensor in memory at a time.
All schemes are evaluated in a SINGLE pass through the data.
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
GLOBAL_LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/models/qwen/blocklut_256.npy"
BLOCK_SIZE = 128
KMEANS_SAMPLE = 50000
KMEANS_BATCH = 2048

# bit allocation: {tier: n_centroids}
SCHEMES = {
    "uniform8":  {"hot": 255, "warm": 255, "cold": 255},
    "uniform7":  {"hot": 127, "warm": 127, "cold": 127},
    "uniform6":  {"hot": 63,  "warm": 63,  "cold": 63},
    "uniform5":  {"hot": 31,  "warm": 31,  "cold": 31},
    "uniform4":  {"hot": 15,  "warm": 15,  "cold": 15},
    "mixed_865": {"hot": 255, "warm": 63,  "cold": 31},
    "mixed_866": {"hot": 255, "warm": 63,  "cold": 63},
    "mixed_855": {"hot": 255, "warm": 31,  "cold": 31},
    "mixed_765": {"hot": 127, "warm": 63,  "cold": 31},
}

FREQ_TOP = 0.2    # top 20% = hot
FREQ_MID = 0.5    # 20%-50% = warm; remainder = cold 50%


def effective_bits(centroids):
    return np.log2(centroids + 1) + 16.0 / BLOCK_SIZE


def learn_lut(tensor_f32, n_centroids, sample_size=KMEANS_SAMPLE):
    flat = tensor_f32.ravel()
    n = flat.numel()
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = n_blocks * BLOCK_SIZE
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n)])
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normalized = (blocks / absmax.unsqueeze(1)).ravel().numpy()

    if len(normalized) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(normalized), sample_size, replace=False)
        data = normalized[idx].reshape(-1, 1)
    else:
        data = normalized.reshape(-1, 1)

    km = MiniBatchKMeans(n_clusters=n_centroids, random_state=42,
                         batch_size=KMEANS_BATCH, n_init=1, max_iter=30)
    km.fit(data)
    centroids = km.cluster_centers_.ravel().astype(np.float32)
    centroids.sort()
    return centroids


def quantize_blocklut(tensor_f32, lut_f32):
    flat = tensor_f32.ravel()
    n = flat.numel()
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = n_blocks * BLOCK_SIZE
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n)])
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normalized = blocks / absmax.unsqueeze(1)

    midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
    idx = np.searchsorted(midpoints, normalized.ravel().numpy()).astype(np.uint8)
    reco_norm = lut_f32[idx].reshape(n_blocks, BLOCK_SIZE)
    reco = torch.from_numpy(
        (reco_norm * absmax.unsqueeze(1).numpy()).astype(np.float32)
    ).to(torch.bfloat16)
    return reco.ravel()[:tensor_f32.numel()].reshape(tensor_f32.shape)


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

    hot_cutoff = all_experts[n_hot - 1][1] if n_hot < n else 0
    warm_cutoff = all_experts[n_warm - 1][1] if n_warm < n else 0
    return tiers, hot_cutoff, warm_cutoff


def main():
    parser = argparse.ArgumentParser("Mixed-Precision Per-Expert BLOCKLUT")
    parser.add_argument("--freq_json", type=str, default="",
                        help="Frequency JSON from freq_weighted_psnr.py")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_type", type=str, default="qwen")
    parser.add_argument("--schemes", type=str,
                        default="uniform8,uniform6,uniform5,mixed_865",
                        help="Comma-separated scheme names")
    parser.add_argument("--max_experts", type=int, default=0,
                        help="Limit number of experts (0=all)")
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

    tiers, hot_cutoff, warm_cutoff = assign_tiers(freq, num_layers, num_experts)
    n_hot = sum(1 for t in tiers.values() if t == "hot")
    n_warm = sum(1 for t in tiers.values() if t == "warm")
    n_cold = sum(1 for t in tiers.values() if t == "cold")
    print(f"Tier assignment: hot={n_hot}, warm={n_warm}, cold={n_cold}")
    if args.freq_json:
        print(f"  Hot cutoff: >= {hot_cutoff} tokens")
        print(f"  Warm cutoff: >= {warm_cutoff} tokens")

    # ---- Load global LUT ----
    global_lut_f32 = None
    if os.path.exists(GLOBAL_LUT_PATH):
        global_lut_f32 = np.sort(np.load(GLOBAL_LUT_PATH).astype(np.float32))
        print(f"Loaded global LUT: {len(global_lut_f32)} entries")

    # ---- Discover safetensor files ----
    safetensor_files = sorted(
        f for f in os.listdir(checkpoint) if f.endswith(".safetensors")
    )
    print(f"Found {len(safetensor_files)} safetensor files")

    # ---- Load model config ----
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(checkpoint, trust_remote=True)

    # ---- Single-pass evaluation ----
    # For each scheme: per_expert_mse[scheme_name][expert_key] = sum_of_squared_errors
    # We accumulate sum-of-squared-errors, total elements, and max |val| per expert
    stats = {sn: {"sse": defaultdict(float), "nelem": defaultdict(int),
                  "maxval": defaultdict(float)}
             for sn in scheme_names}
    stats["global"] = {"sse": defaultdict(float), "nelem": defaultdict(int),
                       "maxval": defaultdict(float)}

    # LUT cache: {(layer, expert, n_centroids): lut_f32}
    lut_cache = {}

    # Count expert tensors for progress bar (lightweight: just key iteration)
    n_expert_keys = 0
    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if "expert" in k and "shared_expert" not in k:
                    ml, _ = parse_expert_id(k, model_config)
                    if ml is not None:
                        n_expert_keys += 1
    if args.max_experts > 0:
        n_expert_keys = min(n_expert_keys, args.max_experts * 3)

    print(f"Processing {n_expert_keys} expert tensors across all schemes...")
    pbar = tqdm(total=n_expert_keys, desc="Processing")
    processed_experts = set()

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

                if args.max_experts > 0 and len(processed_experts) >= args.max_experts:
                    if expert_key not in processed_experts:
                        continue

                tier = tiers.get(expert_key, "cold")

                # Load tensor once
                original = fhandle.get_tensor(k).to(torch.float32)
                orig_np = original.numpy().ravel()
                n_elem = orig_np.size
                max_abs = float(np.abs(orig_np).max())

                # Global LUT baseline (if available)
                if global_lut_f32 is not None:
                    recon_g = quantize_blocklut(original, global_lut_f32)
                    sse_g = float(np.sum((orig_np - recon_g.float().numpy().ravel()) ** 2))
                    stats["global"]["sse"][expert_key] += sse_g
                    stats["global"]["nelem"][expert_key] += n_elem
                    stats["global"]["maxval"][expert_key] = max(
                        stats["global"]["maxval"][expert_key], max_abs)
                    del recon_g

                # Evaluate each scheme
                for sn in scheme_names:
                    scheme = SCHEMES[sn]
                    n_centroids = scheme[tier]
                    cache_key = (moe_layer, expert_id, n_centroids)

                    if cache_key in lut_cache:
                        lut = lut_cache[cache_key]
                    else:
                        lut = learn_lut(original, n_centroids)
                        lut_cache[cache_key] = lut

                    recon = quantize_blocklut(original, lut)
                    sse = float(np.sum((orig_np - recon.float().numpy().ravel()) ** 2))
                    stats[sn]["sse"][expert_key] += sse
                    stats[sn]["nelem"][expert_key] += n_elem
                    stats[sn]["maxval"][expert_key] = max(
                        stats[sn]["maxval"][expert_key], max_abs)
                    del recon

                processed_experts.add(expert_key)
                del original, orig_np
                pbar.update(1)

                if args.max_experts > 0 and len(processed_experts) >= args.max_experts:
                    break

        if args.max_experts > 0 and len(processed_experts) >= args.max_experts:
            break
    pbar.close()

    # ---- Compute per-scheme metrics ----
    print(f"\n{'='*90}")
    print(f"{'Scheme':<16} {'Freq-w PSNR':>12} {'Unw PSNR':>10} {'Avg bits':>10}  "
          f"{'Hot PSNR':>10} {'Warm PSNR':>10} {'Cold PSNR':>10}")
    print(f"{'-'*90}")

    results = {}

    all_names = scheme_names + (["global"] if global_lut_f32 is not None else [])
    for sn in all_names:
        st = stats[sn]
        if not st["sse"]:
            continue

        # Per-expert PSNR
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

        # Average bit rate
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
    print(f"\n{'='*90}")
    print("Key insight: compare mixed_865 vs uniform6 (similar avg bits) or vs uniform5")
    print("If mixed_865 PSNR >> uniform6 PSNR → mixed precision is beneficial")
    print(f"{'='*90}")

    # ---- Save output ----
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        output = {
            "model_type": model_type,
            "tier_split": {"hot_pct": FREQ_TOP, "warm_pct": FREQ_MID - FREQ_TOP,
                           "cold_pct": 1 - FREQ_MID},
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
