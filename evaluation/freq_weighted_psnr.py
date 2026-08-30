# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
Frequency-weighted PSNR — Layer 1 fast diagnostic for mixed-precision expert allocation.

Collects per-expert activation frequencies via calibration prompts, computes
per-expert PSNR by simulating the codec round-trip on original weights, then
ranks experts by frequency × quality. Outputs a report that guides which experts
can lose bits (cold) and which need full precision (hot).
"""

import argparse
import json
import math
import os
import sys
import time
import gc
import numpy as np
import torch

from safetensors import safe_open
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entry.llm_modeling import MoE
from evaluation.profile_tools import (
    sample_first_prompts,
    clear_model_cache,
    MEMMAPPING,
    DISPLAY_NAME,
    get_current_datatime,
)
from utils.constants import *
from utils.hf_config import parse_expert_id


def simulate_quantize_dequant(weight, codec_type, lut_sorted, blocklut_sorted):
    """Simulate quantize→dequantize round-trip. Returns fp32 numpy array."""
    if codec_type in ("LZ4", "LZ4HC", "ZSTD"):
        return weight.detach().to(torch.float32).numpy().ravel()

    x = weight.detach().to(torch.float32).numpy()

    if codec_type == "LUT":
        flat_u16 = weight.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
        flat_mono = np.where(
            flat_u16 & 0x8000, ~flat_u16, flat_u16 ^ np.uint16(0x8000)
        ).astype(np.uint16)
        midpoints = (lut_sorted[:-1] + lut_sorted[1:]) / 2.0
        mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
        mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
        thresholds = np.where(
            mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)
        ).astype(np.uint16)
        indices = np.searchsorted(thresholds, flat_mono).astype(np.uint8)
        return lut_sorted[indices].reshape(x.shape).ravel()

    if codec_type == "BLOCKLUT":
        x_flat = x.ravel()
        n = x_flat.size
        bs = 128
        nb = (n + bs - 1) // bs
        pad = nb * bs - n
        if pad > 0:
            x_flat = np.pad(x_flat, (0, pad))
        blocks = x_flat.reshape(nb, bs)
        absmax_vals = np.max(np.abs(blocks), axis=1)
        absmax_vals = np.maximum(absmax_vals, 1e-12)
        normalized = blocks / absmax_vals[:, np.newaxis]
        midpoints = (blocklut_sorted[:-1] + blocklut_sorted[1:]) / 2.0
        indices = np.searchsorted(midpoints, normalized.ravel()).astype(np.uint8)
        reconstructed_norm = blocklut_sorted[indices].reshape(nb, bs)
        reconstructed_flat = (reconstructed_norm * absmax_vals[:, np.newaxis]).ravel()
        return reconstructed_flat[:x.size]

    raise ValueError(f"Unknown codec: {codec_type}")


def mse_to_psnr(mse, max_val):
    """Convert MSE to PSNR (dB). Returns inf if mse == 0."""
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def psnr_to_mse(psnr_db, max_val):
    """Convert PSNR (dB) back to MSE."""
    if psnr_db == float("inf"):
        return 0.0
    return (max_val / (10.0 ** (psnr_db / 20.0))) ** 2


def main():
    parser = argparse.ArgumentParser("Frequency-Weighted PSNR Evaluation")
    parser.add_argument(
        "--model_type", type=str, default="deepseek",
        choices=["deepseek", "qwen", "switch"]
    )
    parser.add_argument("--memory_footprint", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument(
        "--code_type", type=str, default="LUT",
        choices=["LZ4HC", "LZ4", "ZSTD", "LUT", "BLOCKLUT"]
    )
    parser.add_argument("--lut_path", type=str, default="")
    parser.add_argument("--cache_algorithm", type=str, default="LFU")
    parser.add_argument("--prefetcher_topk", type=int, default=4)
    parser.add_argument("--gpu_pool_ratio", type=float, default=0.95)
    parser.add_argument("--SSD_type", type=str, default="Samsung970EVO")
    parser.add_argument(
        "--dataset_path", type=str,
        default="/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"
    )
    parser.add_argument(
        "--num_calibration_prompts", type=int, default=20,
        help="Number of prompts for collecting activation frequencies"
    )
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--offload_path", type=str, default="",
                        help="Override default offload path (default: offload/{model_type}/)")

    args = parser.parse_args()

    target = args.model_type
    checkpoint = f"/home/hh/zip_Moe/LUT_MoE/models/{target}/"
    offload_path = args.offload_path if args.offload_path else f"/home/hh/zip_Moe/LUT_MoE/offload/{target}/"

    config = {
        "offload_path": offload_path,
        "caching_algorithm": args.cache_algorithm,
        "prefetcher_topk": args.prefetcher_topk,
        "device_memory_ratio": MEMMAPPING["LUT_MoE"][args.memory_footprint],
        "gpu_pool_ratio": args.gpu_pool_ratio,
        "batch_size": args.batch_size,
        "code_type": args.code_type,
        "lut_path": args.lut_path,
        "hyperparam_state_margin": 0.1,
        "num_file_chunks": 3,
        "num_compute_threads": 6,
        "trace_path": f"/home/hh/zip_Moe/LUT_MoE/trace/{target}_trace.pt",
        "expert_topk": List_expert_topk[target],
        "num_elements_per_expert": List_num_elements_per_expert[target],
        "num_tensors_per_expert": List_num_tensors_per_expert[target],
        "num_expert_layers": List_num_expert_layers[target],
        "num_experts": List_num_experts[target],
        "first_k_dense_replace": List_first_k_dense_replace[target],
    }

    # ── Step 1: Load model ──────────────────────────────────────────
    print("[Step 1/4] Loading model...")
    model = MoE(checkpoint, config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote=True)

    if "qwen" in args.model_type.lower():
        tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    custom_kwargs = {}
    if "switch" in args.model_type.lower():
        custom_kwargs = {"decoder_start_token_id": 0}
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif "qwen" in args.model_type.lower():
        custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
    elif "deepseek" in args.model_type.lower():
        custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}

    # ── Step 2: Collect activation frequencies ─────────────────────
    print(f"[Step 2/4] Collecting activation frequencies "
          f"({args.num_calibration_prompts} prompts)...")

    executor = model.engine.expert_executor
    executor.enable_freq_accum()

    prompts = sample_first_prompts(
        args.dataset_path,
        num_samples=args.num_calibration_prompts,
        seed=args.seed,
        max_candidates=500,
    )

    freq_accum = {}

    for prompt in tqdm(prompts, desc="Calibrating"):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=args.max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            model.generate(
                inputs.input_ids,
                max_new_tokens=args.max_new_tokens,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                **custom_kwargs,
            )

        # Merge this prompt's frequencies into the global dict before clearing
        batch_freqs = executor.get_freq_accum()
        if batch_freqs:
            for key, cnt in batch_freqs.items():
                freq_accum[key] = freq_accum.get(key, 0) + cnt
        executor.reset_freq_accum()

        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    executor.disable_freq_accum()
    total_tokens = sum(freq_accum.values())
    print(f"  Total tokens routed: {total_tokens:,}")
    print(f"  Total tokens routed: {total_tokens:,}")

    # ── Step 3: Compute per-expert PSNR ────────────────────────────
    print("[Step 3/4] Computing per-expert PSNR from safetensors...")

    lut_sorted = None
    blocklut_sorted = None
    if args.code_type == "LUT":
        lut_sorted = model.engine.lut_sorted
    elif args.code_type == "BLOCKLUT":
        blocklut_sorted = model.engine.blocklut_sorted

    num_layers = List_num_expert_layers[target]
    num_experts = List_num_experts[target]
    first_k_dense = List_first_k_dense_replace[target]
    model_config = model.model_config

    safetensor_files = sorted(
        f for f in os.listdir(checkpoint) if f.endswith(".safetensors")
    )

    # Per-expert metrics — computed incrementally to avoid holding all tensors in memory
    per_expert_mse = {}   # (layer, expert) -> total_mse
    per_expert_nelem = {}  # (layer, expert) -> total elements
    per_expert_maxval = {}  # (layer, expert) -> max |val|

    # Count total expert tensors for progress bar
    n_expert_keys = 0
    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    model_layer, _ = parse_expert_id(k, model_config)
                    if model_layer is not None:
                        n_expert_keys += 1

    # Stream-process: read one tensor at a time, compute MSE, release immediately
    pbar = tqdm(total=n_expert_keys, desc="PSNR per expert")
    for sf in safetensor_files:
        sf_path = os.path.join(checkpoint, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                model_layer, expert_id = parse_expert_id(k, model_config)
                if model_layer is None or expert_id is None:
                    continue
                moe_layer = model_layer - first_k_dense
                key = (moe_layer, expert_id)

                original = f.get_tensor(k).to(torch.float32)
                recon = simulate_quantize_dequant(
                    original, args.code_type, lut_sorted, blocklut_sorted
                )
                orig_fp32 = original.numpy().ravel()
                n = orig_fp32.size
                mse = float(np.sum((orig_fp32 - recon) ** 2))

                per_expert_mse[key] = per_expert_mse.get(key, 0.0) + mse
                per_expert_nelem[key] = per_expert_nelem.get(key, 0) + n
                per_expert_maxval[key] = max(
                    per_expert_maxval.get(key, 0.0),
                    float(np.max(np.abs(orig_fp32)))
                )

                del original, recon, orig_fp32
                pbar.update(1)
    pbar.close()

    # ── Step 4: Aggregate & report ─────────────────────────────────
    print("[Step 4/4] Computing frequency-weighted metrics...\n")

    # Build expert analysis table
    experts = []
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            key = (layer_id, expert_id)
            freq = freq_accum.get(key, 0)
            mse = per_expert_mse.get(key, None)
            nelem = per_expert_nelem.get(key, None)
            maxval = per_expert_maxval.get(key, None)

            if mse is not None and nelem is not None and nelem > 0:
                avg_mse = mse / nelem
                psnr = mse_to_psnr(avg_mse, maxval)
            elif args.code_type in ("LZ4", "LZ4HC", "ZSTD"):
                psnr = float("inf")
                avg_mse = 0.0
                maxval = 1.0
            else:
                continue  # expert not found in safetensors

            experts.append({
                "layer": layer_id,
                "expert": expert_id,
                "token_count": freq,
                "freq_ratio": freq / total_tokens if total_tokens > 0 else 0,
                "psnr_db": psnr,
                "mse": avg_mse,
                "maxval": maxval,
            })

    # ── Frequency-weighted PSNR (correct MSE-domain computation) ───
    lossy = args.code_type in ("LUT", "BLOCKLUT")

    if lossy:
        weighted_mse_sum = 0.0
        total_weight = 0.0
        global_maxval = 0.0

        for e in experts:
            w = e["token_count"]
            if w == 0:
                continue
            weighted_mse_sum += w * e["mse"]
            total_weight += w
            global_maxval = max(global_maxval, e["maxval"])

        freq_weighted_psnr = (
            mse_to_psnr(weighted_mse_sum / total_weight, global_maxval)
            if total_weight > 0 else float("inf")
        )
        unweighted_psnr_vals = [e["psnr_db"] for e in experts
                                if e["psnr_db"] != float("inf")]
        unweighted_psnr = np.mean(unweighted_psnr_vals) if unweighted_psnr_vals else float("inf")
    else:
        freq_weighted_psnr = float("inf")
        unweighted_psnr = float("inf")

    # ── Hot vs cold analysis ─────────────────────────────────────
    active = sorted(
        [e for e in experts if e["token_count"] > 0],
        key=lambda x: x["token_count"], reverse=True,
    )
    n_hot = max(1, len(active) // 5)  # top 20%
    hot = active[:n_hot]
    cold = active[n_hot:]

    hot_psnr_vals = [e["psnr_db"] for e in hot if e["psnr_db"] != float("inf")]
    cold_psnr_vals = [e["psnr_db"] for e in cold if e["psnr_db"] != float("inf")]

    # ── Print report ──────────────────────────────────────────────
    print("=" * 72)
    print("  FREQUENCY-WEIGHTED PSNR EVALUATION")
    print("=" * 72)
    print(f"  Model:       {DISPLAY_NAME.get(args.model_type, args.model_type)}")
    print(f"  Codec:       {args.code_type}")
    print(f"  Calibration: {args.num_calibration_prompts} prompts, "
          f"max {args.max_new_tokens} new tokens each")
    print(f"  Total tokens routed: {total_tokens:,}")
    print("-" * 72)

    if lossy:
        print(f"  Unweighted PSNR (avg):          {unweighted_psnr:.2f} dB")
        print(f"  Frequency-weighted PSNR:        {freq_weighted_psnr:.2f} dB")
        delta = freq_weighted_psnr - unweighted_psnr
        direction = "hot experts have better PSNR → favorable" if delta > 0 else \
                     "cold experts have better PSNR → need per-expert tuning"
        print(f"  Delta (weighted - unweighted):  {delta:+.2f} dB  ({direction})")
        print("-" * 72)
        print(f"  Hot experts (top 20% by freq, {len(hot)} experts):")
        print(f"    Avg freq:  {np.mean([e['token_count'] for e in hot]):.1f}")
        print(f"    Freq sum:  {sum(e['token_count'] for e in hot):,} "
              f"({sum(e['token_count'] for e in hot)/total_tokens*100:.1f}%)")
        if hot_psnr_vals:
            print(f"    Avg PSNR:  {np.mean(hot_psnr_vals):.2f} dB")
        print(f"  Cold experts (bottom 80%, {len(cold)} experts):")
        print(f"    Avg freq:  {np.mean([e['token_count'] for e in cold]):.1f}")
        print(f"    Freq sum:  {sum(e['token_count'] for e in cold):,} "
              f"({sum(e['token_count'] for e in cold)/total_tokens*100:.1f}%)")
        if cold_psnr_vals:
            print(f"    Avg PSNR:  {np.mean(cold_psnr_vals):.2f} dB")

        # Recommendation
        if hot_psnr_vals and cold_psnr_vals:
            hot_avg = np.mean(hot_psnr_vals)
            cold_avg = np.mean(cold_psnr_vals)
            if hot_avg < cold_avg - 0.5:
                print(f"\n  ⚠ Hot experts have WORSE PSNR than cold. "
                      f"Consider improving hot-expert precision first.")
            elif cold_avg < hot_avg - 0.5:
                print(f"\n  ✓ Hot experts already have better PSNR. "
                      f"Cold experts are candidates for reduced precision.")
            else:
                print(f"\n  → Hot/cold PSNR are similar. "
                      f"Bit reallocation may have limited benefit.")

    # ── Per-layer table ──────────────────────────────────────────
    print("-" * 72)
    print(f"\n  Per-Layer Summary:")
    header = f"  {'Layer':<8} {'Tokens':<10} {'ActiveExp':<11} {'AvgPSNR':<10}"
    print(header)
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*10}")

    for layer_id in range(num_layers):
        layer_experts = [e for e in experts if e["layer"] == layer_id]
        lyr_tokens = sum(e["token_count"] for e in layer_experts)
        lyr_active = sum(1 for e in layer_experts if e["token_count"] > 0)
        lyr_psnr = [e["psnr_db"] for e in layer_experts
                     if e["psnr_db"] != float("inf")]
        psnr_str = f"{np.mean(lyr_psnr):.2f}" if lyr_psnr else "inf"
        print(f"  {layer_id:<8} {lyr_tokens:<10,} {lyr_active:<11} {psnr_str:<10}")

    # ── Top-20 and Bottom-10 by frequency ────────────────────────
    print(f"\n  Top-20 Most Active Experts:")
    print(f"  {'Layer':<8} {'Expert':<8} {'Tokens':<10} {'Freq%':<9} {'PSNR':<10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")
    for e in active[:20]:
        psnr_str = f"{e['psnr_db']:.2f}" if e['psnr_db'] != float("inf") else "inf"
        print(f"  {e['layer']:<8} {e['expert']:<8} {e['token_count']:<10,} "
              f"{e['freq_ratio']*100:<9.3f} {psnr_str:<10}")

    print(f"\n  Bottom-10 Least Active (but still used) Experts:")
    print(f"  {'Layer':<8} {'Expert':<8} {'Tokens':<10} {'Freq%':<9} {'PSNR':<10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")
    for e in active[-10:]:
        psnr_str = f"{e['psnr_db']:.2f}" if e['psnr_db'] != float("inf") else "inf"
        print(f"  {e['layer']:<8} {e['expert']:<8} {e['token_count']:<10,} "
              f"{e['freq_ratio']*100:<9.3f} {psnr_str:<10}")

    print("=" * 72 + "\n")

    # ── Save JSON report ─────────────────────────────────────────
    if not args.output_file:
        args.output_file = (
            f"/home/hh/zip_Moe/LUT_MoE/evaluation/results/"
            f"freq_psnr-{args.model_type}-{args.code_type}-"
            f"{get_current_datatime()}.json"
        )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Convert inf to None for JSON
    def sanitize(v):
        return None if v == float("inf") else v

    report = {
        "model_type": args.model_type,
        "code_type": args.code_type,
        "num_calibration_prompts": args.num_calibration_prompts,
        "max_new_tokens": args.max_new_tokens,
        "total_tokens": total_tokens,
        "unweighted_psnr_db": sanitize(unweighted_psnr),
        "frequency_weighted_psnr_db": sanitize(freq_weighted_psnr),
        "hot_experts_top20pct": {
            "count": len(hot),
            "avg_freq": float(np.mean([e["token_count"] for e in hot])) if hot else 0,
            "freq_share": sum(e["token_count"] for e in hot) / total_tokens if total_tokens > 0 else 0,
            "avg_psnr_db": sanitize(float(np.mean(hot_psnr_vals))) if hot_psnr_vals else None,
        },
        "cold_experts_bottom80pct": {
            "count": len(cold),
            "avg_freq": float(np.mean([e["token_count"] for e in cold])) if cold else 0,
            "freq_share": sum(e["token_count"] for e in cold) / total_tokens if total_tokens > 0 else 0,
            "avg_psnr_db": sanitize(float(np.mean(cold_psnr_vals))) if cold_psnr_vals else None,
        },
        "per_expert": [
            {
                "layer": e["layer"],
                "expert": e["expert"],
                "token_count": e["token_count"],
                "freq_ratio": e["freq_ratio"],
                "psnr_db": sanitize(e["psnr_db"]),
            }
            for e in experts
        ],
    }

    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {args.output_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    print("Done.")
    os._exit(0)


if __name__ == "__main__":
    main()
