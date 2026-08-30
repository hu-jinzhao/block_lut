#!/usr/bin/env python3
"""
KL divergence evaluation: compare model output with compressed vs lossless weights.

Phase A: LZ4HC lossless run → save final logits (ground truth)
Phase B: Weight-space PSNR for multiple compression levels (sample-based, fast)
Phase C: Re-load with BLOCKLUT uniform8 → compute KL(P_lossless || P_blocklut)

The core metric is Phase C's token-level KL divergence on final logits.
"""

import argparse, json, math, os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entry.llm_modeling import MoE
from evaluation.profile_tools import sample_first_prompts, clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)
from utils.hf_config import parse_expert_id

BLOCK_SIZE = 128
CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD = "/home/hh/zip_Moe/LUT_MoE/offload/qwen"
OFFLOAD_BLOCKLUT = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut"
DATASET = "/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"


def block128_uniform_quantize(weight_f32, n_levels):
    """Block128 absmax + uniform quantization → dequantize. Returns fp32 array in original shape."""
    orig_shape = weight_f32.shape
    flat = weight_f32.ravel()
    n = flat.size
    nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    pad = nb * BLOCK_SIZE - n
    if pad > 0:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(nb, BLOCK_SIZE)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    norm = blocks / absmax[:, np.newaxis]

    # Uniform levels in [-1, 1]
    levels = np.linspace(-1.0, 1.0, n_levels, dtype=np.float32)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    idx = np.searchsorted(midpoints, norm.ravel()).astype(np.uint8)
    recon = levels[idx].reshape(nb, BLOCK_SIZE) * absmax[:, np.newaxis]
    return recon.ravel()[:n].reshape(orig_shape).astype(np.float32)


def block128_lut_quantize(weight_f32, lut):
    """Block128 absmax + LUT codebook quantization → dequantize. Returns fp32 array in original shape."""
    orig_shape = weight_f32.shape
    flat = weight_f32.ravel()
    n = flat.size
    nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    pad = nb * BLOCK_SIZE - n
    if pad > 0:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(nb, BLOCK_SIZE)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    norm = blocks / absmax[:, np.newaxis]

    midpoints = (lut[:-1] + lut[1:]) / 2.0
    idx = np.searchsorted(midpoints, norm.ravel()).astype(np.uint8)
    recon = lut[idx].reshape(nb, BLOCK_SIZE) * absmax[:, np.newaxis]
    return recon.ravel()[:n].reshape(orig_shape).astype(np.float32)


def psnr_var(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(np.max(np.abs(orig)) / math.sqrt(mse))


def make_model_config(code_type, lut_path=""):
    return {
        "offload_path": OFFLOAD,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95, "batch_size": 1,
        "code_type": code_type, "lut_path": lut_path,
        "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": f"/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }


def run_phase_a(prompts, max_prompt_length):
    """Phase A: Lossless inference → collect reference logits."""
    print("=" * 70)
    print("  PHASE A: Collecting lossless reference logits")
    print("=" * 70)

    model = MoE(CHECKPOINT, make_model_config("LZ4HC"))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    ref_logits = {}
    for i, prompt in enumerate(tqdm(prompts, desc="Lossless inference")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            ref_logits[i] = outputs.logits.detach().cpu()

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return ref_logits, tokenizer


def run_phase_b(num_samples=200):
    """Phase B: Weight-space PSNR for multiple compression levels (sample-based)."""
    print("\n" + "=" * 70)
    print("  PHASE B: Weight-space PSNR across compression levels")
    print("=" * 70)

    model_cfg = AutoConfig.from_pretrained(CHECKPOINT, trust_remote=True)
    safetensor_files = sorted(f for f in os.listdir(CHECKPOINT) if f.endswith(".safetensors"))
    lut_256 = np.load(os.path.join(CHECKPOINT, "blocklut_256.npy"))

    # Collect all expert tensor keys
    all_keys = []
    for sf in safetensor_files:
        sf_path = os.path.join(CHECKPOINT, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    ml, eid = parse_expert_id(k, model_cfg)
                    if ml is not None and eid is not None:
                        all_keys.append((sf_path, k))

    # Sample
    rng = np.random.RandomState(42)
    sampled = rng.choice(len(all_keys), min(num_samples, len(all_keys)), replace=False)
    print(f"  Sampling {len(sampled)}/{len(all_keys)} expert tensors...")

    # Compression schemes
    schemes = {
        "LUT-255 (=uniform8)": lambda w: block128_lut_quantize(w, lut_256),
        "Uniform 6-bit (64 levels)": lambda w: block128_uniform_quantize(w, 63),
        "Uniform 5-bit (32 levels)": lambda w: block128_uniform_quantize(w, 31),
        "Uniform 4-bit (16 levels)": lambda w: block128_uniform_quantize(w, 15),
    }

    scheme_psnrs = defaultdict(list)
    for idx in tqdm(sampled, desc="PSNR evaluation"):
        sf_path, key = all_keys[idx]
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            W = f.get_tensor(key).float().numpy()

        for name, quant_fn in schemes.items():
            W_rec = quant_fn(W)
            scheme_psnrs[name].append(psnr_var(W, W_rec))

    print(f"\n  {'Scheme':<30} {'Mean PSNR':<12} {'P10':<10} {'P50':<10} {'P90':<10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name in schemes:
        vals = scheme_psnrs[name]
        print(f"  {name:<30} {np.mean(vals):<12.2f} "
              f"{np.percentile(vals,10):<10.2f} {np.percentile(vals,50):<10.2f} {np.percentile(vals,90):<10.2f}")

    return scheme_psnrs


def run_phase_c(ref_logits, tokenizer, prompts, max_prompt_length):
    """Phase C: BLOCKLUT inference → compute KL vs lossless reference."""
    print("\n" + "=" * 70)
    print("  PHASE C: KL Divergence — BLOCKLUT uniform8 vs LZ4HC (lossless)")
    print("=" * 70)
    print("  Loading model with BLOCKLUT codec...")

    lut_path = os.path.join(CHECKPOINT, "blocklut_256.npy")
    cfg = make_model_config("BLOCKLUT", lut_path)
    cfg["offload_path"] = OFFLOAD_BLOCKLUT
    model = MoE(CHECKPOINT, cfg)

    blut_logits = {}
    for i, prompt in enumerate(tqdm(prompts, desc="BLOCKLUT inference")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            blut_logits[i] = outputs.logits.detach().cpu()

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Compute KL divergence ──
    print(f"\n  Token-level KL divergence (P_lossless || P_blocklut):")
    print(f"  {'Prompt':<10} {'Mean KL':<14} {'Median KL':<14} {'Max KL':<14} {'Tokens':<10}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")

    all_kl_values = []
    for i in range(len(prompts)):
        if i not in ref_logits or i not in blut_logits:
            continue
        ref = ref_logits[i].float()
        blut = blut_logits[i].float()
        min_len = min(ref.shape[1], blut.shape[1])
        ref = ref[:, :min_len, :]
        blut = blut[:, :min_len, :]

        ref_prob = F.softmax(ref, dim=-1)
        blut_prob = F.softmax(blut, dim=-1)

        # Per-token KL
        kl = F.kl_div(ref_prob.log(), blut_prob, reduction='none', log_target=False).sum(dim=-1)
        kl_flat = kl.flatten().numpy()
        all_kl_values.extend(kl_flat.tolist())

        print(f"  {i:<10} {np.mean(kl_flat):<14.6f} {np.median(kl_flat):<14.6f} "
              f"{np.max(kl_flat):<14.6f} {len(kl_flat):<10}")

    all_kl = np.array(all_kl_values)
    print(f"\n  {'─'*60}")
    print(f"  Overall (all tokens):")
    print(f"    Mean KL:   {np.mean(all_kl):.6f}")
    print(f"    Median KL: {np.median(all_kl):.6f}")
    print(f"    P90 KL:    {np.percentile(all_kl, 90):.6f}")
    print(f"    Max KL:    {np.max(all_kl):.6f}")
    print(f"    % tokens with KL < 0.001: {100*np.mean(all_kl < 0.001):.1f}%")
    print(f"    % tokens with KL < 0.01:  {100*np.mean(all_kl < 0.01):.1f}%")
    print(f"    % tokens with KL < 0.1:   {100*np.mean(all_kl < 0.1):.1f}%")

    # ── Also compute logit cosine similarity (more intuitive) ──
    cos_sims = []
    for i in range(len(prompts)):
        if i not in ref_logits or i not in blut_logits:
            continue
        ref = ref_logits[i].float()
        blut = blut_logits[i].float()
        min_len = min(ref.shape[1], blut.shape[1])
        ref = ref[:, :min_len, :]
        blut = blut[:, :min_len, :]
        cs = F.cosine_similarity(ref, blut, dim=-1).flatten().numpy()
        cos_sims.extend(cs.tolist())

    cos_sims = np.array(cos_sims)
    print(f"\n  Logit cosine similarity:")
    print(f"    Mean:   {np.mean(cos_sims):.6f}")
    print(f"    Median: {np.median(cos_sims):.6f}")
    print(f"    Min:    {np.min(cos_sims):.6f}")

    # ── Interpretation ──
    print(f"\n  Interpretation:")
    if np.mean(all_kl) < 0.001:
        print(f"    ✓ KL < 0.001: BLOCKLUT output is essentially identical to lossless")
    elif np.mean(all_kl) < 0.01:
        print(f"    ✓ KL < 0.01: very high fidelity, no perceptible difference")
    elif np.mean(all_kl) < 0.1:
        print(f"    ~ KL < 0.1: minor differences, likely semantically equivalent")
    else:
        print(f"    ✗ KL > 0.1: noticeable output divergence")

    return all_kl, cos_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_phase_b", action="store_true")
    parser.add_argument("--skip_phase_c", action="store_true")
    args = parser.parse_args()

    prompts = sample_first_prompts(DATASET, num_samples=args.num_prompts,
                                   seed=args.seed, max_candidates=500)

    # Phase A: Lossless reference
    ref_logits, tokenizer = run_phase_a(prompts, args.max_prompt_length)

    # Phase B: Weight-space PSNR comparison
    if not args.skip_phase_b:
        run_phase_b(num_samples=200)

    # Phase C: Actual KL divergence
    if not args.skip_phase_c:
        run_phase_c(ref_logits, tokenizer, prompts, args.max_prompt_length)

    print("\nDone.")


if __name__ == "__main__":
    main()
