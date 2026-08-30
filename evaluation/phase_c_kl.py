#!/usr/bin/env python3
"""Phase C: BLOCKLUT inference → load ref logits from disk, compute KL divergence."""
import argparse, os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from evaluation.profile_tools import clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD_BLOCKLUT = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut"
REF_LOGITS_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/ref_logits.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", type=str, default=REF_LOGITS_PATH)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--offload_path", type=str, default=OFFLOAD_BLOCKLUT)
    parser.add_argument("--lut_name", type=str, default="256")
    parser.add_argument("--scheme_name", type=str, default="BLOCKLUT")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PHASE C: KL Divergence — {args.scheme_name} vs Lossless Reference")
    print("=" * 60)

    # Load reference logits
    print(f"\nLoading reference logits from {args.ref_path}...")
    ref_data = np.load(args.ref_path, allow_pickle=True)
    num_prompts = int(ref_data["num_prompts"])
    prompts = list(ref_data["prompts"])

    ref_logits = {}
    for i in range(num_prompts):
        ref_logits[i] = torch.from_numpy(ref_data[f"logits_{i}"])
    print(f"  Loaded {num_prompts} reference logit tensors")

    # Load BLOCKLUT model
    print(f"\nLoading model with BLOCKLUT codec ({args.scheme_name})...")
    lut_path = os.path.join(CHECKPOINT, f"blocklut_{args.lut_name}.npy")
    config = {
        "offload_path": args.offload_path,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95, "batch_size": 1,
        "code_type": "BLOCKLUT", "lut_path": lut_path,
        "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }

    model = MoE(CHECKPOINT, config)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    blut_logits = {}
    for i, prompt in enumerate(tqdm(prompts, desc="BLOCKLUT inference")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=args.max_prompt_length, return_tensors="pt",
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
    for i in range(num_prompts):
        if i not in ref_logits or i not in blut_logits:
            continue
        ref = ref_logits[i].float()
        blut = blut_logits[i].float()
        min_len = min(ref.shape[1], blut.shape[1])
        ref = ref[:, :min_len, :]
        blut = blut[:, :min_len, :]

        ref_prob = F.softmax(ref, dim=-1)
        blut_prob = F.softmax(blut, dim=-1)

        kl = F.kl_div(ref_prob.log(), blut_prob, reduction='none', log_target=False).sum(dim=-1)
        kl_flat = kl.flatten().numpy()
        all_kl_values.extend(kl_flat.tolist())

        print(f"  {i:<10} {np.mean(kl_flat):<14.6f} {np.median(kl_flat):<14.6f} "
              f"{np.max(kl_flat):<14.6f} {len(kl_flat):<10}")

    all_kl = np.array(all_kl_values)
    print(f"\n  Overall (all tokens):")
    print(f"    Mean KL:   {np.mean(all_kl):.6f}")
    print(f"    Median KL: {np.median(all_kl):.6f}")
    print(f"    P90 KL:    {np.percentile(all_kl, 90):.6f}")
    print(f"    Max KL:    {np.max(all_kl):.6f}")
    print(f"    % tokens with KL < 0.001: {100*np.mean(all_kl < 0.001):.1f}%")
    print(f"    % tokens with KL < 0.01:  {100*np.mean(all_kl < 0.01):.1f}%")
    print(f"    % tokens with KL < 0.1:   {100*np.mean(all_kl < 0.1):.1f}%")

    # Logit cosine similarity
    cos_sims = []
    for i in range(num_prompts):
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

    print(f"\n  Interpretation:")
    if np.mean(all_kl) < 0.001:
        print(f"    KL < 0.001: BLOCKLUT output is essentially identical to lossless")
    elif np.mean(all_kl) < 0.01:
        print(f"    KL < 0.01: very high fidelity, no perceptible difference")
    elif np.mean(all_kl) < 0.1:
        print(f"    KL < 0.1: minor differences, likely semantically equivalent")
    else:
        print(f"    KL > 0.1: noticeable output divergence")

    # Save results to JSON if requested
    if args.output_json:
        import json
        result = {
            "scheme": args.scheme_name,
            "offload_path": args.offload_path,
            "lut_name": args.lut_name,
            "num_prompts": num_prompts,
            "total_tokens": len(all_kl),
            "kl": {
                "mean": float(np.mean(all_kl)),
                "median": float(np.median(all_kl)),
                "p90": float(np.percentile(all_kl, 90)),
                "max": float(np.max(all_kl)),
                "pct_lt_0_001": float(100 * np.mean(all_kl < 0.001)),
                "pct_lt_0_01": float(100 * np.mean(all_kl < 0.01)),
                "pct_lt_0_1": float(100 * np.mean(all_kl < 0.1)),
            },
            "cosine_similarity": {
                "mean": float(np.mean(cos_sims)),
                "median": float(np.median(cos_sims)),
                "min": float(np.min(cos_sims)),
            },
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
