#!/usr/bin/env python3
"""Phase D: KL divergence for multiple BLOCKLUT schemes (uniform8/6/4 degradation curve).

Loads the reference logits from Phase A, then runs inference with each BLOCKLUT
variant and computes token-level KL(P_ref || P_variant). Results are saved to JSON.
"""
import argparse, json, os, sys, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from evaluation.profile_tools import clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD_BASE = "/home/hh/zip_Moe/LUT_MoE/offload"
REF_LOGITS_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/ref_logits.npz"
RESULTS_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results"


def run_inference_and_collect_logits(offload_path, lut_name, prompts, tokenizer,
                                      max_prompt_length=512):
    """Run BLOCKLUT inference and return logits dict {prompt_idx: tensor}."""
    lut_path = os.path.join(CHECKPOINT, f"blocklut_{lut_name}.npy")
    if not os.path.exists(lut_path):
        raise FileNotFoundError(f"LUT not found: {lut_path}")

    config = {
        "offload_path": offload_path,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.75, "gpu_pool_ratio": 0.90, "batch_size": 1,
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
    logits_dict = {}

    for i, prompt in enumerate(tqdm(prompts, desc=f"Inference ({os.path.basename(offload_path)})")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            logits_dict[i] = outputs.logits.detach().cpu()

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return logits_dict


def compute_kl_stats(ref_logits, variant_logits, num_prompts):
    """Compute token-level KL divergence and cosine similarity stats."""
    all_kl = []
    all_cos = []
    per_prompt = []

    for i in range(num_prompts):
        if i not in ref_logits or i not in variant_logits:
            continue
        ref = ref_logits[i].float()
        var = variant_logits[i].float()
        min_len = min(ref.shape[1], var.shape[1])
        ref = ref[:, :min_len, :]
        var = var[:, :min_len, :]

        ref_prob = F.softmax(ref, dim=-1)
        var_prob = F.softmax(var, dim=-1)

        kl = F.kl_div(ref_prob.log(), var_prob, reduction='none', log_target=False).sum(dim=-1)
        kl_flat = kl.flatten().numpy()
        all_kl.extend(kl_flat.tolist())

        cs = F.cosine_similarity(ref, var, dim=-1).flatten().numpy()
        all_cos.extend(cs.tolist())

        per_prompt.append({
            "mean_kl": float(np.mean(kl_flat)),
            "median_kl": float(np.median(kl_flat)),
            "max_kl": float(np.max(kl_flat)),
            "n_tokens": len(kl_flat),
        })

    all_kl = np.array(all_kl)
    all_cos = np.array(all_cos)

    return {
        "per_prompt": per_prompt,
        "overall": {
            "mean_kl": float(np.mean(all_kl)),
            "median_kl": float(np.median(all_kl)),
            "p90_kl": float(np.percentile(all_kl, 90)),
            "max_kl": float(np.max(all_kl)),
            "pct_kl_lt_0_001": float(100 * np.mean(all_kl < 0.001)),
            "pct_kl_lt_0_01": float(100 * np.mean(all_kl < 0.01)),
            "pct_kl_lt_0_1": float(100 * np.mean(all_kl < 0.1)),
            "n_tokens": len(all_kl),
        },
        "cosine_similarity": {
            "mean": float(np.mean(all_cos)),
            "median": float(np.median(all_cos)),
            "min": float(np.min(all_cos)),
        },
    }


def main():
    parser = argparse.ArgumentParser("Phase D: KL Degradation Curve")
    parser.add_argument("--ref_path", type=str, default=REF_LOGITS_PATH)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--schemes", type=str,
                        default="uniform8:qwen_blocklut:256,uniform6:qwen_blocklut_uniform6:63,uniform4:qwen_blocklut_uniform4:15",
                        help="Comma-separated name:offload_dir:lut_size")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    # Load reference logits
    print("=" * 60)
    print("  PHASE D: KL Degradation Curve")
    print("=" * 60)
    print(f"\nLoading reference logits from {args.ref_path}...")
    ref_data = np.load(args.ref_path, allow_pickle=True)
    num_prompts = int(ref_data["num_prompts"])
    prompts = list(ref_data["prompts"])

    ref_logits = {}
    for i in range(num_prompts):
        ref_logits[i] = torch.from_numpy(ref_data[f"logits_{i}"])
    print(f"  Loaded {num_prompts} reference logit tensors")
    total_ref_tokens = sum(ref_logits[i].shape[1] for i in range(num_prompts))
    print(f"  Total reference tokens: {total_ref_tokens}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for scheme_spec in args.schemes.split(","):
        parts = scheme_spec.strip().split(":")
        scheme_name = parts[0]
        offload_dir = parts[1]
        lut_name = parts[2] if len(parts) > 2 else "256"

        offload_path = os.path.join(OFFLOAD_BASE, offload_dir)
        print(f"\n{'='*60}")
        print(f"  Scheme: {scheme_name}")
        print(f"  Offload: {offload_path}")
        print(f"  LUT: blocklut_{lut_name}.npy")
        print(f"{'='*60}")

        if not os.path.exists(offload_path):
            print(f"  SKIPPED: offload directory not found")
            continue

        variant_logits = run_inference_and_collect_logits(
            offload_path, lut_name, prompts, tokenizer, args.max_prompt_length)

        stats = compute_kl_stats(ref_logits, variant_logits, num_prompts)
        all_results[scheme_name] = stats

        # Print per-scheme summary
        print(f"\n  --- {scheme_name} KL Summary ---")
        ov = stats["overall"]
        print(f"  Mean KL:   {ov['mean_kl']:.6f}")
        print(f"  Median KL: {ov['median_kl']:.6f}")
        print(f"  P90 KL:    {ov['p90_kl']:.6f}")
        print(f"  Max KL:    {ov['max_kl']:.6f}")
        print(f"  KL < 0.001: {ov['pct_kl_lt_0_001']:.1f}%")
        print(f"  KL < 0.01:  {ov['pct_kl_lt_0_01']:.1f}%")
        print(f"  KL < 0.1:   {ov['pct_kl_lt_0_1']:.1f}%")
        cs = stats["cosine_similarity"]
        print(f"  CosSim mean: {cs['mean']:.6f}, median: {cs['median']:.6f}, min: {cs['min']:.6f}")

        del variant_logits
        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  KL Degradation Curve Summary")
    print(f"{'='*80}")
    print(f"{'Scheme':<16} {'Mean KL':<14} {'Median KL':<14} {'P90 KL':<14} {'CosSim':<14}")
    print(f"{'-'*16} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for sn, st in all_results.items():
        print(f"{sn:<16} {st['overall']['mean_kl']:<14.6f} {st['overall']['median_kl']:<14.6f} "
              f"{st['overall']['p90_kl']:<14.6f} {st['cosine_similarity']['mean']:<14.6f}")

    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        output = {
            "reference_path": args.ref_path,
            "num_prompts": num_prompts,
            "total_ref_tokens": total_ref_tokens,
            "results": all_results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
