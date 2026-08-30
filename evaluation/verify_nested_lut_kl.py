#!/usr/bin/env python3
"""
Combined verification: nested LUT KL divergence + expert activation frequency tracking.

Phase A: LZ4HC lossless inference on prompts.
  - Captures expert activation frequencies (per prompt)
  - Captures router gate logits (per-token expert probabilities)
  - Saves final logits as reference

Phase B: BLOCKLUT uniform8 inference on the SAME prompts.
  - Saves final logits

Phase C: Compute KL(P_lossless || P_blocklut) per prompt + overall.

Phase D: Report expert activation frequencies per prompt, compare across prompts.

Output: JSON result file + NPZ data.
"""

import argparse
import json
import os
import sys
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entry.llm_modeling import MoE
from evaluation.profile_tools import sample_first_prompts, clear_model_cache, get_current_datatime
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD_LOSSLESS = "/home/hh/zip_Moe/LUT_MoE/offload/qwen"
OFFLOAD_BLOCKLUT = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut"
DATASET = "/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"
RESULTS_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results"

# Global storage for router gate hooks
router_data = {}  # layer_id -> {"probs": [tensor, ...]}  (per token group)


def capture_router_hook(module, input, output, layer_id):
    """Forward hook on Qwen2MoEBlock.gate to capture router probabilities."""
    probs = F.softmax(output.detach().cpu().float(), dim=-1)  # [batch*seq, 60]
    if layer_id not in router_data:
        router_data[layer_id] = {"probs": [], "topk_idx": []}
    router_data[layer_id]["probs"].append(probs)
    topk_idx = torch.topk(probs, k=4, dim=-1).indices
    router_data[layer_id]["topk_idx"].append(topk_idx)


def make_model_config(code_type, lut_path="", offload_path=OFFLOAD_LOSSLESS):
    return {
        "offload_path": offload_path,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95, "batch_size": 1,
        "code_type": code_type, "lut_path": lut_path,
        "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }


def register_router_hooks(model):
    """Register forward hooks on all MoE gate layers."""
    global router_data
    router_data = {}
    moe_blocks = []
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "Qwen2MoEBlock":
            moe_blocks.append((name, module))
            layer_id = module.layer_id
            module.gate.register_forward_hook(
                lambda m, inp, out, lid=layer_id: capture_router_hook(m, inp, out, lid)
            )
    return len(moe_blocks)


def run_inference(model, tokenizer, prompts, max_prompt_length,
                  executor=None, track_freq=False):
    """
    Run forward-pass inference on prompts (prefill only).
    Returns dict prompt_idx -> logits tensor (cpu) + per-prompt freq snapshots.

    Uses model.model() forward pass like phase_a_save.py / phase_c_kl.py.
    freq_accum is populated during MoE dispatch within the forward pass,
    so one call captures both logits AND expert activation frequencies.
    """
    logits_dict = {}
    prompt_freqs = {}  # prompt_idx -> freq_accum snapshot

    for i, prompt in enumerate(tqdm(prompts, desc="Inference")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            logits_dict[i] = outputs.logits.detach().cpu()

        # Snapshot expert activation frequencies for this prompt
        if track_freq and executor is not None:
            batch_freqs = executor.get_freq_accum()
            if batch_freqs:
                prompt_freqs[i] = dict(batch_freqs)
            executor.reset_freq_accum()

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    return logits_dict, prompt_freqs


def compute_kl_stats(ref_logits, test_logits, num_prompts):
    """Compute per-prompt and overall KL divergence."""
    all_kl_values = []
    per_prompt_stats = []

    for i in range(num_prompts):
        if i not in ref_logits or i not in test_logits:
            continue
        ref = ref_logits[i].float()
        tst = test_logits[i].float()
        min_tokens = min(ref.shape[1], tst.shape[1])
        min_vocab = min(ref.shape[2], tst.shape[2])
        ref = ref[:, :min_tokens, :min_vocab]
        tst = tst[:, :min_tokens, :min_vocab]

        ref_prob = F.softmax(ref, dim=-1)
        tst_prob = F.softmax(tst, dim=-1)

        kl = F.kl_div(ref_prob.log(), tst_prob, reduction='none', log_target=False).sum(dim=-1)
        kl_flat = kl.flatten().numpy()
        all_kl_values.extend(kl_flat.tolist())

        # Cosine similarity
        cs = F.cosine_similarity(ref, tst, dim=-1).flatten().numpy()

        per_prompt_stats.append({
            "prompt_idx": i,
            "num_tokens": len(kl_flat),
            "kl_mean": float(np.mean(kl_flat)),
            "kl_median": float(np.median(kl_flat)),
            "kl_p90": float(np.percentile(kl_flat, 90)),
            "kl_max": float(np.max(kl_flat)),
            "cos_mean": float(np.mean(cs)),
            "cos_median": float(np.median(cs)),
            "cos_min": float(np.min(cs)),
        })

    all_kl = np.array(all_kl_values)
    overall = {
        "mean": float(np.mean(all_kl)),
        "median": float(np.median(all_kl)),
        "p90": float(np.percentile(all_kl, 90)),
        "max": float(np.max(all_kl)),
        "pct_lt_0_001": float(100 * np.mean(all_kl < 0.001)),
        "pct_lt_0_01": float(100 * np.mean(all_kl < 0.01)),
        "pct_lt_0_1": float(100 * np.mean(all_kl < 0.1)),
    }

    return overall, per_prompt_stats


def summarize_router_data(num_layers):
    """Summarize router gate probabilities per layer."""
    layer_summary = {}
    for layer_id in range(num_layers):
        if layer_id not in router_data:
            continue
        all_probs = torch.cat(router_data[layer_id]["probs"], dim=0)  # [n_tokens, 60]
        mean_probs = all_probs.mean(dim=0)
        top10_vals, top10_idx = torch.topk(mean_probs, k=10)

        layer_summary[layer_id] = {
            "n_tokens": int(all_probs.shape[0]),
            "top10_experts": top10_idx.tolist(),
            "top10_mean_probs": [round(float(p), 6) for p in top10_vals.tolist()],
            "unique_experts_ever_top4": len(set(
                torch.cat(router_data[layer_id]["topk_idx"], dim=0).flatten().tolist()
            )),
        }
    return layer_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tag = get_current_datatime()

    prompts = sample_first_prompts(DATASET, num_samples=args.num_prompts,
                                   seed=args.seed, max_candidates=500)
    num_layers = List_num_expert_layers["qwen"]
    num_experts = List_num_experts["qwen"]

    # ─── Phase A: Lossless reference + expert activation tracking ───
    print("=" * 70)
    print("  PHASE A: Lossless (LZ4HC) reference inference")
    print("  Capturing: final logits + expert activation frequencies + router probs")
    print("=" * 70)

    model_a = MoE(CHECKPOINT, make_model_config("LZ4HC"))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    n_blocks = register_router_hooks(model_a)
    print(f"  Hooked {n_blocks} MoE blocks")

    executor = model_a.engine.expert_executor
    executor.enable_freq_accum()

    ref_logits, ref_freqs = run_inference(
        model_a, tokenizer, prompts, args.max_prompt_length,
        executor=executor, track_freq=True
    )
    executor.disable_freq_accum()

    # Snapshot router data from Phase A
    router_summary_a = summarize_router_data(num_layers)
    router_data_a = {str(k): v for k, v in router_data.items()}  # for JSON

    # Save Phase A intermediate results to disk
    phase_a_cache = os.path.join(RESULTS_DIR, f"phase_a_cache-{tag}.npz")
    phase_a_data = {
        "num_prompts": np.array(args.num_prompts),
        "prompts": np.array(prompts),
    }
    for i, logit in ref_logits.items():
        phase_a_data[f"logits_{i}"] = logit.float().numpy()
    np.savez_compressed(phase_a_cache, **phase_a_data)
    print(f"  Phase A cache saved to {phase_a_cache}")

    del model_a
    del ref_logits
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)  # Let C++ destructors complete
    print("  GPU memory freed, launching Phase B in subprocess...")

    # ─── Phase B: BLOCKLUT uniform8 inference (in subprocess) ───
    phase_b_script = os.path.join(os.path.dirname(__file__), "_phase_b_runner.py")
    phase_b_out = os.path.join(RESULTS_DIR, f"phase_b_cache-{tag}.npz")
    import subprocess
    result = subprocess.run(
        [sys.executable, phase_b_script,
         "--phase_a_cache", phase_a_cache,
         "--output", phase_b_out,
         "--max_prompt_length", str(args.max_prompt_length)],
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Phase B subprocess failed!")
        print(result.stderr)
        sys.exit(1)

    # Load Phase B results
    phase_b_data = np.load(phase_b_out, allow_pickle=True)
    blut_logits = {}
    for i in range(args.num_prompts):
        blut_logits[i] = torch.from_numpy(phase_b_data[f"logits_{i}"])
    print(f"  Phase B complete, loaded {len(blut_logits)} logit tensors")

    # ─── Phase C: KL divergence ───
    print("\n" + "=" * 70)
    print("  PHASE C: KL Divergence — P_lossless || P_blocklut")
    print("=" * 70)

    kl_overall, kl_per_prompt = compute_kl_stats(ref_logits, blut_logits, args.num_prompts)

    print(f"\n  Per-prompt KL divergence:")
    print(f"  {'Prompt':<10} {'Tokens':<10} {'Mean KL':<14} {'Median KL':<14} "
          f"{'P90 KL':<14} {'CosMean':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    for s in kl_per_prompt:
        print(f"  {s['prompt_idx']:<10} {s['num_tokens']:<10} "
              f"{s['kl_mean']:<14.6f} {s['kl_median']:<14.6f} "
              f"{s['kl_p90']:<14.6f} {s['cos_mean']:<10.6f}")

    print(f"\n  Overall KL:")
    print(f"    Mean:   {kl_overall['mean']:.6f}")
    print(f"    Median: {kl_overall['median']:.6f}")
    print(f"    P90:    {kl_overall['p90']:.6f}")
    print(f"    Max:    {kl_overall['max']:.6f}")
    print(f"    % KL < 0.001: {kl_overall['pct_lt_0_001']:.1f}%")
    print(f"    % KL < 0.01:  {kl_overall['pct_lt_0_01']:.1f}%")
    print(f"    % KL < 0.1:   {kl_overall['pct_lt_0_1']:.1f}%")

    # ─── Phase D: Expert activation frequency report ───
    print("\n" + "=" * 70)
    print("  PHASE D: Expert Activation Frequency Report")
    print("=" * 70)

    # Aggregate across all prompts
    total_freq = defaultdict(int)
    for p_idx, freqs in ref_freqs.items():
        for key, cnt in freqs.items():
            total_freq[key] += cnt
    total_tokens = sum(total_freq.values())

    print(f"\n  Total routed tokens across {len(prompts)} prompts: {total_tokens:,}")
    print(f"\n  Per-prompt routed tokens:")
    for p_idx in sorted(ref_freqs.keys()):
        p_total = sum(ref_freqs[p_idx].values())
        p_active = len(ref_freqs[p_idx])
        print(f"    Prompt {p_idx}: {p_total:,} tokens, {p_active} unique (layer,expert) pairs")

    # Per-layer summary
    print(f"\n  Per-layer expert activation frequencies (all prompts combined):")
    print(f"  {'Layer':<8} {'Tokens':<12} {'ActiveExp':<12} {'Top-3 experts (freq%)'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*40}")

    layer_freqs = defaultdict(lambda: defaultdict(int))
    for (lyr, eid), cnt in total_freq.items():
        layer_freqs[lyr][eid] += cnt

    for lyr in sorted(layer_freqs.keys()):
        lyr_total = sum(layer_freqs[lyr].values())
        active = len(layer_freqs[lyr])
        sorted_exp = sorted(layer_freqs[lyr].items(), key=lambda x: x[1], reverse=True)
        top3_str = ", ".join(f"e{eid}:{cnt/lyr_total*100:.1f}%" for eid, cnt in sorted_exp[:3])
        print(f"  {lyr:<8} {lyr_total:<12,} {active:<12} {top3_str}")

    # ─── Router gate analysis per prompt ───
    print(f"\n  Router gate probability concentration (from Phase A gate hooks):")
    print(f"  {'Layer':<8} {'Mean max prob':<16} {'Mean top4 mass':<16} {'Unique ever top4':<18}")
    print(f"  {'-'*8} {'-'*16} {'-'*16} {'-'*18}")
    for lyr in sorted(router_summary_a.keys()):
        rs = router_summary_a[lyr]
        top10 = rs["top10_mean_probs"]
        print(f"  {lyr:<8} {top10[0]*100:<16.2f}% {sum(top10[:4])*100:<16.2f}% "
              f"{rs['unique_experts_ever_top4']:<18}")

    # ─── Compare expert frequences between prompts ───
    if len(ref_freqs) > 1:
        print(f"\n  Cross-prompt frequency correlation:")
        prompt_ids = sorted(ref_freqs.keys())
        for i in range(len(prompt_ids)):
            for j in range(i + 1, len(prompt_ids)):
                pi, pj = prompt_ids[i], prompt_ids[j]
                fi, fj = ref_freqs[pi], ref_freqs[pj]
                common_keys = set(fi.keys()) & set(fj.keys())
                if common_keys:
                    vals_i = [fi[k] for k in common_keys]
                    vals_j = [fj[k] for k in common_keys]
                    corr = np.corrcoef(vals_i, vals_j)[0, 1]
                    print(f"    Prompt {pi} vs Prompt {pj}: "
                          f"{len(common_keys)} shared keys, Pearson r = {corr:.4f}")

    # ─── Save results ───
    output_json = os.path.join(RESULTS_DIR, f"nested_lut_kl_verify-{tag}.json")
    report = {
        "description": "Nested LUT KL verification — same prompts: lossless vs BLOCKLUT uniform8",
        "date": tag,
        "num_prompts": args.num_prompts,
        "max_new_tokens": args.max_new_tokens,
        "total_routed_tokens": total_tokens,
        "kl_divergence": {
            "overall": kl_overall,
            "per_prompt": kl_per_prompt,
        },
        "expert_activation": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "per_prompt": {
                str(k): {
                    "total_tokens": sum(v.values()),
                    "num_active_pairs": len(v),
                    "top5_experts": sorted(v.items(), key=lambda x: x[1], reverse=True)[:5],
                }
                for k, v in ref_freqs.items()
            },
            "aggregate_frequencies": {
                f"L{lyr}_E{eid}": cnt
                for (lyr, eid), cnt in sorted(total_freq.items())
            },
        },
        "router_gate_analysis": router_summary_a,
        "kl_vs_freq_correlation": None,  # computed below if applicable
    }

    # KL vs frequency correlation: per prompt, correlate expert freq with...
    # Actually we can't directly correlate per-expert KL with freq since KL is per-token,
    # not per-expert. But we can note the routing pattern.

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_json}")

    # Also save raw data as NPZ for later analysis
    npz_path = os.path.join(RESULTS_DIR, f"nested_lut_kl_verify-{tag}.npz")
    np.savez_compressed(
        npz_path,
        num_prompts=np.array(args.num_prompts),
        prompts=np.array(prompts),
        kl_per_prompt_means=np.array([s["kl_mean"] for s in kl_per_prompt]),
        kl_per_prompt_medians=np.array([s["kl_median"] for s in kl_per_prompt]),
        kl_per_prompt_p90=np.array([s["kl_p90"] for s in kl_per_prompt]),
        kl_per_prompt_cos_mean=np.array([s["cos_mean"] for s in kl_per_prompt]),
        total_routed_tokens=np.array(total_tokens),
    )
    print(f"  Raw data saved to: {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
