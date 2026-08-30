#!/usr/bin/env python3
"""
Dump Qwen MoE router logits to understand the expert activation pattern.

Key questions:
1. Are the 4 "always-active" experts per layer shared experts or routed experts?
2. What does the probability distribution over all 60 experts look like?
3. Do different prompts activate different experts?
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entry.llm_modeling import MoE
from evaluation.profile_tools import sample_first_prompts, clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

# ── Global storage for captured router data ──
router_data = {}  # layer_id -> { "logits": [...], "probs": [...], "selected": [...] }


def capture_router_hook(module, input, output, layer_id):
    """Forward hook on Qwen2MoEBlock.gate to capture raw router logits."""
    # input[0] = hidden_states [batch*seq, hidden_dim]
    # output = raw logits [batch*seq, 60]
    logits = output.detach().cpu()
    probs = F.softmax(logits.float(), dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k=4, dim=-1)

    if layer_id not in router_data:
        router_data[layer_id] = {"logits": [], "probs": [], "selected": []}
    router_data[layer_id]["logits"].append(logits)
    router_data[layer_id]["probs"].append(probs)
    router_data[layer_id]["selected"].append(topk_idx)


def main():
    target = "qwen"
    checkpoint = f"/home/hh/zip_Moe/LUT_MoE/models/{target}/"
    offload_path = f"/home/hh/zip_Moe/LUT_MoE/offload/{target}/"

    config = {
        "offload_path": offload_path,
        "caching_algorithm": "LFU",
        "prefetcher_topk": 4,
        "device_memory_ratio": 0.85,
        "gpu_pool_ratio": 0.95,
        "batch_size": 1,
        "code_type": "LZ4HC",
        "lut_path": "",
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

    print("[1/3] Loading model...")
    model = MoE(checkpoint, config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # ── Register hooks on all MoE blocks' gate layers ──
    print("[2/3] Registering router hooks...")
    moe_blocks = []
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "Qwen2MoEBlock":
            moe_blocks.append((name, module))
            layer_id = module.layer_id
            module.gate.register_forward_hook(
                lambda m, inp, out, lid=layer_id: capture_router_hook(m, inp, out, lid)
            )
    print(f"  Found {len(moe_blocks)} MoE blocks, hooked {len(router_data)} gates")

    # ── Run 3 prompts with different content ──
    print("[3/3] Running calibration prompts...")
    dataset_path = "/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"
    prompts = sample_first_prompts(dataset_path, num_samples=3, seed=42, max_candidates=500)

    executor = model.engine.expert_executor
    executor.enable_freq_accum()

    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}...")
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            model.generate(
                inputs.input_ids,
                max_new_tokens=32,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        executor.reset_freq_accum()
        clear_model_cache(model, aggressive=True)

    executor.disable_freq_accum()

    # ── Analysis ──
    print("\n" + "=" * 70)
    print("  ROUTER LOGITS ANALYSIS")
    print("=" * 70)

    # For each layer, concatenate all token logits and analyze
    num_layers = List_num_expert_layers[target]

    overall_expert_hits = {}  # expert_id -> count of how many layers it appears in top-4
    layer_top4 = {}  # layer_id -> set of top-4 expert IDs (across ALL tokens)
    layer_expert_heatmap = {}  # layer_id -> [60] mean probability per expert

    for layer_id in range(num_layers):
        if layer_id not in router_data:
            print(f"\n  Layer {layer_id}: NO DATA")
            continue

        all_logits = torch.cat(router_data[layer_id]["logits"], dim=0).float()   # [total_tokens, 60]
        all_probs = torch.cat(router_data[layer_id]["probs"], dim=0)              # [total_tokens, 60]
        all_selected = torch.cat(router_data[layer_id]["selected"], dim=0)        # [total_tokens, 4]

        n_tokens = all_logits.shape[0]
        print(f"\n  Layer {layer_id}: {n_tokens} tokens processed")

        # Mean probability per expert (across all tokens)
        mean_probs = all_probs.mean(dim=0)  # [60]
        top_probs, top_experts = torch.topk(mean_probs, k=8)
        print(f"    Top-8 experts by mean probability:")
        for rank, (eid, prob) in enumerate(zip(top_experts.tolist(), top_probs.tolist())):
            pct = prob * 100
            print(f"      #{rank+1}: expert {eid:2d}  mean_prob={pct:.2f}%")

        # Unique experts ever selected
        unique_selected = set(all_selected.flatten().tolist())
        print(f"    Unique experts selected: {len(unique_selected)}/60 = {sorted(unique_selected)}")

        # How many experts have mean prob > 1/60 (~1.67%)?
        threshold = 1.0 / 60
        above_uniform = (mean_probs > threshold).sum().item()
        print(f"    Experts above uniform threshold (1.67%): {above_uniform}/60")

        # Concentration: what fraction of probability mass goes to top-4?
        top4_mass = top_probs[:4].sum().item() * 100
        print(f"    Top-4 probability mass: {top4_mass:.1f}%")

        # Per-token: are the same 4 experts always selected?
        selected_sets = set()
        for t in range(min(n_tokens, 1000)):  # check first 1000 tokens
            selected_sets.add(tuple(sorted(all_selected[t].tolist())))
        print(f"    Unique top-4 combinations (first 1000 tokens): {len(selected_sets)}")
        if len(selected_sets) <= 5:
            for s in sorted(selected_sets):
                print(f"      {s}")

        layer_expert_heatmap[layer_id] = mean_probs.numpy()

    # ── Cross-layer summary ──
    print("\n" + "=" * 70)
    print("  CROSS-LAYER SUMMARY")
    print("=" * 70)

    # For each layer, find the set of experts that are ALWAYS in top-4 vs sometimes
    print(f"\n  Per-layer routing diversity:")
    for layer_id in range(num_layers):
        if layer_id not in router_data:
            continue
        all_selected = torch.cat(router_data[layer_id]["selected"], dim=0)
        n_tokens = all_selected.shape[0]

        # Count per-expert selection frequency
        expert_freq = torch.zeros(60)
        for t in range(all_selected.shape[0]):
            for k in range(4):
                expert_freq[all_selected[t, k]] += 1

        always_selected = (expert_freq == n_tokens * 4).nonzero().flatten().tolist()
        # Actually: an expert selected for EVERY token would have freq = n_tokens
        freq_per_token = expert_freq / n_tokens
        always = (freq_per_token == 1.0).nonzero().flatten().tolist()
        often = ((freq_per_token >= 0.1) & (freq_per_token < 1.0)).nonzero().flatten().tolist()
        rarely = ((freq_per_token > 0) & (freq_per_token < 0.1)).nonzero().flatten().tolist()
        never = (freq_per_token == 0).nonzero().flatten().tolist()

        print(f"  Layer {layer_id}: always={len(always)} often={len(often)} "
              f"rarely={len(rarely)} never={len(never)}")
        if always:
            print(f"    Always: {always}")
        if often:
            print(f"    Often (>=10%): {often}")
        if rarely:
            print(f"    Rarely (<10%): {len(rarely)} experts")
        if never:
            print(f"    Never: {len(never)} experts")

    # ── Save raw data ──
    output_path = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/router_logits_analysis.npz"
    np.savez_compressed(output_path, **{f"layer_{k}": v for k, v in layer_expert_heatmap.items()})
    print(f"\nRaw data saved to {output_path}")


if __name__ == "__main__":
    main()
