#!/usr/bin/env python3
"""Phase B subprocess runner: loads BLOCKLUT model, runs inference on same prompts as Phase A.
Standalone process to avoid GPU OOM from C++ runtime memory not being freed.
"""

import argparse
import os
import sys
import gc
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_a_cache", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    args = parser.parse_args()

    # Load Phase A cache
    print("  Loading Phase A cache...")
    cache = np.load(args.phase_a_cache, allow_pickle=True)
    num_prompts = int(cache["num_prompts"])
    prompts = list(cache["prompts"])

    # Load BLOCKLUT model
    print("  Loading BLOCKLUT model...")
    lut_path = os.path.join(CHECKPOINT, "blocklut_256.npy")
    config = {
        "offload_path": OFFLOAD_BLOCKLUT,
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
            blut_logits[i] = outputs.logits.detach().cpu().float().numpy()

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    save_data = {"num_prompts": np.array(num_prompts)}
    for i, logit in blut_logits.items():
        save_data[f"logits_{i}"] = logit
    np.savez_compressed(args.output, **save_data)
    print(f"\nPhase B cache saved to {args.output}")
    print("Phase B complete.")


if __name__ == "__main__":
    main()
