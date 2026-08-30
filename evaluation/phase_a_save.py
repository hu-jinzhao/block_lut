#!/usr/bin/env python3
"""Phase A: Lossless inference → save reference logits to disk."""
import argparse, os, sys, time, gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from evaluation.profile_tools import sample_first_prompts, clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD = "/home/hh/zip_Moe/LUT_MoE/offload/qwen"
DATASET = "/home/hh/zip_Moe/LUT_MoE/evaluation/dataset/sharegpt_gpt4.jsonl"
OUTPUT = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/ref_logits.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "offload_path": OFFLOAD,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95, "batch_size": 1,
        "code_type": "LZ4HC", "lut_path": "",
        "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }

    prompts = sample_first_prompts(DATASET, num_samples=args.num_prompts,
                                   seed=args.seed, max_candidates=500)

    print("=" * 60)
    print("  PHASE A: Saving lossless reference logits")
    print("=" * 60)

    model = MoE(CHECKPOINT, config)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    save_data = {}
    for i, prompt in enumerate(tqdm(prompts, desc="Lossless inference")):
        inputs = tokenizer(
            prompt, padding=True, truncation=True,
            max_length=args.max_prompt_length, return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.model(inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits.detach().cpu().float().numpy()
            save_data[f"logits_{i}"] = logits
            save_data[f"shape_{i}"] = np.array(logits.shape)

        del outputs
        clear_model_cache(model, aggressive=True)
        gc.collect()
        torch.cuda.empty_cache()

    # Save tokenizer info needed for Phase C
    save_data["num_prompts"] = np.array(args.num_prompts)
    save_data["prompts"] = np.array(prompts)
    np.savez_compressed(OUTPUT, **save_data)
    print(f"\nReference logits saved to {OUTPUT}")
    print(f"Keys: {list(save_data.keys())}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    main()
