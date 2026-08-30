#!/usr/bin/env python3
"""Single-tier runner: measures decode-phase cold/hot GPU cache effects.

Uses model.generate() with a streamer to capture TTFT and per-token decode times.
First prompt in a same-topic sequence is "cold" (experts loaded from SSD),
subsequent prompts are "hot" (experts served from GPU cache).
"""
import argparse, json, os, sys, time, gc
import numpy as np
import torch
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut"
LUT_PATH = os.path.join(CHECKPOINT, "blocklut_256.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, required=True)
    parser.add_argument("--code_type", type=str, default="NESTEDLUT")
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.prompts_file) as f:
        prompts_data = json.load(f)
    # prompts_data is a list of {"text": "...", "topic": "..."}

    config = {
        "offload_path": OFFLOAD,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95,
        "batch_size": 1,
        "code_type": args.code_type, "lut_path": LUT_PATH,
        "lut_tier": args.tier,
        "hyperparam_state_margin": 0.1,
        "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }

    t0 = time.perf_counter()
    model = MoE(CHECKPOINT, config)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.perf_counter() - t0

    # Warmup: short generate to initialize CUDA/GPU cache state
    warmup_ids = tokenizer.encode("hello", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.model(warmup_ids)

    prompt_results = []

    for pi, entry in enumerate(prompts_data):
        prompt_text = entry["text"]
        model.engine.lut_moe_engine.reset_access_counts()

        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda:0")
        prompt_len = input_ids.shape[1]
        attention_mask = inputs.attention_mask.to("cuda:0") if "attention_mask" in inputs else None

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True,
        )

        # Time the full generation
        t_start = time.perf_counter()
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        chunk_times = []  # (wall_time_since_start, text_chunk)
        is_first = True
        for text_chunk in streamer:
            now = time.perf_counter()
            if is_first:
                ttft = now - t_start
                is_first = False
            else:
                chunk_times.append((now - t_start, text_chunk))

        thread.join()
        total_time = time.perf_counter() - t_start

        # Tokenize each chunk to get per-token timing
        token_times = []
        if chunk_times:
            # Estimate per-chunk token count
            prev_time = ttft
            for chunk_t, chunk_text in chunk_times:
                n_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
                if n_tokens > 0:
                    dt = chunk_t - prev_time
                    per_token_dt = dt / n_tokens
                    token_times.extend([per_token_dt] * n_tokens)
                prev_time = chunk_t

        total_tokens = len(token_times) + 1  # +1 for first token

        prompt_result = {
            "prompt_idx": pi,
            "prompt_len_tokens": prompt_len,
            "total_time": total_time,
            "ttft": ttft,
            "total_tokens_generated": total_tokens,
            "tpot_mean": float(np.mean(token_times)) if token_times else None,
            "tpot_min": float(np.min(token_times)) if token_times else None,
            "tpot_max": float(np.max(token_times)) if token_times else None,
            "first_5_tpot": token_times[:5] if len(token_times) >= 5 else token_times,
            "last_5_tpot": token_times[-5:] if len(token_times) >= 5 else token_times,
            "all_tpot": token_times,
            "is_cold": pi == 0,
        }
        prompt_results.append(prompt_result)

        print(f"  [{pi}] len={prompt_len} tokens, "
              f"TTFT={ttft:.3f}s, "
              f"generated={total_tokens} tokens, "
              f"TPOT_mean={prompt_result['tpot_mean']:.3f}s" if prompt_result['tpot_mean'] else f"TPOT_mean=N/A",
              f"({'COLD' if pi == 0 else 'HOT'})")

    # Aggregate: cold = first prompt, hot = rest
    cold = prompt_results[0]
    hot_results = prompt_results[1:]

    cold_decode_times = cold["all_tpot"]
    hot_decode_times = []
    for hr in hot_results:
        hot_decode_times.extend(hr["all_tpot"])

    result = {
        "tier": args.tier,
        "code_type": args.code_type,
        "load_time": load_time,
        "max_new_tokens": args.max_new_tokens,
        "num_prompts": len(prompts_data),
        "cold": {
            "ttft": cold["ttft"],
            "tpot_mean": cold["tpot_mean"],
            "tpot_min": cold["tpot_min"],
            "tpot_max": cold["tpot_max"],
            "first_5_tpot": cold["first_5_tpot"],
            "last_5_tpot": cold["last_5_tpot"],
        },
        "hot": {
            "ttft_mean": float(np.mean([r["ttft"] for r in hot_results])) if hot_results else None,
            "tpot_mean": float(np.mean(hot_decode_times)) if hot_decode_times else None,
            "tpot_min": float(np.min(hot_decode_times)) if hot_decode_times else None,
            "tpot_max": float(np.max(hot_decode_times)) if hot_decode_times else None,
        },
        "speedup": {
            "ttft": cold["ttft"] / (float(np.mean([r["ttft"] for r in hot_results]))) if hot_results else None,
            "tpot": cold["tpot_mean"] / float(np.mean(hot_decode_times)) if hot_decode_times and cold["tpot_mean"] else None,
        },
        "per_prompt": prompt_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Tier {args.tier} decode results:")
    print(f"  Cold TTFT: {cold['ttft']:.3f}s  TPOT: {cold['tpot_mean']:.3f}s")
    if hot_results:
        hot_ttft = float(np.mean([r['ttft'] for r in hot_results]))
        hot_tpot = float(np.mean(hot_decode_times))
        print(f"  Hot  TTFT: {hot_ttft:.3f}s  TPOT: {hot_tpot:.3f}s")
        print(f"  Speedup: TTFT={cold['ttft']/hot_ttft:.2f}x  TPOT={cold['tpot_mean']/hot_tpot:.2f}x")


if __name__ == "__main__":
    main()
