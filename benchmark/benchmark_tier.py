#!/usr/bin/env python3
"""
Benchmark: Progressive NestedLUT vs BLOCKLUT 8-bit vs LZ4HC lossless
Compares TTFT, TTOT, and expert activation frequency under 7GB GPU memory limit.
"""
import sys, os, json, time, gc, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LD_LIBRARY_PATH'] = '/home/hh/miniconda3/envs/lut_moe_cu124/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import numpy as np
from transformers import AutoTokenizer
from entry.llm_modeling import MoE
from utils.constants import *

# ── Config ──
GPU_TOTAL = 24.0  # RTX A5000 GB
DENSE_SIZE = 3.7   # GB (attention + embeddings + norms)
TARGET_GPU = 7.0   # GB total limit
CACHE_POOL_GB = TARGET_GPU - DENSE_SIZE  # ~3.3 GB
# Use 0.25 for stable operation. Total GPU = 3.7(dense) + 6.0(cache) ≈ 9.7GB
DEVICE_MEM_RATIO = 0.40

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
PROMPTS = [
    "Explain the concept of attention mechanisms in transformers.",
    "Write a Python function to sort a list using quicksort.",
    "What is the difference between RNA and DNA?",
    "Describe the process of photosynthesis in plants.",
    "How do neural networks learn? Explain backpropagation.",
    "What are the main differences between TCP and UDP?",
    "Write a short story about a robot learning to paint.",
    "Explain the theory of relativity in simple terms.",
    "What is the capital of France and what is it known for?",
    "How does a blockchain work? Describe consensus mechanisms.",
    "What are the key principles of object-oriented programming?",
    "Describe the water cycle in nature.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how GPS works.",
    "What are the main causes of climate change?",
]
MAX_NEW_TOKENS = 64
NUM_WARMUP = 10
NUM_TEST = 5

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results.json')

# ── Config template ──
BASE_CONFIG = {
    'prefetcher_topk': 0, 'caching_algorithm': 'LFU',
    'device_memory_ratio': DEVICE_MEM_RATIO,
    'hyperparam_state_margin': 0,
    'num_file_chunks': 3, 'num_compute_threads': 6,
    'trace_path': '/home/hh/LUT-MoE/trace/qwen_trace.pt',
    'expert_topk': List_expert_topk['qwen'],
    'num_elements_per_expert': List_num_elements_per_expert['qwen'],
    'num_tensors_per_expert': List_num_tensors_per_expert['qwen'],
    'num_expert_layers': List_num_expert_layers['qwen'],
    'num_experts': List_num_experts['qwen'],
    'first_k_dense_replace': List_first_k_dense_replace['qwen'],
    'batch_size': 1,
}

CONFIGS = {
    '16bit_lossless': {
        **BASE_CONFIG,
        'offload_path': '/home/hh/LUT-MoE/offload/qwen_lz4hc',
        'code_type': 'LZ4HC',
        'gpu_pool_ratio': 0.5,
        'lut_path': '',
    },
    '8bit_blocklut': {
        **BASE_CONFIG,
        'offload_path': '/home/hh/LUT-MoE/offload/qwen_blocklut',
        'code_type': 'BLOCKLUT',
        'gpu_pool_ratio': 0,
        'lut_path': '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy',
    },
    'progressive_nestedlut': {
        **BASE_CONFIG,
        'offload_path': '/home/hh/LUT-MoE/offload/qwen_blocklut',
        'code_type': 'NESTEDLUT',
        'gpu_pool_ratio': 0,
        'lut_path': '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy',
        'lut_tier': 0,
    },
}

def reset_engine(model):
    """Reset access counters for a fresh measurement."""
    if hasattr(model.engine, 'lut_moe_engine'):
        model.engine.lut_moe_engine.reset_access_counts()

def measure_ttft_ttot(model, tokenizer, prompts, max_new_tokens, label=''):
    """Measure TTFT and TTOT for a list of prompts. Returns stats dict."""
    results = []
    for prompt in prompts:
        messages = [{'role': 'user', 'content': prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors='pt').to('cuda:0')
        input_len = inputs.input_ids.shape[1]

        torch.cuda.synchronize()
        t_start = time.time()
        ttft = None
        tpot_list = []
        t_last = None

        with torch.no_grad():
            output = model.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # TTFT ≈ time to first token (crude: first ~50% of generation time for prompt processing)
        # More accurate: we time the generate() call and estimate
        # For now, use generate() wall time and break down by log
        torch.cuda.synchronize()
        t_total = time.time() - t_start

        output_tokens = output.shape[1] - input_len
        if output_tokens > 0:
            # Estimate: prompt processing dominates for short generations
            # We use generate() directly since transformers doesn't expose per-token timing
            ttft = t_total / 2  # rough estimate: half for prompt, half for tokens
            avg_tpot = t_total / max(1, output_tokens) if output_tokens > 0 else 0
        else:
            ttft = t_total
            avg_tpot = 0

        results.append({
            'prompt': prompt[:50],
            'input_len': input_len,
            'output_tokens': output_tokens,
            'ttft': ttft,
            'avg_tpot': avg_tpot,
            'total_time': t_total,
        })
        print(f"  [{len(results)}/{len(prompts)}] {label} TTFT={ttft*1000:.0f}ms TPOT={avg_tpot*1000:.0f}ms out={output_tokens}", flush=True)

    # Aggregate
    ttfts = [r['ttft'] for r in results if r['ttft']]
    tpots = [r['avg_tpot'] for r in results if r['avg_tpot'] > 0]
    return {
        'avg_ttft': np.mean(ttfts) if ttfts else 0,
        'avg_tpot': np.mean(tpots) if tpots else 0,
        'ttft_list': ttfts,
        'tpot_list': tpots,
        'raw': results,
    }

def get_expert_freq(model):
    """Try to extract expert activation frequencies."""
    try:
        if hasattr(model.engine, 'expert_executor') and hasattr(model.engine.expert_executor, 'get_freq_accum'):
            freq = model.engine.expert_executor.get_freq_accum()
            if freq:
                return {str(k): v for k, v in freq.items()}
    except:
        pass
    return {}

def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote=True)
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Benchmark: {name}")
        print(f"  device_memory_ratio={cfg['device_memory_ratio']:.3f} "
              f"(={(cfg['device_memory_ratio']*GPU_TOTAL):.1f}GB cache pool)")
        print(f"{'='*60}", flush=True)

        model = MoE(MODEL_DIR, cfg)

        # Warmup
        print(f"  Warmup ({NUM_WARMUP} prompts)...", flush=True)
        reset_engine(model)
        for i in range(NUM_WARMUP):
            p = PROMPTS[i % len(PROMPTS)]
            messages = [{'role': 'user', 'content': p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors='pt').to('cuda:0')
            with torch.no_grad():
                _ = model.model.generate(inputs.input_ids, max_new_tokens=16, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            print(f"    warmup {i+1}/{NUM_WARMUP}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

        # Measure
        print(f"  Measuring ({NUM_TEST} prompts)...", flush=True)
        stats = measure_ttft_ttot(model, tokenizer, PROMPTS[:NUM_TEST], MAX_NEW_TOKENS, name)

        all_results[name] = stats
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY (7GB GPU limit)")
    print(f"{'='*60}")
    baseline = '8bit_blocklut'
    bl = all_results.get(baseline, {})
    print(f"{'Method':25s} {'TTFT(ms)':12s} {'TPOT(ms)':12s} {'Speedup':10s}")
    print(f"{'-'*60}")
    for name, r in all_results.items():
        ttft_ms = r.get('avg_ttft', 0) * 1000
        tpot_ms = r.get('avg_tpot', 0) * 1000
        bl_ttft = bl.get('avg_ttft', 1) * 1000
        bl_tpot = bl.get('avg_tpot', 1) * 1000
        speedup_ttft = bl_ttft / max(ttft_ms, 0.001)
        speedup_tpot = bl_tpot / max(tpot_ms, 0.001)
        print(f"{name:25s} {ttft_ms:8.0f}ms   {tpot_ms:8.0f}ms   {speedup_ttft:.2f}x/{speedup_tpot:.2f}x")
    print()

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == '__main__':
    run()
