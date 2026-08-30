#!/usr/bin/env python3
"""Run a single benchmark config. Called by benchmark_tier.py for isolation."""
import sys, os, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['LD_LIBRARY_PATH'] = '/home/hh/miniconda3/envs/lut_moe_cu124/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch, numpy as np
from transformers import AutoTokenizer
from entry.llm_modeling import MoE
from utils.constants import *

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
GPU_TOTAL = 24.0

PROMPTS = [
    "Explain the concept of attention mechanisms.",
    "Write a Python function to sort a list using quicksort.",
    "What is the difference between RNA and DNA?",
    "Describe the process of photosynthesis.",
    "How do neural networks learn? Explain backpropagation.",
    "What are the main differences between TCP and UDP?",
    "Write a short story about a robot learning to paint.",
    "Explain the theory of relativity in simple terms.",
    "What is the capital of France?",
    "How does a blockchain work?",
    "What are the key principles of OOP?",
    "Describe the water cycle.",
    "What is supervised vs unsupervised learning?",
    "Explain how GPS works.",
    "What are the main causes of climate change?",
]

if __name__ == '__main__':
    name = sys.argv[1]
    device_mem_ratio = float(sys.argv[2])

    configs = {
        '16bit_raw_bf16': {'offload_path': '/home/hh/LUT-MoE/offload/qwen_raw', 'code_type': 'RAW', 'gpu_pool_ratio': 0.9, 'lut_path': ''},
        '8bit_blocklut':  {'offload_path': '/home/hh/LUT-MoE/offload/qwen_blocklut', 'code_type': 'BLOCKLUT', 'gpu_pool_ratio': 0, 'lut_path': '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy'},
        'progressive_nestedlut': {'offload_path': '/home/hh/LUT-MoE/offload/qwen_blocklut', 'code_type': 'NESTEDLUT', 'gpu_pool_ratio': 0, 'lut_path': '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy', 'lut_tier': 0},
    }

    cfg = configs[name]
    full_cfg = {
        **cfg,
        'prefetcher_topk': 0, 'caching_algorithm': 'LFU',
        'device_memory_ratio': device_mem_ratio,
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

    print(f"  Loading {name}...", flush=True)
    t0 = time.time()
    model = MoE(MODEL_DIR, full_cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    print(f"  Warmup...", flush=True)
    for i in range(10):
        p = PROMPTS[i % len(PROMPTS)]
        msgs = [{'role': 'user', 'content': p}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            _ = model.model.generate(inp.input_ids, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if i == 0:
            print(f"  First warmup done", flush=True)

    # Measure
    print(f"  Measuring...", flush=True)
    results = []
    for prompt in PROMPTS[:5]:
        msgs = [{'role': 'user', 'content': prompt}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors='pt').to('cuda:0')
        input_len = inp.input_ids.shape[1]
        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            output = model.model.generate(inp.input_ids, max_new_tokens=64, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        t_total = time.time() - t_start
        out_tokens = output.shape[1] - input_len
        ttft = t_total / 2  # estimate
        tpot = t_total / max(out_tokens, 1)
        results.append({'prompt': prompt[:30], 'input_len': input_len, 'out_tokens': out_tokens, 'ttft': ttft, 'tpot': tpot})
        print(f"    [{len(results)}/5] TTFT={ttft*1000:.0f}ms TPOT={tpot*1000:.0f}ms out={out_tokens}", flush=True)

    # Summary
    avg_ttft = np.mean([r['ttft'] for r in results])
    avg_tpot = np.mean([r['tpot'] for r in results])
    print(f"\n  RESULT {name}: TTFT={avg_ttft*1000:.0f}ms TPOT={avg_tpot*1000:.0f}ms", flush=True)
    print(json.dumps({'name': name, 'avg_ttft_ms': avg_ttft*1000, 'avg_tpot_ms': avg_tpot*1000, 'raw': results}))
