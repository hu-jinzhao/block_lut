#!/usr/bin/env python3
"""End-to-end test for dynamic LUT tier switching.

Strategy: Run multiple prompts in sequence. Each prompt's prefill + decode
phases accumulate visit_count on cached experts. After enough prompts (>10),
some experts should reach visit_count >= 10 → tier 1 (warm). With 50+ hits,
they reach tier 0 (hot).

Tier change messages are printed by C++ via stdout/stderr:
  [LUT_MoE] UpdateOnHit: node X tier 2 -> 1 (visit_count=10)
  [LUT_MoE] tier changed 2 -> 1, forcing re-decompress
  [LUT_MoE] Evict: node X tier 1 -> 2 (cold)
"""
import sys, time, torch
sys.path.insert(0, '/home/hh/zip_Moe/LUT_MoE')
from entry.llm_modeling import MoE
from utils.constants import *
from transformers import AutoTokenizer

config = {
    'offload_path': '/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut',
    'caching_algorithm': 'LFU', 'prefetcher_topk': 4,
    'device_memory_ratio': 0.85, 'gpu_pool_ratio': 0.95,
    'batch_size': 1,
    'code_type': 'NESTEDLUT',
    'lut_path': '/home/hh/zip_Moe/LUT_MoE/models/qwen/blocklut_256.npy',
    'hyperparam_state_margin': 0.1,
    'num_file_chunks': 3, 'num_compute_threads': 6,
    'trace_path': '/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt',
    'expert_topk': List_expert_topk['qwen'],
    'num_elements_per_expert': List_num_elements_per_expert['qwen'],
    'num_tensors_per_expert': List_num_tensors_per_expert['qwen'],
    'num_expert_layers': List_num_expert_layers['qwen'],
    'num_experts': List_num_experts['qwen'],
    'first_k_dense_replace': List_first_k_dense_replace['qwen'],
}

prompts = [
    "Write a Python function to reverse a linked list.",
    "How do I use defaultdict in Python?",
    "Explain Python list comprehensions with examples.",
]

sys.stderr.write(f'=== Dynamic Tier E2E Test ===\n')
sys.stderr.write(f'Prompts: {len(prompts)}\n')
sys.stderr.write(f'Loading model...\n')
t0 = time.time()
model = MoE('/home/hh/zip_Moe/LUT_MoE/models/qwen', config)
tokenizer = AutoTokenizer.from_pretrained('/home/hh/zip_Moe/LUT_MoE/models/qwen', trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
sys.stderr.write(f'Load: {time.time()-t0:.1f}s\n')

# Warmup
sys.stderr.write('Warmup...\n')
warmup = tokenizer.encode('hello', return_tensors='pt').to('cuda:0')
with torch.no_grad():
    _ = model.model(warmup)
sys.stderr.write('Warmup done\n\n')

tier_change_count = 0
evict_count = 0

for i, prompt in enumerate(prompts):
    sys.stderr.write(f'[{i+1}/{len(prompts)}] {prompt[:60]}...\n')
    sys.stderr.flush()

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt').to('cuda:0')

    with torch.no_grad():
        output = model.model.generate(
            inputs.input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract response part
    if '\n' in text:
        response = text.split('\n')[-1][:80]
    else:
        response = text[-80:]
    sys.stderr.write(f'  -> {response}\n')
    sys.stderr.flush()

sys.stderr.write(f'\n=== Test complete ===\n')
sys.stderr.write(f'Check output above for tier promotion/demotion messages.\n')
print('OK')
