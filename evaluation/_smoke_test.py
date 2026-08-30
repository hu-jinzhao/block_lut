#!/usr/bin/env python3
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

sys.stderr.write('Loading model...\n')
t0 = time.time()
model = MoE('/home/hh/zip_Moe/LUT_MoE/models/qwen', config)
tokenizer = AutoTokenizer.from_pretrained('/home/hh/zip_Moe/LUT_MoE/models/qwen', trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
sys.stderr.write(f'Load: {time.time()-t0:.1f}s\n')

sys.stderr.write('Warmup...\n')
warmup = tokenizer.encode('hello', return_tensors='pt').to('cuda:0')
with torch.no_grad():
    _ = model.model(warmup)
sys.stderr.write('Warmup done\n')

prompt = 'Write a Python function to sort a list.'
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sys.stderr.write(f'Generating: {prompt}\n')
inputs = tokenizer(formatted, return_tensors='pt').to('cuda:0')
with torch.no_grad():
    output = model.model.generate(inputs.input_ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(output[0], skip_special_tokens=True)
sys.stderr.write(f'Generated: {text}\n')
print('OK')
