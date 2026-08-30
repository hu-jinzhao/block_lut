#!/usr/bin/env python3
"""Quick test: load with cold tier (4-bit) and verify inference works."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = '/home/hh/miniconda3/envs/lut_moe_cu124/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

from entry.llm_modeling import MoE
from utils.constants import *
from transformers import AutoTokenizer
import torch

config = {
    'offload_path': '/home/hh/LUT-MoE/offload/qwen_blocklut',
    'prefetcher_topk': 0, 'caching_algorithm': 'LFU',
    'device_memory_ratio': 0.5, 'gpu_pool_ratio': 0,
    'code_type': 'NESTEDLUT', 'lut_tier': 2,  # cold: 4-bit (read 50% from SSD)
    'hyperparam_state_margin': 0,
    'num_file_chunks': 3, 'num_compute_threads': 6,
    'trace_path': '/home/hh/LUT-MoE/trace/qwen_trace.pt',
    'expert_topk': List_expert_topk['qwen'],
    'num_elements_per_expert': List_num_elements_per_expert['qwen'],
    'num_tensors_per_expert': List_num_tensors_per_expert['qwen'],
    'num_expert_layers': List_num_expert_layers['qwen'],
    'num_experts': List_num_experts['qwen'],
    'first_k_dense_replace': List_first_k_dense_replace['qwen'],
    'lut_path': '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy',
    'batch_size': 1,
}
print('Loading with cold tier (4-bit, ~50% I/O expected)...', flush=True)
t0 = time.time()
model = MoE('/home/hh/LUT-MoE/models/qwen', config)
print(f'Load: {time.time()-t0:.1f}s', flush=True)

tokenizer = AutoTokenizer.from_pretrained('/home/hh/LUT-MoE/models/qwen', trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
messages = [{'role': 'user', 'content': 'Hello!'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
print('Generating...', flush=True)
with torch.no_grad():
    output = model.model.generate(inputs.input_ids, max_new_tokens=16, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
print('Output:', tokenizer.decode(output[0], skip_special_tokens=True), flush=True)
print('SUCCESS!', flush=True)
