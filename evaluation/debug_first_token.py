"""对比 LUT 模型 first-token 输出和期望值"""
import os, sys
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import torch
import numpy as np
from transformers import AutoTokenizer

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

from entry.llm_modeling import MoE

# Find the LZ4HC offload directory (known to work)
import glob
offload_dirs = glob.glob("/home/hh/zip_Moe/LUT_MoE/offload/qwen*")
print(f"Available offload dirs: {offload_dirs}")

# Test LUT model first (most important)
lut_config = {
    "offload_path": "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/",
    "caching_algorithm": "LFU",
    "prefetcher_topk": 0,
    "device_memory_ratio": 0.34,
    "gpu_pool_ratio": 0.6,
    "batch_size": 1,
    "code_type": "LUT",
    "lut_path": LUT_PATH,
    "num_file_chunks": 5,
    "num_compute_threads": 6,
    "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
    "expert_topk": 4,
    "num_elements_per_expert": 1408 * 2048,
    "num_tensors_per_expert": 3,
    "num_expert_layers": 24,
    "num_experts": 60,
    "first_k_dense_replace": 0,
    "hyperparam_state_margin": 0.1,
}

prompt = "你好，请介绍一下你自己"
inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda:0")

print(f"Input: '{prompt}'")
print(f"Input shape: {inputs.input_ids.shape}")

print("\nLoading LUT model...")
model = MoE(CHECKPOINT, lut_config)

import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Running forward pass (first token)...")
with torch.no_grad():
    out = model(inputs.input_ids, attention_mask=inputs.attention_mask)

logits = out.logits[0, -1].float()
top10_tokens = torch.topk(logits, 10)
print(f"\nTop-10 predictions:")
for i in range(10):
    tok = tokenizer.decode([top10_tokens.indices[i]])
    print(f"  {i+1}. '{tok}' (logit={top10_tokens.values[i]:.2f})")

# Check if first token is a reasonable continuation
top1 = tokenizer.decode([top10_tokens.indices[0]])
print(f"\nTop-1 token: '{top1}'")

# Test with generation (just 5 tokens)
print("\n--- Generation test (5 new tokens) ---")
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=5,
        attention_mask=inputs.attention_mask,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Full output: '{result}'")
print(f"New tokens: '{tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)}'")

print("\nDone.")
