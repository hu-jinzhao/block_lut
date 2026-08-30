"""快速测试 LUT 模式 offload + 推理"""
import os, sys
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")

# 使用新的 offload 目录避免覆盖现有数据
os.environ["LUT_MOE_TEST"] = "1"

from entry.llm_modeling import MoE
from transformers import AutoTokenizer
import torch
import gc

MODEL = "qwen"
CHECKPOINT = f"/home/hh/zip_Moe/LUT_MoE/models/{MODEL}/"
OFFLOAD = f"/home/hh/zip_Moe/LUT_MoE/offload/{MODEL}_lut/"

# 先删除旧 LUT offload（如果存在）
import shutil
if os.path.exists(OFFLOAD):
    shutil.rmtree(OFFLOAD)
os.makedirs(OFFLOAD, exist_ok=True)

LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
if not os.path.exists(LUT_PATH):
    print("LUT file not found! Run lut_prototype.py first.")
    sys.exit(1)

config = {
    "offload_path": OFFLOAD,
    "caching_algorithm": "LFU",  # 用 LFU 跳过缓存规划
    "prefetcher_topk": 0,
    "device_memory_ratio": 0.34,
    "gpu_pool_ratio": 0.6,
    "batch_size": 1,
    "code_type": "LUT",
    "lut_path": LUT_PATH,
    "hyperparam_state_margin": 0.1,
    "num_file_chunks": 5,
    "num_compute_threads": 6,
    "trace_path": f"/home/hh/zip_Moe/LUT_MoE/trace/{MODEL}_trace.pt",
    "expert_topk": 4,
    "num_elements_per_expert": 1408 * 2048,
    "num_tensors_per_expert": 3,
    "num_expert_layers": 24,
    "num_experts": 60,
    "first_k_dense_replace": 0,
}

print("=" * 60)
print("Step 1: LUT Offload")
print("=" * 60)

model = MoE(CHECKPOINT, config)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 检查 offload 文件大小
param_size = os.path.getsize(os.path.join(OFFLOAD, "lut_moe_param"))
index_size = os.path.getsize(os.path.join(OFFLOAD, "lut_moe_index"))
print(f"\nOffload files:")
print(f"  lut_moe_param: {param_size/1e9:.2f} GB")
print(f"  lut_moe_index: {index_size/1e6:.1f} MB")

# 对比原始 LZ4HC
orig_param = "/home/hh/zip_Moe/LUT_MoE/offload/qwen/lut_moe_param"
if os.path.exists(orig_param):
    orig_size = os.path.getsize(orig_param)
    print(f"  Original LZ4HC: {orig_size/1e9:.2f} GB")
    print(f"  LUT vs LZ4HC: {param_size/orig_size*100:.1f}%")

print("\n" + "=" * 60)
print("Step 2: Warmup Inference")
print("=" * 60)

warmup = "Hello, please introduce yourself."
inputs = tokenizer(warmup, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda:0")

import time
t0 = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        attention_mask=inputs.attention_mask,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
t1 = time.perf_counter()
generated = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"  Output: {generated}")
print(f"  TTFT: {t1-t0:.2f}s")

print("\n" + "=" * 60)
print("Step 3: Multi-token Inference")
print("=" * 60)

prompt = "What is machine learning? Explain in detail."
inputs = tokenizer(prompt, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda:0")

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=64,
        attention_mask=inputs.attention_mask,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
t1 = time.perf_counter()
num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
generated = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"  Output ({num_tokens} tokens): {generated[:200]}...")
print(f"  E2E: {t1-t0:.2f}s, Throughput: {num_tokens/(t1-t0):.1f} tok/s")

print("\n" + "=" * 60)
print("LUT Integration Test PASSED")
print("=" * 60)
