"""快速诊断 LUT 推理路径 - 使用已有 offload"""
import os, sys, time
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch
import LUT_MoE
from utils.config import LUT_MoEConfig
from utils.constants import DELAY_PROFILE, COMPRESSION_RATIO_PROFILE

OFFLOAD_DIR = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"

# 加载 LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16_arr = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

config = {
    "offload_path": OFFLOAD_DIR,
    "caching_algorithm": "LFU",
    "prefetcher_topk": 0,
    "device_memory_ratio": 0.34,
    "gpu_pool_ratio": 0.6,
    "batch_size": 1,
    "code_type": "LUT",
    "lut_path": LUT_PATH,
    "hyperparam_state_margin": 0.1,
    "num_file_chunks": 5,
    "num_compute_threads": 6,
    "trace_path": f"/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
    "expert_topk": 4,
    "num_elements_per_expert": 1408 * 2048,
    "num_tensors_per_expert": 3,
    "num_expert_layers": 24,
    "num_experts": 60,
    "first_k_dense_replace": 0,
}

# 测试: 手动做一个 pread + decompress + LUT recover
# 检查 index 里第一个 sparse tensor 的元数据
print(f"\nFile size: {os.path.getsize(os.path.join(OFFLOAD_DIR, 'lut_moe_param')) / 1e9:.2f} GB")
print(f"Index size: {os.path.getsize(os.path.join(OFFLOAD_DIR, 'lut_moe_index')) / 1e6:.1f} MB")

# 测试能否 run model
from entry.llm_modeling import MoE
from transformers import AutoTokenizer

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"

print("\nLoading model...")
model = MoE(CHECKPOINT, config)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Model loaded, testing warmup inference...")
warmup = "Hello."
inputs = tokenizer(warmup, padding=True, truncation=True, max_length=512,
                   return_tensors="pt").to("cuda:0")

import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Starting generate...")
t0 = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=5,
        attention_mask=inputs.attention_mask,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
td = time.perf_counter() - t0
print(f"Generate completed in {td:.1f}s")
print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print("DONE")
