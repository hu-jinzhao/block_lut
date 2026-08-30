#!/usr/bin/env python3
"""LUT 4bit (K=16) PPL - re-encodes bf16 weights with 4-bit codebook on-the-fly."""
import os, sys, json, math, time, glob, shutil, numpy as np, gc
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict

ORIG_DIR = "/home/hh/LUT-MoE/models/qwen"
BF16_DIR = "/tmp/qwen_lut4_bf16_v2"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
BS = 128

# 4-bit codebook: K=16
c16 = torch.from_numpy(np.load(os.path.join(ORIG_DIR, 'blocklut_16.npy'))).float()
# Build 256-entry table: each of 16 centroids repeated 16 times
codebook_256 = c16.repeat_interleave(16)  # [256]

def lut4_quantize(weight, codebook=codebook_256, block_size=BS):
    """Quantize weight to LUT 4bit and dequantize back."""
    w = weight.float().reshape(-1)
    N = w.shape[0]
    nb = (N + block_size - 1) // block_size
    pad = torch.zeros(nb * block_size)
    pad[:N] = w
    blocks = pad.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = (blocks / amax).reshape(-1)[:N]
    # Find nearest centroid in 256-entry table
    dists = (normalized.unsqueeze(1) - codebook.unsqueeze(0)).abs()
    idx = dists.argmin(dim=1)  # [N] uint8
    # Dequantize using 256-entry codebook
    dq = codebook[idx] * amax.reshape(-1)[torch.arange(N) // block_size].clamp(max=nb-1)
    return dq.reshape(weight.shape).to(torch.bfloat16)

# Step 1: Convert original bf16 → LUT 4bit → bf16
print("Converting bf16 → LUT 4bit → bf16...", flush=True)
os.makedirs(BF16_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(ORIG_DIR, 'model-*.safetensors')))

total_exp = 0
for f in files:
    fname = os.path.basename(f)
    print(f"  {fname}...", flush=True)
    state = load_file(f, device='cpu')
    new_state = OrderedDict()
    for k, v in state.items():
        is_exp = 'experts.' in k and 'shared_expert' not in k
        if is_exp and k.endswith('.weight') and v.dtype == torch.bfloat16:
            # Quantize with LUT 4bit, then dequantize back to bf16
            new_state[k] = lut4_quantize(v)
            total_exp += 1
        else:
            new_state[k] = v
    out_path = os.path.join(BF16_DIR, fname)
    save_file(new_state, out_path)
    del state, new_state
    gc.collect()
    print(f"    {os.path.getsize(out_path)/1e9:.1f}GB", flush=True)

print(f"  {total_exp} experts quantized with 4-bit codebook", flush=True)

# Copy config
for fn in os.listdir(ORIG_DIR):
    if fn.endswith(('.json','.txt','.py')) and not fn.startswith('model-'):
        shutil.copy2(os.path.join(ORIG_DIR, fn), os.path.join(BF16_DIR, fn))

# PPL
print("\nPPL evaluation...", flush=True)
with open(DATASET) as f:
    texts = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(BF16_DIR, trust_remote_code=True)
enc = tokenizer.encode(" ".join(texts))
print(f"WikiText-2: {len(enc)} tokens", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BF16_DIR, torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True,
)
model.eval()

MAX_LEN, STRIDE = 2048, 1024
nlls, total_tok = [], 0
t1 = time.time()

for i in range(0, len(enc), STRIDE):
    end = min(i+MAX_LEN, len(enc))
    if end-i < 128: break
    inp = torch.tensor([enc[i:end]], dtype=torch.long)
    with torch.no_grad():
        out = model(inp, labels=inp)
    n = inp.shape[1]-1
    nlls.append(out.loss.item()*n)
    total_tok += n
    if (i//STRIDE+1)%5==0:
        print(f"  step {i//STRIDE+1}: PPL={math.exp(sum(nlls)/total_tok):.4f}", flush=True)

ppl = math.exp(sum(nlls)/total_tok)
print(f"\n=== LUT 4bit (K=16) PPL = {ppl:.4f} ===", flush=True)
