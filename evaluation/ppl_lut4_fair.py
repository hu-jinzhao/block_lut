#!/usr/bin/env python3
"""LUT 4bit (K=16) PPL - fair comparison with GPTQ."""
import os, sys, json, math, time, glob, shutil, numpy as np
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LUT_DIR = "/home/hh/LUT-MoE/models/qwen_lut4bit"
ORIG_DIR = "/home/hh/LUT-MoE/models/qwen"
BF16_DIR = "/tmp/qwen_lut4_bf16"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
BS = 128

# 4-bit codebook: K=16 mapped to 256 entries
c16 = torch.from_numpy(np.load(os.path.join(ORIG_DIR, 'blocklut_16.npy'))).float()
codebook_4bit = c16.repeat_interleave(16)

# Load absmax
print("Loading absmax...", flush=True)
absmax_npz = np.load(os.path.join(LUT_DIR, 'lut_absmax.npz'))
absmax_map = {k: torch.from_numpy(v) for k, v in absmax_npz.items()}
print(f"  {len(absmax_map)} entries", flush=True)

# Convert
print("Decompressing LUT 4bit → bf16...", flush=True)
os.makedirs(BF16_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(LUT_DIR, 'model-*_lut.safetensors')))
total_exp = 0

for f in files:
    fname = os.path.basename(f)
    print(f"  {fname}...", flush=True)
    state = load_file(f, device='cpu')
    new_state = {}
    for k, v in state.items():
        if k.endswith('.absmax'): continue
        is_exp = 'experts.' in k and 'shared_expert' not in k
        if is_exp and k.endswith('.weight') and v.dtype == torch.uint8:
            amax = absmax_map.get(k)
            if amax is not None:
                flat = v.reshape(-1).long()
                norm = codebook_4bit[flat]  # uint8 index directly into 256-entry table
                N = flat.shape[0]
                bid = torch.arange(N) // BS
                bid = bid.clamp(max=amax.shape[0]-1)
                new_state[k] = (norm * amax[bid]).reshape(v.shape).to(torch.bfloat16)
                total_exp += 1
                continue
        new_state[k] = v
    out_name = fname.replace('_lut.safetensors', '.safetensors')
    save_file(new_state, os.path.join(BF16_DIR, out_name))
    print(f"    {os.path.getsize(os.path.join(BF16_DIR, out_name))/1e9:.1f}GB", flush=True)

print(f"  {total_exp} experts decompressed", flush=True)

# Copy config
for fn in os.listdir(ORIG_DIR):
    if fn.endswith(('.json','.txt','.py')) and not fn.startswith('model-'):
        shutil.copy2(os.path.join(ORIG_DIR, fn), os.path.join(BF16_DIR, fn))

# PPL
print("\nPPL evaluation...", flush=True)
with open(DATASET) as f:
    texts = json.load(f)
full_text = " ".join(texts)

tokenizer = AutoTokenizer.from_pretrained(BF16_DIR, trust_remote_code=True)
enc = tokenizer.encode(full_text)
print(f"WikiText-2: {len(enc)} tokens", flush=True)

print("Loading model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BF16_DIR, torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True, low_cpu_mem_usage=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

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
        ppl = math.exp(sum(nlls)/total_tok)
        print(f"  step {i//STRIDE+1}: PPL={ppl:.4f}", flush=True)

ppl = math.exp(sum(nlls)/total_tok)
print(f"\n=== LUT 4bit (K=16) PPL = {ppl:.4f} ===", flush=True)
print(f"Tokens: {total_tok}, Time: {time.time()-t1:.1f}s", flush=True)
