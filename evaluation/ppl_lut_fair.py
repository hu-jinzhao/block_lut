#!/usr/bin/env python3
"""LUT 4bit PPL with same settings as GPTQ evaluation (fair comparison)."""
import os, sys, json, math, time, glob, shutil, numpy as np
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LUT_DIR = "/home/hh/LUT-MoE/models/qwen_lut4bit"
ORIG_DIR = "/home/hh/LUT-MoE/models/qwen"
BF16_DIR = "/tmp/qwen_lut_bf16"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
BS = 128

# Load LUT codebook (256-entry, same as used for encoding)
codebook = torch.from_numpy(np.load(os.path.join(ORIG_DIR, 'blocklut_256.npy'))).float()

def decompress_lut(indices, amax, codebook, shape):
    """Decompress LUT uint8 → bf16."""
    flat = indices.reshape(-1).long()
    norm = codebook[flat]
    N = flat.shape[0]
    bid = torch.arange(N) // BS
    bid = bid.clamp(max=amax.shape[0] - 1)
    return (norm * amax[bid]).reshape(shape).to(torch.bfloat16)

# Load absmax from npz
print("Loading absmax...", flush=True)
absmax_npz = np.load(os.path.join(LUT_DIR, 'lut_absmax.npz'))
absmax_map = {k: torch.from_numpy(v) for k, v in absmax_npz.items()}
print(f"  {len(absmax_map)} absmax entries", flush=True)

# Step 1: Convert LUT format → bf16
print("Converting LUT to bf16...", flush=True)
os.makedirs(BF16_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(LUT_DIR, 'model-*_lut.safetensors')))

for f in files:
    fname = os.path.basename(f)
    print(f"  {fname}...", flush=True)
    state = load_file(f, device='cpu')
    new_state = {}
    for k, v in state.items():
        is_exp = 'experts.' in k and 'shared_expert' not in k
        is_q = k.endswith('.absmax')
        if is_q:
            continue  # skip absmax entries (loaded from npz)
        if is_exp and k.endswith('.weight') and v.dtype == torch.uint8:
            # Find absmax from npz
            amax = absmax_map.get(k)
            if amax is not None:
                new_state[k] = decompress_lut(v, amax, codebook, v.shape)
                continue
        new_state[k] = v

    out_name = fname.replace('_lut.safetensors', '.safetensors')
    out_path = os.path.join(BF16_DIR, out_name)
    save_file(new_state, out_path)
    print(f"    {os.path.getsize(out_path)/1e9:.1f}GB", flush=True)
    del state, new_state

# Copy config files from original model
for f in os.listdir(ORIG_DIR):
    if f.endswith(('.json', '.txt', '.py')) and not f.startswith('model-'):
        shutil.copy2(os.path.join(ORIG_DIR, f), os.path.join(BF16_DIR, f))

print(f"Conversion done: {BF16_DIR}", flush=True)

# Step 2: PPL evaluation (same as GPTQ)
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
    device_map='auto', trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

MAX_LEN = 2048
STRIDE = 1024
nlls, total_tok = [], 0
t1 = time.time()

for i in range(0, len(enc), STRIDE):
    end = min(i + MAX_LEN, len(enc))
    if end - i < 128: break
    inp = torch.tensor([enc[i:end]], dtype=torch.long)
    with torch.no_grad():
        out = model(inp, labels=inp)
    n = inp.shape[1] - 1
    nlls.append(out.loss.item() * n)
    total_tok += n
    if (i // STRIDE + 1) % 5 == 0:
        ppl = math.exp(sum(nlls) / total_tok)
        print(f"  step {i//STRIDE+1}: PPL={ppl:.4f}", flush=True)

ppl = math.exp(sum(nlls) / total_tok)
print(f"\n=== LUT 4bit PPL = {ppl:.4f} ===", flush=True)
print(f"Tokens: {total_tok}, Time: {time.time()-t1:.1f}s", flush=True)
