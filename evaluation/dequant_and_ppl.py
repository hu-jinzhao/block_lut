#!/usr/bin/env python3
"""Dequantize GPTQ to bf16, then run PPL via HuggingFace."""

import os, sys, json, math, time, glob, shutil
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

GPTQ_DIR = "/home/hh/LUT-MoE/models/qwen_gptq"
BF16_DIR = "/tmp/qwen_bf16_tmp"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"

# Dequantize GPTQ → bf16 (same function as before)
def dequant_gptq(qweight, scales, qzeros, gs=128):
    pf = 8  # 32/4
    rows = qweight.shape[1]
    cols = qweight.shape[0] * pf
    dq = torch.zeros((rows, cols), dtype=torch.bfloat16)
    for g in range(scales.shape[0]):
        cs, ce = g*gs, min((g+1)*gs, cols)
        for c in range(cs, ce):
            nib = c % pf
            qc = ((qweight[c//pf] >> (nib*4)) & 0xF).float()
            zc = ((qzeros[g] >> (nib*4)) & 0xF).float() + 1
            dq[:, c] = ((qc - zc.repeat_interleave(8)[:rows]) * scales[g].float()).to(torch.bfloat16)
    return dq

# Step 1: Convert to bf16
print("Step 1: Dequantizing GPTQ weights to bf16...", flush=True)
os.makedirs(BF16_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(GPTQ_DIR, 'model-*.safetensors')))

for f in files:
    fname = os.path.basename(f)
    print(f"  {fname}...", flush=True)
    from safetensors.torch import load_file
    state = load_file(f, device='cpu')
    new_state = {}
    for k, v in state.items():
        if 'qweight' in k:
            base = k.replace('.qweight', '')
            qw = v
            sc = state[base + '.scales']
            qz = state[base + '.qzeros']
            new_state[base + '.weight'] = dequant_gptq(qw, sc, qz)
        elif 'scales' in k or 'qzeros' in k or 'g_idx' in k:
            continue  # skip quantization metadata
        else:
            new_state[k] = v

    out_path = os.path.join(BF16_DIR, fname)
    save_file(new_state, out_path)
    print(f"    -> {os.path.getsize(out_path)/1e9:.1f}GB", flush=True)
    del state, new_state

# Copy config
for f in os.listdir(GPTQ_DIR):
    if f.endswith(('.json', '.txt', '.py')):
        shutil.copy2(os.path.join(GPTQ_DIR, f), os.path.join(BF16_DIR, f))
# Remove quantization config
cfg_path = os.path.join(BF16_DIR, 'config.json')
with open(cfg_path) as f:
    cfg = json.load(f)
cfg.pop('quantization_config', None)
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)

# Remove quantize_config.json
if os.path.exists(os.path.join(BF16_DIR, 'quantize_config.json')):
    os.remove(os.path.join(BF16_DIR, 'quantize_config.json'))

print(f"\nDequantization complete. Output: {BF16_DIR}", flush=True)

# Step 2: PPL evaluation
print("\nStep 2: PPL evaluation (CPU offloading)...", flush=True)
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
nlls, total_tok = [], []
t1 = time.time()

for i in range(0, len(enc), STRIDE):
    end = min(i + MAX_LEN, len(enc))
    if end - i < 128: break
    inp = torch.tensor([enc[i:end]], dtype=torch.long)
    with torch.no_grad():
        out = model(inp, labels=inp)
    n = inp.shape[1] - 1
    nlls.append(out.loss.item() * n)
    total_tok.append(n)
    if (i // STRIDE + 1) % 5 == 0:
        ppl = math.exp(sum(nlls) / sum(total_tok))
        print(f"  step {i//STRIDE+1}: PPL={ppl:.4f}", flush=True)

ppl = math.exp(sum(nlls) / sum(total_tok))
print(f"\n=== GPTQ 4bit PPL (via bf16 dequant) = {ppl:.4f} ===", flush=True)
print(f"Tokens: {sum(total_tok)}, Time: {time.time()-t1:.1f}s", flush=True)
