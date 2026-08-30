#!/usr/bin/env python3
"""bf16 baseline PPL with same settings as GPTQ/LUT eval."""
import os, sys, json, math, time
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
MAX_LEN = 2048  # Same as GPTQ/LUT eval
STRIDE = 1024

with open(DATASET) as f:
    texts = json.load(f)
full_text = " ".join(texts)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
enc = tokenizer.encode(full_text)
print(f"WikiText-2: {len(enc)} tokens", flush=True)

print("Loading bf16 model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    device_map='auto', trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

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
print(f"\n=== bf16 PPL = {ppl:.4f} ===", flush=True)
print(f"Tokens: {total_tok}, Time: {time.time()-t1:.1f}s", flush=True)
