#!/usr/bin/env python3
"""PPL for GPTQ model using HuggingFace (CPU offloading)."""
import os, sys, json, math, time
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen_gptq"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
DTYPE = "auto"
MAX_LEN = 2048
STRIDE = 1024

if __name__ == '__main__':
    with open(DATASET) as f:
        texts = json.load(f)
    full_text = " ".join(texts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    enc = tokenizer.encode(full_text)
    print(f"WikiText-2: {len(enc)} tokens", flush=True)

    print("Loading GPTQ model (CPU offloading)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map='auto',
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    # PPL with sliding window
    nlls = []
    total_tok = 0
    t1 = time.time()

    for i in range(0, len(enc), STRIDE):
        end = min(i + MAX_LEN, len(enc))
        if end - i < 128:
            break
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
    print(f"\n=== GPTQ 4bit PPL = {ppl:.4f} ===", flush=True)
    print(f"Tokens: {total_tok}, Time: {time.time()-t1:.1f}s", flush=True)
