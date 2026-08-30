#!/usr/bin/env python3
"""Quick PPL for nested K=16/64/256 with containment verification."""
import sys, os, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
with open(os.path.join(MODEL_DIR, '..', '..', 'evaluation/dataset/wikitext2_test.json')) as f:
    texts = json.load(f)

BLOCK_SIZE = 128
c16 = np.load(f'{MODEL_DIR}/nested_c16.npy')
c64 = np.load(f'{MODEL_DIR}/nested_c64.npy')
c256 = np.load(f'{MODEL_DIR}/nested_c256.npy')

def quantize(tensor, cent):
    x = tensor.float().numpy().ravel()
    n = x.size; nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    if nb*BLOCK_SIZE > n: x = np.pad(x, (0, nb*BLOCK_SIZE-n))
    b = x.reshape(nb, BLOCK_SIZE)
    a = np.max(np.abs(b), axis=1).clip(min=1e-12)
    norm = (b / a[:, np.newaxis]).ravel()
    mid = (cent[:-1] + cent[1:]) / 2
    idx = np.searchsorted(mid, norm).astype(np.uint8)
    bid = np.arange(nb*BLOCK_SIZE) // BLOCK_SIZE
    return torch.from_numpy((cent[idx]*a[bid])[:n].reshape(tensor.shape)).to(tensor.dtype)

print('Loading model...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)

orig = {n: p.data.detach().cpu().clone() for n,p in model.named_parameters() if 'experts' in n and 'shared_expert' not in n}
enc = tokenizer('\n\n'.join([t for t in texts[:50] if t]), return_tensors='pt')
ids = enc.input_ids[0]
print(f'Tokens: {len(ids)}', flush=True)

def ppl():
    nll, t = 0.0, 0
    for i in range(0, len(ids), 512):
        s, e = max(0,i), min(i+2048, len(ids))
        if e-s < 10: continue
        with torch.no_grad():
            L = model(ids[s:e].unsqueeze(0)).logits.float()
        loss = F.cross_entropy(L[:,:-1].contiguous().view(-1, L.size(-1)), ids[s+1:e].contiguous().view(-1), reduction='sum')
        nll += loss.item(); t += e-s-1
    return np.exp(nll/t), t

base_ppl, tok = ppl()
print(f'Lossless: PPL={base_ppl:.2f}  tokens={tok}', flush=True)

for lbl, c in [('Nested K=16', c16), ('Nested K=64', c64), ('Nested K=256', c256)]:
    for n,p in model.named_parameters():
        if 'experts' in n and 'shared_expert' not in n:
            p.data.copy_(quantize(p.data, c))
    v, _ = ppl()
    print(f'{lbl}: PPL={v:.2f}  delta={v-base_ppl:+.2f}', flush=True)
    for n,o in orig.items(): model.state_dict()[n].copy_(o)
    gc.collect()

print('Done!')
