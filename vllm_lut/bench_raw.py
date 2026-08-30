#!/usr/bin/env python3
"""Measure raw MoE inference speed without vLLM overhead."""
import os, sys, time, json, numpy as np
sys.path.insert(0, '/home/hh/LUT-MoE')
os.environ['CUDA_HOME'] = '/home/hh/miniconda3/envs/lut_moe_cu124'

import torch
import torch.nn.functional as F

from vllm_lut.cuda_lut_gemv import lut_gemv as _gemv

# Load model config
with open('/home/hh/LUT-MoE/models/qwen_lut4bit/config.json') as f:
    cfg = json.load(f)

hidden = cfg['hidden_size']
inter = cfg.get('moe_intermediate_size', 1408)
n_layers = cfg['num_hidden_layers']
n_experts = cfg.get('num_experts', 60)
top_k = cfg.get('num_experts_per_tok', 4)
n_selected = top_k

# Load LUT weights from safetensors
from safetensors import safe_open
import glob
files = sorted(glob.glob('/home/hh/LUT-MoE/models/qwen_lut4bit/model-*_lut.safetensors'))

# Load codebook
cb = torch.from_numpy(np.load('/home/hh/LUT-MoE/models/qwen_lut4bit/blocklut_256.npy')).to(torch.bfloat16)

# Load weights and organize by layer
layer_w13 = []  # list of [n_experts, 2*inter, hidden] uint8
layer_w2 = []   # list of [n_experts, hidden, inter] uint8
layer_a13 = []  # list of [n_experts, blocks_13] bf16
layer_a2 = []   # list of [n_experts, blocks_2] bf16

# Initialize per-layer lists
for l in range(n_layers):
    layer_w13.append({})
    layer_w2.append({})
    layer_a13.append({})
    layer_a2.append({})

for f in files:
    with safe_open(f, framework="pt", device="cpu") as sf:
        for name in sf.keys():
            tensor = sf.get_tensor(name)
            # Parse: model.layers.{l}.mlp.experts.{e}.{type}
            parts = name.split('.')
            if len(parts) >= 6 and parts[2] == 'layers':
                l = int(parts[3])
                e = int(parts[5])
                wtype = parts[6]

                if wtype == 'q_weight':
                    # indices stored directly
                    continue  # skip - we need absmax too
                elif wtype == 'absmax':
                    if 'down_proj' in name:
                        pass  # w2 absmax
                    elif 'gate_proj' in name or 'up_proj' in name:
                        pass  # w13 absmax

print("Need to restructure weight loading...")
print(f"Model: {n_layers} layers, {n_experts} experts, top-{top_k}")
print(f"hidden={hidden}, inter={inter}")

# Simpler approach: just use the ORIGINAL bf16 weights to test raw throughput
# The weights in the LUT directory still have the original .weight files for dense layers
# But expert weights are in .q_weight format

# Let me just test the LUT GEMV speed directly and estimate
x = torch.randn(1, hidden, dtype=torch.bfloat16, device='cuda')
topk_ids = torch.randint(0, n_experts, (1, top_k), device='cuda')
topk_weights = torch.softmax(torch.randn(1, top_k, device='cuda'), dim=-1)

# Create fake LUT weights
w13 = torch.randint(0, 256, (n_experts, 2*inter, hidden), dtype=torch.uint8, device='cuda')
w2 = torch.randint(0, 256, (n_experts, hidden, inter), dtype=torch.uint8, device='cuda')
a13 = torch.randn(n_experts, (2*inter*hidden+127)//128, dtype=torch.bfloat16, device='cuda').abs()
a2 = torch.randn(n_experts, (hidden*inter+127)//128, dtype=torch.bfloat16, device='cuda').abs()

# Warmup
for _ in range(10):
    for eid in torch.unique(topk_ids).tolist():
        _gemv(x[0], w13[eid], cb, a13[eid])
        _gemv(x[0], w2[eid], cb, a2[eid])
torch.cuda.synchronize()

# Measure MoE-only time
t0 = time.time()
for _ in range(200):
    for eid in torch.unique(topk_ids).tolist():
        gate_up = _gemv(x[0], w13[eid], cb, a13[eid])
        ni = inter
        activated = F.silu(gate_up[:ni]) * gate_up[ni:]
        out = _gemv(activated, w2[eid], cb, a2[eid])
torch.cuda.synchronize()
per = (time.time() - t0) / 200
print(f"\nRaw MoE (1 layer, {n_selected} experts): {per*1000:.1f}ms")
print(f"All {n_layers} layers: {per*n_layers*1000:.0f}ms")
print(f"Estimated tok/s (MoE only): {1/(per*n_layers):.1f}")
print(f"Estimated tok/s (+50ms overhead): {1/(per*n_layers+0.05):.1f}")
