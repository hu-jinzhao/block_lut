#!/usr/bin/env python3
"""Replace expert tensors in a standard GGUF with BlockLUT quantization."""
import sys, os, json, gc, numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/hh/llama.cpp/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

IN = sys.argv[1]
OUT = sys.argv[2]
LUT = np.load(sys.argv[3]) if len(sys.argv) > 3 else None

if LUT is None:
    # Check model dir for pre-computed LUT
    model_dir = os.path.dirname(IN) if os.path.dirname(IN) else '.'
    for fname in ['blocklut_256.npy', 'blocklut_192.npy', 'blocklut_128.npy']:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            LUT = np.load(p)
            print(f'Loaded LUT: {p}')
            break
if LUT is None:
    print('ERROR: no LUT found')
    sys.exit(1)

mid_gpu = torch.from_numpy((LUT[:-1] + LUT[1:]) / 2.0).float().to('cuda')
print(f'LUT: {len(LUT)} entries, min={LUT.min():.4f}, max={LUT.max():.4f}')

# Read base GGUF
reader = GGUFReader(IN)
print(f'Base GGUF: {len(reader.tensors)} tensors')

# Determine expert tensors by name
expert_name_patterns = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'gate_up_exps']
tensor_map = {t.name: t for t in reader.tensors}
expert_names = [n for n in tensor_map if any(p in n for p in expert_name_patterns)]
print(f'Found {len(expert_names)} expert tensors')

# Create output GGUF - need to replicate input metadata
# (simplest: copy base file then overwrite expert tensor data)
# For now, read all tensor data, quantize experts, write new GGUF

print(f'Reading tensor data...')
all_data = {}
for name, t in tqdm(tensor_map.items()):
    all_data[name] = t.data

print(f'Quantizing experts on GPU...')
n_exp = 0
for name in tqdm(expert_names):
    data = all_data[name]
    # Convert numpy to torch, quantize on GPU
    t = torch.from_numpy(data).float().to('cuda').ravel()
    n = t.numel()
    bs = 128; nb = (n + bs - 1) // bs
    if nb * bs > n:
        t = torch.nn.functional.pad(t, (0, nb * bs - n))
    blocks = t.view(nb, bs)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normed = (blocks / absmax[:, None]).ravel()
    indices = torch.bucketize(normed, mid_gpu).to(torch.uint8)[:n]
    absmax_bytes = absmax.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint8).tobytes()
    idx_bytes = indices.cpu().numpy().tobytes()
    all_data[name] = np.frombuffer(idx_bytes + absmax_bytes, dtype=np.uint8)
    n_exp += 1
    del t, blocks, absmax, normed, indices
    gc.collect()

print(f'Writing BlockLUT GGUF ({n_exp} experts)...')
# Write new GGUF with same metadata + BLOCKLUT tensors
w = GGUFWriter(OUT, reader.fields.get('general.architecture', 'unknown'))
# Copy all metadata and tensors
# (Using a simple approach: write each tensor with type BLOCKLUT)
# ... (this is the complex part - need to properly replicate metadata)

print('TODO: finish writer replication')
print(f'Quantized {n_exp}/{len(expert_names)} expert tensors on GPU')
