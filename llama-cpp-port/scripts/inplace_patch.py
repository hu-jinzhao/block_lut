#!/usr/bin/env python3
"""Replace expert tensor data in GGUF in-place. BlockLUT < BF16, so always fits."""
import sys, os, struct, numpy as np
import torch

sys.path.insert(0, '/home/hh/llama.cpp/gguf-py')
from gguf import GGUFReader, GGMLQuantizationType

GGUF_PATH = sys.argv[1]
LUT_PATH = '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy'

print(f'Backing up {GGUF_PATH}...')
os.system(f'cp "{GGUF_PATH}" "{GGUF_PATH}.bak"')

lut = np.load(LUT_PATH)
mid_gpu = torch.from_numpy((lut[:-1] + lut[1:]) / 2.0).float().to('cuda')
print(f'LUT: {len(lut)} entries')

reader = GGUFReader(GGUF_PATH, 'r')
EXPERT_PATTERNS = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'gate_up_exps']

f = open(GGUF_PATH, 'r+b')

n_exp = 0
for ti in reader.tensors:
    if not any(p in ti.name for p in EXPERT_PATTERNS):
        continue

    # Quantize on GPU
    raw = ti.data.tobytes()
    t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().to('cuda').ravel()
    n = t.numel(); bs = 128; nb = (n + bs - 1) // bs
    if nb * bs > n: t = torch.nn.functional.pad(t, (0, nb * bs - n))
    b = t.view(nb, bs)
    a = b.abs().max(dim=1).values.clamp(min=1e-12)
    nrm = (b / a[:, None]).ravel()
    idx = torch.bucketize(nrm, mid_gpu).to(torch.uint8)[:n]
    abs_bytes = a.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint8).tobytes()
    idx_bytes = idx.cpu().numpy().tobytes()
    packed = idx_bytes + abs_bytes
    del t, b, a, nrm, idx

    # Overwrite data in-place (BlockLUT is always smaller than BF16)
    f.seek(ti.data_offset)
    f.write(packed)

    # Update tensor type from BF16(30) to BLOCKLUT8(43) in TI section
    name_bytes = ti.name.encode('utf-8')
    # TI entry layout:
    #   name_len: uint64 (8 bytes)
    #   name: name_len bytes
    #   n_dims: uint32 (4 bytes)
    #   dims: uint64 * n_dims (8 * n_dims bytes)
    #   type: uint32 (4 bytes) ← THIS
    #   offset: uint64 (8 bytes)
    n_dims = len(ti.shape)
    type_offset = ti.field.offset + 8 + len(name_bytes) + 4 + 8 * n_dims
    f.seek(type_offset)
    f.write(struct.pack('<I', 43))  # BLOCKLUT8

    n_exp += 1
    if n_exp % 10 == 0:
        sys.stdout.write(f'\rPatched {n_exp} experts...')
        sys.stdout.flush()

f.close()
print(f'\nDone! Patched {n_exp} expert tensors in-place.')
print(f'Original backed up to {GGUF_PATH}.bak')
