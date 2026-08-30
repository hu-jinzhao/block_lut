#!/usr/bin/env python3
"""Read base GGUF, replace expert tensors with BlockLUT, write new GGUF."""
import sys, os, json, gc, numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/hh/llama.cpp/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType

IN = sys.argv[1]
OUT = sys.argv[2]

LUT_PATH = '/home/hh/LUT-MoE/models/qwen/blocklut_256.npy'
lut = np.load(LUT_PATH)
mid_gpu = torch.from_numpy((lut[:-1] + lut[1:]) / 2.0).float().to('cuda')
print(f'LUT: {len(lut)} entries')

# Read base GGUF
reader = GGUFReader(IN)
fields = reader.fields

# Determine architecture
arch = fields['general.architecture'].contents(0)
w = GGUFWriter(OUT, arch)

# Replicate all metadata
SKIP_KEYS = {'general.architecture', 'general.alignment', 'general.quantization_version', 'general.file_type',
             'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}
for key, field in fields.items():
    if key in SKIP_KEYS: continue
    try:
        ft = field.types[0]
        if ft == GGUFValueType.STRING: w.add_string(key, str(field.contents(0)))
        elif ft == GGUFValueType.ARRAY: w.add_array(key, list(field.contents()))
        elif ft == GGUFValueType.BOOL: w.add_bool(key, bool(field.contents(0)))
        elif ft in (GGUFValueType.UINT8, GGUFValueType.INT8): w.add_uint8(key, int(field.contents(0)))
        elif ft in (GGUFValueType.UINT16, GGUFValueType.INT16): w.add_uint16(key, int(field.contents(0)))
        elif ft in (GGUFValueType.UINT32, GGUFValueType.INT32): w.add_uint32(key, int(field.contents(0)))
        elif ft in (GGUFValueType.UINT64, GGUFValueType.INT64): w.add_uint64(key, int(field.contents(0)))
        elif ft == GGUFValueType.FLOAT32: w.add_float32(key, float(field.contents(0)))
        elif ft == GGUFValueType.FLOAT64: w.add_float64(key, float(field.contents(0)))
    except: pass

# Set alignment to match base GGUF
if 'general.alignment' in fields:
    w.add_custom_alignment(fields['general.alignment'].contents(0))
    print(f'Alignment: {fields["general.alignment"].contents(0)}')

# LUT-MoE metadata
w.add_bool('lut-moe.enabled', True)
w.add_uint32('lut-moe.block_size', 128)
w.add_uint32('lut-moe.k', 256)
lut_bf16 = torch.from_numpy(lut).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
w.add_array('lut-moe.lut_table', lut_bf16.tolist())

# Define expert tensor name patterns
EXPERT_PATTERNS = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'gate_up_exps']

# Process tensors
tensors_list = reader.tensors
n_exp = 0
for ti in tqdm(tensors_list, desc='Processing'):
    name = ti.name
    is_exp = any(p in name for p in EXPERT_PATTERNS)

    if is_exp:
        # Read BF16 data from base GGUF
        raw = ti.data.tobytes()  # contiguous bytes
        # Convert BF16 bytes to float32
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).to(torch.float32)
        n = t.numel()
        bs = 128; nb = (n + bs - 1) // bs
        if nb * bs > n:
            t = torch.nn.functional.pad(t, (0, nb * bs - n))
        # Move to GPU for quantization
        t = t.to('cuda')
        b = t.view(nb, bs)
        a = b.abs().max(dim=1).values.clamp(min=1e-12)
        nrm = (b / a[:, None]).ravel()
        idx = torch.bucketize(nrm, mid_gpu).to(torch.uint8)[:n].cpu().numpy().tobytes()
        abs_bytes = a.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint8).tobytes()
        packed = np.frombuffer(idx + abs_bytes, dtype=np.uint8)
        del t, b, a, nrm, idx
        w.add_tensor(name, packed, raw_shape=(len(packed),), raw_dtype=GGMLQuantizationType.BLOCKLUT8)
        n_exp += 1
    else:
        # Copy as raw BF16 bytes (data memmap shape already has doubled last dim)
        raw_data = np.frombuffer(ti.data.tobytes(), dtype=np.uint8)
        w.add_tensor(name, raw_data, raw_shape=ti.data.shape, raw_dtype=GGMLQuantizationType.BF16)
    gc.collect()

print(f'Writing GGUF ({n_exp} BlockLUT experts)...')
w.open_output_file()
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
print(f'Done: {OUT}')
