#!/usr/bin/env python3
"""Convert base GGUF to BlockLUT: copies all metadata/tensors, quantizes experts."""
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

# Read source GGUF
r = GGUFReader(IN)
arch = r.fields['general.architecture'].contents(0)

# Get alignment from source
alignment = 32
if 'general.alignment' in r.fields:
    alignment = int(r.fields['general.alignment'].contents(0))

w = GGUFWriter(OUT, arch)

# Set alignment first
w.add_custom_alignment(alignment)

# Copy metadata
for key, field in r.fields.items():
    if key.startswith('GGUF.') or key == 'general.alignment':
        continue
    try:
        ft = field.types[0]
        val = field.contents(0) if len(field) > 0 else field.contents()
        if ft == GGUFValueType.STRING: w.add_string(key, str(val))
        elif ft == GGUFValueType.ARRAY: w.add_array(key, list(field.contents() if len(field.types) > 1 else [val]))
        elif ft == GGUFValueType.BOOL: w.add_bool(key, bool(val))
        elif ft in (GGUFValueType.UINT8, GGUFValueType.INT8): w.add_uint8(key, int(val))
        elif ft in (GGUFValueType.UINT16, GGUFValueType.INT16): w.add_uint16(key, int(val))
        elif ft in (GGUFValueType.UINT32, GGUFValueType.INT32): w.add_uint32(key, int(val))
        elif ft in (GGUFValueType.UINT64, GGUFValueType.INT64): w.add_uint64(key, int(val))
        elif ft == GGUFValueType.FLOAT32: w.add_float32(key, float(val))
        elif ft == GGUFValueType.FLOAT64: w.add_float64(key, float(val))
    except Exception as e:
        print(f'skip {key}: {e}')

# LUT-MoE metadata
w.add_bool('lut-moe.enabled', True)
w.add_uint32('lut-moe.block_size', 128)
w.add_uint32('lut-moe.k', 256)
lut_u16 = torch.from_numpy(lut).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
w.add_array('lut-moe.lut_table', lut_u16.tolist())

# Process tensors
EXPERT_PATTERNS = ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'gate_up_exps']
n_exp = 0

for ti in tqdm(r.tensors, desc='Processing'):
    is_exp = any(p in ti.name for p in EXPERT_PATTERNS)

    if is_exp:
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
        packed = np.frombuffer(idx_bytes + abs_bytes, dtype=np.uint8)
        del t, b, a, nrm, idx

        # Use 1D byte count as raw_shape, BLOCKLUT type
        w.add_tensor(ti.name, packed, raw_shape=(len(packed),),
                     raw_dtype=GGMLQuantizationType.BLOCKLUT8)
        # Override shape to NumPy order (GGUFWriter reverses it for GGML order)
        key = list(w.tensors[-1].keys())[-1]
        w.tensors[-1][key].shape = tuple(ti.shape[::-1])
        n_exp += 1
    else:
        # Copy original data preserving type
        orig_type = GGMLQuantizationType(ti.tensor_type)
        raw_bytes = ti.data.tobytes()
        if ti.tensor_type == 0:  # F32
            # Shape in GGUF file order (reversed from numpy). Pass reversed.
            arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(ti.shape[::-1])
            w.add_tensor(ti.name, arr)
        else:  # BF16 (30): ti.data.shape has doubled last dim
            raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            w.add_tensor(ti.name, raw_arr, raw_shape=ti.data.shape,
                         raw_dtype=orig_type)
    gc.collect()

print(f'Writing GGUF ({n_exp} BlockLUT experts)...')
w.open_output_file()
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
print(f'Done: {OUT}')
