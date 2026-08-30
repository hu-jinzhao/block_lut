#!/usr/bin/env python3
"""Run official HF→GGUF converter with BlockLUT quantization monkey-patched."""
import sys, os, numpy as np
import torch

# Add paths
LLAMA_CPP = '/home/hh/llama.cpp'
sys.path.insert(0, f'{LLAMA_CPP}/gguf-py')
sys.path.insert(0, f'{LLAMA_CPP}')

from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.constants import Keys

MODEL = sys.argv[1]
OUT = sys.argv[2]

# Load LUT from model directory
LUT_PATH = f'{MODEL}/blocklut_256.npy'
lut = np.load(LUT_PATH)
mid_gpu = torch.from_numpy((lut[:-1] + lut[1:]) / 2.0).float().to('cuda')
print(f'LUT: {len(lut)} entries')

# Store for monkey-patch
original_add_tensor = None

def blocklut_add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
    """Monkey-patched add_tensor: quantize expert weights to BlockLUT."""
    is_expert = any(p in name for p in ['ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'gate_up_exps'])

    if is_expert:
        # Quantize this expert tensor on GPU
        t = torch.from_numpy(tensor).float().to('cuda')
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
        # Call original with BLOCKLUT type
        original_add_tensor(self, name, packed, raw_shape=(len(packed),),
                          raw_dtype=GGMLQuantizationType.BLOCKLUT8)
    else:
        # Non-expert: pass through as-is
        original_add_tensor(self, name, tensor, raw_shape=raw_shape,
                          raw_dtype=raw_dtype)

# Apply monkey-patch BEFORE importing the converter modules
GGUFWriter.add_tensor = blocklut_add_tensor
original_add_tensor = lambda self, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None: \
    GGUFWriter._original_add_tensor(self, name, tensor, raw_shape, raw_dtype, tensor_endianess)
# Need to save original
from gguf import gguf_writer
GGUFWriter._original_add_tensor = gguf_writer.GGUFWriter.add_tensor
gguf_writer.GGUFWriter.add_tensor = blocklut_add_tensor

# Now import and run the official converter
from conversion import get_model_class
from conversion.base import BaseModel
from gguf.constants import MODEL_ARCH

# Run the converter
print(f'Converting {MODEL} to {OUT} with BlockLUT experts...')

# Import and run the main conversion function
from convert_hf_to_gguf import main as convert_main
import argparse

# Build args
sys.argv = ['convert_hf_to_gguf.py', MODEL, '--outfile', OUT]

# Add LUT-MoE metadata to the writer before it starts
_original_init = GGUFWriter.__init__
def patched_init(self, path, arch, **kwargs):
    _original_init(self, path, arch, **kwargs)
    # Check if we already added
    if not hasattr(self, '_lut_moe_added'):
        self.add_bool('lut-moe.enabled', True)
        self.add_uint32('lut-moe.block_size', 128)
        self.add_uint32('lut-moe.k', 256)
        lut_bf16 = torch.from_numpy(lut).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
        self.add_array('lut-moe.lut_table', lut_bf16.tolist())
        self._lut_moe_added = True
        print('Added LUT-MoE metadata')
GGUFWriter.__init__ = patched_init

# Run converter
try:
    convert_main()
    print(f'Done: {OUT}')
except SystemExit:
    print(f'Done (SystemExit): {OUT}')
