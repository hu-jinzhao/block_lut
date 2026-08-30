"""对比 offload 文件里的 indices vs 重新量化，找出不一致的根因"""
import os, sys, struct
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch

INDEX = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/lut_moe_index"
DATA = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/lut_moe_param"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
ORIG_CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"

# Load LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

# Compute thresholds (same as _load_lut)
midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)

def to_mono(u16):
    return np.where(u16 & 0x8000, ~u16, u16 ^ np.uint16(0x8000)).astype(np.uint16)

def quantize_py(tensor):
    """Same as model_offload._quantize_weight_to_lut"""
    flat_u16 = tensor.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
    flat_mono = np.where(flat_u16 & 0x8000, ~flat_u16, flat_u16 ^ np.uint16(0x8000)).astype(np.uint16)
    return np.searchsorted(thresholds, flat_mono).astype(np.uint8)

def recover_py(indices):
    reco_u16 = lut_u16[indices]
    return torch.from_numpy(reco_u16.astype(np.int16)).view(torch.bfloat16)

# Read index
def read_index(path):
    with open(path, "rb") as f:
        num_entries = struct.unpack("<I", f.read(4))[0]
        entries = {}
        for _ in range(num_entries):
            key = struct.unpack("<I", f.read(4))[0]
            tid = struct.unpack("<I", f.read(4))[0]
            assert tid == key
            shape_size = struct.unpack("<q", f.read(8))[0]
            shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_size)]
            size = struct.unpack("<Q", f.read(8))[0]
            num_elements = struct.unpack("<Q", f.read(8))[0]
            f.read(6)  # options
            offload_file_id = struct.unpack("<i", f.read(4))[0]
            num_file_chunks = struct.unpack("<i", f.read(4))[0]
            exp_size = struct.unpack("<I", f.read(4))[0]
            exp_offsets = [struct.unpack("<Q", f.read(8))[0] for _ in range(exp_size)]
            comp_size = struct.unpack("<I", f.read(4))[0]
            compressed_sizes = [struct.unpack("<Q", f.read(8))[0] for _ in range(comp_size)]
            orig_size = struct.unpack("<I", f.read(4))[0]
            original_sizes = [struct.unpack("<Q", f.read(8))[0] for _ in range(orig_size)]
            sm_offset = struct.unpack("<Q", f.read(8))[0]
            sm_size = struct.unpack("<Q", f.read(8))[0]
            entries[key] = {
                "shape": shape, "num_elements": num_elements,
                "exp_offsets": exp_offsets, "compressed_sizes": compressed_sizes,
                "original_sizes": original_sizes,
            }
        return entries

entries = read_index(INDEX)

import json
with open("/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/name_id_map.json") as f:
    name_id_map = json.load(f)
id_name_map = {int(v): k for k, v in name_id_map.items()}

# Pick first expert weight
target_tid = None
target_name = None
for tid, meta in entries.items():
    if meta["num_elements"] == 2883584 and sum(meta["original_sizes"]) > 0:
        if tid in id_name_map:
            target_tid = tid
            target_name = id_name_map[tid]
            break

print(f"Testing tensor {target_tid}: {target_name}")

# Read from offload file
meta = entries[target_tid]
total_exp = sum(meta["original_sizes"])
exp_data = bytearray(total_exp)
with open(DATA, "rb") as f:
    off = 0
    for ci in range(5):
        f.seek(meta["exp_offsets"][ci])
        chunk = f.read(meta["compressed_sizes"][ci])
        exp_data[off:off+len(chunk)] = chunk
        off += len(chunk)

file_indices = np.frombuffer(exp_data, dtype=np.uint8)
print(f"File indices: range=[{file_indices.min()}, {file_indices.max()}], unique={len(np.unique(file_indices))}")

# Load original weight
from safetensors import safe_open
orig = None
for i in range(1, 9):
    ckpt = os.path.join(ORIG_CKPT, f"model-0000{i}-of-00008.safetensors")
    if os.path.exists(ckpt):
        with safe_open(ckpt, framework="pt", device="cpu") as sf:
            if target_name in sf.keys():
                orig = sf.get_tensor(target_name)
                break

orig_bf16 = orig.to(torch.bfloat16)
print(f"Original: dtype={orig_bf16.dtype}, shape={list(orig_bf16.shape)}")

# Re-quantize
fresh_indices = quantize_py(orig_bf16)
print(f"Fresh indices: range=[{fresh_indices.min()}, {fresh_indices.max()}], unique={len(np.unique(fresh_indices))}")

# COMPARE
match = (file_indices == fresh_indices).mean()
print(f"\nIndices match: {match*100:.1f}%")
if match < 1.0:
    diff_positions = np.where(file_indices != fresh_indices)[0]
    print(f"First 10 mismatch positions: {diff_positions[:10]}")
    for pos in diff_positions[:5]:
        print(f"  pos {pos}: file={file_indices[pos]}, fresh={fresh_indices[pos]}, "
              f"u16={orig_bf16.view(torch.int16).numpy().astype(np.uint16).ravel()[pos]}")

# Compare PSNR
reco_file = recover_py(file_indices).float().numpy().reshape(orig.shape)
reco_fresh = recover_py(fresh_indices).float().numpy().reshape(orig.shape)
orig_f32 = orig_bf16.float().numpy()

mse_file = ((orig_f32 - reco_file) ** 2).mean()
mse_fresh = ((orig_f32 - reco_fresh) ** 2).mean()
psnr_file = 10 * np.log10(orig_f32.var() / mse_file) if mse_file > 0 else float('inf')
psnr_fresh = 10 * np.log10(orig_f32.var() / mse_fresh) if mse_fresh > 0 else float('inf')
print(f"\nFile indices PSNR: {psnr_file:.1f} dB")
print(f"Fresh indices PSNR: {psnr_fresh:.1f} dB")

# Check if the mismatch is systematic (offset, byte swap, etc.)
print(f"\nMean index diff: {(file_indices.astype(float) - fresh_indices.astype(float)).mean():.2f}")
print(f"Abs diff distribution:")
diffs = np.abs(file_indices.astype(int) - fresh_indices.astype(int))
for thresh in [0, 1, 5, 10, 50, 100]:
    print(f"  diff > {thresh}: {(diffs > thresh).mean()*100:.1f}%")

# Check: maybe file indices are shifted by 1 chunk?
# Let's compare chunk by chunk
print("\n--- Chunk-by-chunk comparison ---")
off = 0
for ci in range(5):
    chunk_size = meta["original_sizes"][ci]
    file_chunk = file_indices[off:off+chunk_size]
    fresh_chunk = fresh_indices[off:off+chunk_size]
    match = (file_chunk == fresh_chunk).mean()
    print(f"  Chunk {ci}: size={chunk_size}, match={match*100:.1f}%")
    off += chunk_size

print("\nDone.")
