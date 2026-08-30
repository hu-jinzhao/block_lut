"""诊断 LUT offload 文件: 读取 index + data, 验证 indices 恢复后的精度"""
import os, sys, struct
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch

INDEX = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/lut_moe_index"
DATA = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/lut_moe_param"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
ORIG_CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"

# --- Load LUT ---
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

def read_index(path):
    with open(path, "rb") as f:
        num_entries = struct.unpack("<I", f.read(4))[0]
        print(f"Index entries: {num_entries}")
        entries = {}
        for _ in range(num_entries):
            key = struct.unpack("<I", f.read(4))[0]
            tid = struct.unpack("<I", f.read(4))[0]
            assert tid == key, f"Key mismatch: {key} != {tid}"
            shape_size = struct.unpack("<q", f.read(8))[0]
            shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_size)]
            size = struct.unpack("<Q", f.read(8))[0]
            num_elements = struct.unpack("<Q", f.read(8))[0]
            # write_options: 6 bytes
            pinned = f.read(1); requires_grad = f.read(1)
            dtype_byte = struct.unpack("<b", f.read(1))[0]
            device_index = f.read(1); device_type = f.read(1); layout = f.read(1)
            offload_file_id = struct.unpack("<i", f.read(4))[0]
            num_file_chunks = struct.unpack("<i", f.read(4))[0]

            exp_size = struct.unpack("<I", f.read(4))[0]
            exp_file_offsets = [struct.unpack("<Q", f.read(8))[0] for _ in range(exp_size)]
            comp_size = struct.unpack("<I", f.read(4))[0]
            compressed_sizes = [struct.unpack("<Q", f.read(8))[0] for _ in range(comp_size)]
            orig_size = struct.unpack("<I", f.read(4))[0]
            original_sizes = [struct.unpack("<Q", f.read(8))[0] for _ in range(orig_size)]
            sm_file_offset = struct.unpack("<Q", f.read(8))[0]
            sm_size = struct.unpack("<Q", f.read(8))[0]

            entries[key] = {
                "shape": shape, "num_elements": num_elements, "size": size,
                "dtype": dtype_byte, "nchunks": num_file_chunks,
                "exp_offsets": exp_file_offsets, "compressed_sizes": compressed_sizes,
                "original_sizes": original_sizes,
                "sm_offset": sm_file_offset, "sm_size": sm_size,
            }
        return entries

entries = read_index(INDEX)

# Categorize
nchunks_dist = {}
total_exp_zero = 0
total_exp_nonzero = 0
shapes_dist = {}
for k, v in entries.items():
    total_exp = sum(v["original_sizes"])
    shape_key = str(v["shape"])
    shapes_dist[shape_key] = shapes_dist.get(shape_key, 0) + 1
    nchunks_dist[v["nchunks"]] = nchunks_dist.get(v["nchunks"], 0) + 1
    if total_exp == 0:
        total_exp_zero += 1
    else:
        total_exp_nonzero += 1

print(f"nchunks distribution: {nchunks_dist}")
print(f"Shape distribution: {shapes_dist}")
print(f"Tensors with zero exp data: {total_exp_zero}, non-zero: {total_exp_nonzero}")
print(f"File size: {os.path.getsize(DATA)/1e9:.2f} GB")

# Load name map
import json
with open("/home/hh/zip_Moe/LUT_MoE/offload/qwen_lut/name_id_map.json") as f:
    name_id_map = json.load(f)
id_name_map = {int(v): k for k, v in name_id_map.items()}

# Find a tensor with shape [1408, 2048]
target_tid = None
for tid, meta in entries.items():
    if meta["shape"] == [1408, 2048] and sum(meta["original_sizes"]) > 0:
        target_tid = tid
        break

if target_tid is None:
    print("ERROR: No [1408, 2048] tensor with non-zero data found!")
    # Try any tensor with num_elements > 1M
    for tid, meta in entries.items():
        if meta["num_elements"] > 1000000 and sum(meta["original_sizes"]) > 0:
            target_tid = tid
            break

if target_tid:
    meta = entries[target_tid]
    name = id_name_map.get(target_tid, f"unknown_{target_tid}")
    print(f"\n--- Testing tensor {target_tid} ({name}) ---")
    print(f"  shape: {meta['shape']}, elements: {meta['num_elements']}")
    print(f"  original_sizes: {meta['original_sizes']}")
    print(f"  compressed_sizes: {meta['compressed_sizes']}")
    print(f"  exp_offsets: {meta['exp_offsets']}")

    total_exp = sum(meta["original_sizes"])
    exp_data = bytearray(total_exp)
    with open(DATA, "rb") as f:
        off = 0
        for ci in range(meta["nchunks"]):
            f.seek(meta["exp_offsets"][ci])
            chunk = f.read(meta["compressed_sizes"][ci])
            exp_data[off:off+len(chunk)] = chunk
            off += len(chunk)

    indices = np.frombuffer(exp_data, dtype=np.uint8)
    print(f"  Indices: min={indices.min()}, max={indices.max()}, unique={len(np.unique(indices))}")

    # Recover
    reco_u16 = lut_u16[indices]
    recovered = torch.from_numpy(reco_u16.astype(np.int16)).view(torch.bfloat16)

    # Load original
    from safetensors import safe_open
    orig = None
    for i in range(1, 9):
        ckpt = os.path.join(ORIG_CKPT, f"model-0000{i}-of-00008.safetensors")
        if os.path.exists(ckpt):
            with safe_open(ckpt, framework="pt", device="cpu") as sf:
                if name in sf.keys():
                    orig = sf.get_tensor(name)
                    break

    if orig is not None:
        orig_bf16 = orig.to(torch.bfloat16)
        reco_f32 = recovered.float().numpy().reshape(orig.shape)
        orig_f32 = orig_bf16.float().numpy()
        mse = ((orig_f32 - reco_f32) ** 2).mean()
        psnr = 10 * np.log10(orig_f32.var() / mse) if mse > 0 else float('inf')
        max_err = np.abs(orig_f32 - reco_f32).max()
        exact = (orig_f32 == reco_f32).mean()
        print(f"  PSNR: {psnr:.1f} dB, max_err: {max_err:.6f}, exact: {exact*100:.1f}%")

        # Sample values
        print(f"  First 10 orig: {orig_f32.ravel()[:10]}")
        print(f"  First 10 reco: {reco_f32.ravel()[:10]}")
        print(f"  First 10 indices: {indices[:10]}")

        # Check if LUT values match
        print(f"  lut[0]={lut_u16[0]} (bf16={lut_bf16[0]:.6f}), lut[128]={lut_u16[128]} (bf16={lut_bf16[128]:.6f}), lut[255]={lut_u16[255]} (bf16={lut_bf16[255]:.6f})")
    else:
        print(f"  WARNING: Could not find original tensor for {name}")
else:
    print("No suitable tensor found!")

# Also test first 10 large tensors
print("\n--- PSNR for first 10 expert weight tensors ---")
count = 0
for tid, meta in entries.items():
    if meta["num_elements"] < 1000000 or sum(meta["original_sizes"]) == 0:
        continue
    if tid not in id_name_map:
        continue
    if count >= 10:
        break
    name = id_name_map[tid]
    total_exp = sum(meta["original_sizes"])
    exp_data = bytearray(total_exp)
    with open(DATA, "rb") as f:
        off = 0
        for ci in range(meta["nchunks"]):
            f.seek(meta["exp_offsets"][ci])
            chunk = f.read(meta["compressed_sizes"][ci])
            exp_data[off:off+len(chunk)] = chunk
            off += len(chunk)
    indices = np.frombuffer(exp_data, dtype=np.uint8)
    reco_u16 = lut_u16[indices]
    recovered = torch.from_numpy(reco_u16.astype(np.int16)).view(torch.bfloat16)

    orig = None
    for i in range(1, 9):
        ckpt = os.path.join(ORIG_CKPT, f"model-0000{i}-of-00008.safetensors")
        if os.path.exists(ckpt):
            with safe_open(ckpt, framework="pt", device="cpu") as sf:
                if name in sf.keys():
                    orig = sf.get_tensor(name)
                    break
    if orig is not None:
        reco_f32 = recovered.float().numpy().reshape(orig.shape)
        orig_f32 = orig.to(torch.bfloat16).float().numpy()
        mse = ((orig_f32 - reco_f32) ** 2).mean()
        psnr = 10 * np.log10(orig_f32.var() / mse) if mse > 0 else float('inf')
        print(f"  [{count}] {name.split('.')[-1]}: PSNR={psnr:.1f} dB")
    count += 1

print("\nDone.")
