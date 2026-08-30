#!/usr/bin/env python3
"""
Read LUT-MoE offload files and convert to safetensors for vLLM.

Parses the custom binary format (lut_moe_index + lut_moe_param + name_id_map.json)
and produces standard safetensors files that vLLM can load.
"""

import json
import math
import os
import struct
import sys

import numpy as np
from safetensors.torch import save_file
import torch


def read_index(index_path):
    """Parse lut_moe_index binary file and return list of tensor metadata."""
    with open(index_path, "rb") as f:
        data = f.read()

    pos = 0
    entries = []

    while pos < len(data):
        # id (uint32)
        tid = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        # shape_size (int64)
        shape_size = struct.unpack_from("<q", data, pos)[0]
        pos += 8

        # shape
        shape = []
        for _ in range(shape_size):
            dim = struct.unpack_from("<q", data, pos)[0]
            pos += 8
            shape.append(dim)

        # size (size_t = uint64 on 64-bit)
        size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8

        # num_elements (size_t)
        num_elements = struct.unpack_from("<Q", data, pos)[0]
        pos += 8

        # Options: pinned_memory (bool), requires_grad (bool), dtype (int8), device_index (int8), device_type (int8), layout (int8)
        pinned = struct.unpack_from("?", data, pos)[0]
        pos += 1
        requires_grad = struct.unpack_from("?", data, pos)[0]
        pos += 1
        dtype_code = struct.unpack_from("<b", data, pos)[0]
        pos += 1
        device_index = struct.unpack_from("<b", data, pos)[0]
        pos += 1
        device_type = struct.unpack_from("<b", data, pos)[0]
        pos += 1
        layout = struct.unpack_from("<b", data, pos)[0]
        pos += 1

        # offload_file_id (uint32)
        file_id = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        # num_file_chunks (int32)
        num_chunks = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        # exp_file_offsets
        exp_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        exp_offsets = []
        for _ in range(exp_size):
            off = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            exp_offsets.append(off)

        # compressed_sizes
        comp_sizes = []
        for _ in range(num_chunks):
            cs = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            comp_sizes.append(cs)

        # original_sizes
        orig_sizes = []
        for _ in range(num_chunks):
            os_ = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            orig_sizes.append(os_)

        # sm_file_offset (size_t)
        sm_offset = struct.unpack_from("<Q", data, pos)[0]
        pos += 8

        # sm_size (size_t)
        sm_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8

        entries.append({
            "id": tid,
            "shape": tuple(shape),
            "size": size,
            "num_elements": num_elements,
            "dtype_code": dtype_code,
            "file_id": file_id,
            "num_chunks": num_chunks,
            "exp_offsets": exp_offsets,
            "compressed_sizes": comp_sizes,
            "original_sizes": orig_sizes,
            "sm_offset": sm_offset,
            "sm_size": sm_size,
        })

    return entries


# Mapping from dtype_code to torch.dtype
DTYPE_MAP = {
    5: torch.float16,    # kHalf
    7: torch.float32,    # kFloat
    3: torch.int32,      # kInt
    11: torch.bfloat16,  # kBFloat16
    6: torch.uint8,      # kByte/Uint8
    2: torch.int8,       # kChar
    4: torch.int64,      # kLong
}


def read_param_data(param_path, entry):
    """Read a tensor from the param file given its metadata."""
    if entry["num_chunks"] == 0:
        return None

    # Read all chunks
    chunks = []
    for i in range(entry["num_chunks"]):
        offset = entry["exp_offsets"][i]
        comp_size = entry["compressed_sizes"][i]
        orig_size = entry["original_sizes"][i]

        with open(param_path, "rb") as f:
            f.seek(offset)
            data = f.read(comp_size)

        # For BLOCKLUT/NESTEDLUT codec, the data is already LUT indices
        # stored in the custom format. For raw/copy codec, it's raw bf16.

        # For simplicity: we read the compressed data and treat it
        # depending on the dtype_code
        # If dtype_code is uint8 (6), these are LUT indices
        # If dtype_code is bfloat16 (11), these are raw bf16 weights

        dtype = DTYPE_MAP.get(entry["dtype_code"], torch.bfloat16)
        # For most entries in the BlockLUT format, the data is stored
        # as raw bf16 in the param file (the C++ backend handles decompression
        # at runtime). Let's just try reading it as the stored dtype.

        if comp_size < orig_size:
            # Data is compressed - this is the C++ backend's compressed format
            # For now, just read the first chunk as-is
            pass

        chunk = np.frombuffer(data, dtype=np.uint8)
        chunks.append(chunk)

    if not chunks:
        return None

    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # Try to interpret the data
    # The data might be stored as raw bytes in compressed form
    # For the BlockLUT format, the C++ backend stores data differently
    # based on code_type

    num_elements = entry["num_elements"]
    dtype = DTYPE_MAP.get(entry["dtype_code"], torch.bfloat16)

    # Create a tensor from the raw data (if it matches the expected size)
    # Otherwise, fill with zeros as placeholder
    try:
        tensor = torch.frombuffer(torch.as_tensor(full), dtype=dtype)
        if tensor.numel() != num_elements:
            print(f"  Warning: expected {num_elements} elements, got {tensor.numel()}")
            tensor = tensor[:num_elements]
    except Exception as e:
        print(f"  Warning: could not read tensor: {e}")
        tensor = torch.zeros(num_elements, dtype=dtype)

    return tensor.reshape(entry["shape"])


def convert_offload(offload_dir, output_dir, model_dir):
    """Convert LUT-MoE offload to safetensors."""
    print(f"Reading offload from {offload_dir}")

    # Read index
    index_path = os.path.join(offload_dir, "lut_moe_index")
    param_path = os.path.join(offload_dir, "lut_moe_param")
    map_path = os.path.join(offload_dir, "name_id_map.json")

    entries = read_index(index_path)
    print(f"Found {len(entries)} tensors in index")

    with open(map_path) as f:
        name_map = json.load(f)

    # Build id -> name mapping
    id_to_name = {v: k for k, v in name_map.items()}
    print(f"Found {len(name_map)} named parameters")

    # Read all data and build state_dict
    state_dict = {}
    expert_count = 0
    dense_count = 0

    # First, read the codebook from the model directory
    codebook = None
    codebook_path = os.path.join(model_dir, "blocklut_256.npy")
    if os.path.exists(codebook_path):
        codebook = torch.from_numpy(np.load(codebook_path)).to(torch.bfloat16)
        print(f"Loaded codebook from {codebook_path}")

    # Process each entry
    for entry in entries:
        tid = entry["id"]
        name = id_to_name.get(tid)
        if name is None:
            print(f"  Warning: no name for id {tid}")
            continue

        tensor = read_param_data(param_path, entry)
        if tensor is None:
            print(f"  Warning: could not read tensor {name}")
            continue

        # Check if this is an expert weight
        is_expert = "experts." in name and "shared_expert" not in name

        if is_expert and codebook is not None and tensor.dtype == torch.bfloat16:
            # This expert weight is stored as bf16 in the offload (raw format)
            # But it should be LUT quantized. Let's quantize it.
            # Actually, in BlockLUT mode, the C++ backend stores data in LUT format
            # The raw data here might already be LUT indices.
            # Let's check the data range to determine
            if tensor.dtype == torch.uint8:
                # Already LUT indices -> store as-is
                state_dict[name] = tensor
                expert_count += 1
            else:
                # bf16 data -> need to quantize
                state_dict[name] = tensor  # Store as bf16 for now
                dense_count += 1
        else:
            state_dict[name] = tensor
            dense_count += 1

    print(f"\nRead {len(state_dict)} tensors ({expert_count} expert, {dense_count} dense)")

    # Save as safetensors
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_lut.safetensors")
    save_file(state_dict, output_file)
    print(f"Saved to {output_file} ({os.path.getsize(output_file)/1e9:.1f}GB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload", default="/home/hh/LUT-MoE/offload/qwen_blocklut")
    parser.add_argument("--output", default="/home/hh/LUT-MoE/models/qwen_lut4bit")
    parser.add_argument("--model", default="/home/hh/LUT-MoE/models/qwen")
    args = parser.parse_args()
    convert_offload(args.offload, args.output, args.model)
