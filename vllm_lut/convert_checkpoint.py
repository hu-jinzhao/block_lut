#!/usr/bin/env python3
"""
Convert bf16 model checkpoint → LUT-quantized safetensors.

Reads original bf16 safetensors + pre-computed LUT codebook,
quantizes expert weights, saves as LUT-format safetensors.

Usage:
    python3 convert_checkpoint.py --model /path/to/model --code_type BLOCKLUT
    # Then use with vLLM:
    python3 -m vllm_lut.run_lut_only --model /path/to/model_lut
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/hh/LUT-MoE/models/qwen")
    parser.add_argument("--output", default="")
    parser.add_argument("--code_type", default="BLOCKLUT",
                        choices=["BLOCKLUT", "NESTEDLUT"])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_codebooks(model_path, code_type="BLOCKLUT"):
    """Load LUT codebook files."""
    codebooks = {}
    for fname, key in [
        ("blocklut_256.npy", "blocklut"),
        ("nested_lut_mapped16.npy", "mapped16"),
        ("nested_lut_mapped64.npy", "mapped64"),
    ]:
        fp = os.path.join(model_path, fname)
        if os.path.exists(fp):
            arr = np.load(fp)
            codebooks[key] = torch.from_numpy(arr)
            print(f"  Loaded {fname} ({len(arr)} entries)")

    if code_type == "BLOCKLUT" and "blocklut" not in codebooks:
        raise FileNotFoundError(f"blocklut_256.npy not found in {model_path}")
    if code_type == "NESTEDLUT":
        for k in ["blocklut", "mapped16", "mapped64"]:
            if k not in codebooks:
                print(f"  Warning: {k} codebook not found, using blocklut")
                codebooks[k] = codebooks.get("blocklut")

    return codebooks


def quantize_weight(weight, codebook):
    """
    Quantize a bf16 weight tensor to BlockLUT format.

    Args:
        weight: bf16 tensor [rows, cols]
        codebook: bf16 codebook [256]

    Returns:
        (indices, absmax) where indices is uint8 and absmax is bf16
    """
    device = weight.device
    orig_shape = weight.shape
    flat = weight.reshape(-1)
    N = flat.shape[0]
    block_size = 128

    # Pad to block boundary
    num_blocks = (N + block_size - 1) // block_size
    padded = torch.zeros(num_blocks * block_size, dtype=flat.dtype, device=device)
    padded[:N] = flat

    # Block normalize
    blocks = padded.reshape(-1, block_size)
    absmax, _ = blocks.abs().max(dim=1, keepdim=True)
    absmax = absmax.clamp(min=1e-10)
    normalized = (blocks / absmax).reshape(-1)

    # Quantize: nearest neighbor in codebook
    codebook_f = codebook.float().to(device)
    data_f = normalized.float()
    # For efficiency, compute dot products instead of full distance
    # Use chunked nearest neighbor search to avoid OOM
    chunk_size = 1024 * 1024  # 1M elements per chunk
    all_indices = []
    for start in range(0, data_f.shape[0], chunk_size):
        end = min(start + chunk_size, data_f.shape[0])
        chunk = data_f[start:end]
        # [chunk_size, 256]
        dists = (chunk.unsqueeze(1) - codebook_f.unsqueeze(0)).abs()
        all_indices.append(dists.argmin(dim=1).to(torch.uint8))

    indices = torch.cat(all_indices)[:N].cpu()
    return indices, absmax.reshape(-1).cpu().to(torch.bfloat16)


def convert_checkpoint(args):
    model_path = args.model
    output_path = args.output or (model_path.rstrip("/") + "_lut_4bit")
    os.makedirs(output_path, exist_ok=True)

    print(f"Converting: {model_path} → {output_path}")
    print(f"Code type: {args.code_type}")

    # Load codebooks
    codebooks = load_codebooks(model_path, args.code_type)
    codebook = codebooks["blocklut"]
    if args.code_type == "NESTEDLUT":
        nested_codebook = codebooks.get("mapped16", codebook)
    else:
        nested_codebook = codebook

    # Find safetensors files
    import glob
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not safetensors_files:
        # Maybe single file
        single = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single):
            safetensors_files = [single]
    print(f"Found {len(safetensors_files)} safetensors files")

    # Load index file
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index_data = json.load(f)
    else:
        index_data = {"weight_map": {}}

    # Process each safetensors file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for ckpt_path in safetensors_files:
        print(f"\nProcessing: {os.path.basename(ckpt_path)}")
        state_dict = load_file(ckpt_path, device=device)
        print(f"  Loaded {len(state_dict)} tensors")

        new_dict = {}
        quant_stats = {"dense": 0, "expert_quantized": 0, "expert_size": 0}

        for name, tensor in state_dict.items():
            is_expert = "experts." in name and "shared_expert" not in name

            if is_expert:
                # Quantize expert weights to LUT format
                indices, absmax = quantize_weight(tensor, codebook)
                # Store quantized versions
                q_name = name.replace(".weight", ".q_weight")
                new_dict[name] = indices  # uint8 indices
                new_dict[name + ".absmax"] = absmax  # bf16 absmax
                quant_stats["expert_quantized"] += 1
                quant_stats["expert_size"] += indices.numel() + absmax.numel()
            else:
                # Dense weights: copy as-is
                new_dict[name] = tensor.cpu()
                quant_stats["dense"] += 1

            del tensor

        # Save codebook with the checkpoint
        codebook_key = "model._lut_codebook"
        new_dict[codebook_key] = codebook.cpu()

        # Save
        out_name = os.path.basename(ckpt_path).replace(".safetensors", "_lut.safetensors")
        out_path = os.path.join(output_path, out_name)
        save_file(new_dict, out_path)

        orig_size = os.path.getsize(ckpt_path)
        new_size = os.path.getsize(out_path)
        print(f"  Saved {out_name}: {orig_size/1e9:.1f}GB → {new_size/1e9:.1f}GB "
              f"({(1 - new_size/orig_size)*100:.0f}% saved)")

    # Write new index file
    new_index = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": {}
    }
    for old_name, old_file in index_data.get("weight_map", {}).items():
        new_index["weight_map"][old_name] = old_file.replace(
            ".safetensors", "_lut.safetensors"
        )
    # Add codebook reference
    new_index["weight_map"][codebook_key] = os.path.basename(
        safetensors_files[0]
    ).replace(".safetensors", "_lut.safetensors")

    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy config and codebook files
    for f in os.listdir(model_path):
        if f.endswith((".json", ".py", ".txt", ".npy")) and not f.startswith("model-"):
            shutil.copy2(os.path.join(model_path, f), os.path.join(output_path, f))

    print(f"\n=== Conversion complete! ===")
    print(f"Output: {output_path}")
    print(f"Quantized {quant_stats['expert_quantized']} expert weights")
    print(f"Dense weights copied: {quant_stats['dense']}")

    total_orig = sum(os.path.getsize(f) for f in safetensors_files)
    total_new = sum(os.path.getsize(
        os.path.join(output_path, os.path.basename(f).replace(
            ".safetensors", "_lut.safetensors"
        ))
    ) for f in safetensors_files)
    print(f"Total: {total_orig/1e9:.1f}GB → {total_new/1e9:.1f}GB "
          f"({(1 - total_new/total_orig)*100:.0f}% saved)")


if __name__ == "__main__":
    args = parse_args()
    convert_checkpoint(args)
