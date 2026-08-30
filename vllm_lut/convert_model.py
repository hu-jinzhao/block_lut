#!/usr/bin/env python3
"""
Convert bf16 MoE model → LUT-quantized safetensors using fast CUDA encoder.

Pre-computed codebook (blocklut_256.npy) must exist in the model directory.
Output is ~75% smaller than original and can be loaded by vLLM with LUT method.

Usage:
    python3 convert_model.py --model /home/hh/LUT-MoE/models/qwen  --output /home/hh/LUT-MoE/models/qwen_lut
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/hh/LUT-MoE/models/qwen")
    parser.add_argument("--output", default="")
    parser.add_argument("--code_type", default="NESTEDLUT", choices=["BLOCKLUT", "NESTEDLUT"])
    return parser.parse_args()


def load_cuda_encoder():
    """Import the CUDA encoder module (compiles once)."""
    from vllm_lut.fast_encode import encode_lut
    return encode_lut


def convert_checkpoint(args):
    model_path = args.model
    output_path = args.output or (model_path.rstrip("/") + "_lut")
    os.makedirs(output_path, exist_ok=True)

    # Load codebook (NestedLUT 4-bit = mapped16, BlockLUT = blocklut_256)
    if args.code_type == "BLOCKLUT":
        cb_name = "blocklut_256.npy"
    else:  # NESTEDLUT
        cb_name = "nested_lut_mapped16.npy"
    cb_path = os.path.join(model_path, cb_name)
    if not os.path.exists(cb_path):
        # Fallback to blocklut
        cb_path = os.path.join(model_path, "blocklut_256.npy")
    if not os.path.exists(cb_path):
        raise FileNotFoundError(f"Codebook not found in {model_path}")
    codebook_np = np.load(cb_path)
    codebook = torch.from_numpy(codebook_np).to(torch.bfloat16)
    codebook_f32 = codebook.float().cuda()
    print(f"Codebook ({os.path.basename(cb_path)}): {codebook.shape} ({codebook_np.dtype})")
    print(f"Codebook: {codebook.shape} ({codebook_np.dtype})")

    # Load CUDA encoder
    print("Compiling CUDA encoder (one-time)...")
    encode_lut = load_cuda_encoder()
    block_size = 128

    # Find safetensors files
    import glob
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not safetensors_files:
        sf = os.path.join(model_path, "model.safetensors")
        if os.path.exists(sf):
            safetensors_files = [sf]

    print(f"Found {len(safetensors_files)} checkpoint files")

    total_orig = 0
    total_new = 0
    quant_stats = {"experts": 0, "dense": 0, "encode_time": 0}

    for ckpt_path in safetensors_files:
        fname = os.path.basename(ckpt_path)
        print(f"\n=== {fname} ===")
        # Load tensor keys without moving to GPU yet
        from safetensors import safe_open
        new_state = {}
        t0 = time.time()

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print(f"  Found {len(keys)} tensors")

            for name in keys:
                is_expert = "experts." in name and "shared_expert" not in name
                is_weight = name.endswith(".weight")
                tensor = f.get_tensor(name)  # Load to CPU

                if is_expert and is_weight and tensor.dtype == torch.bfloat16:
                    # Encode to LUT format (move single tensor to GPU)
                    orig_shape = tensor.shape
                    flat = tensor.cuda().reshape(-1)
                    indices_cpu, absmax_cpu = encode_lut(flat, codebook_f32)
                    del flat
                    # Store under ORIGINAL name (.weight) keeping ORIGINAL shape
                    new_state[name] = indices_cpu.reshape(orig_shape)  # uint8, 2D
                    # absmax stored separately in npy file (not in safetensors)
                    quant_stats["experts"] += 1
                elif is_expert and is_weight:
                    new_state[name] = tensor
                else:
                    new_state[name] = tensor
                    quant_stats["dense"] += 1
            is_weight = name.endswith(".weight")

            if is_expert and is_weight and tensor.dtype == torch.bfloat16:
                # Encode to LUT format (on GPU)
                flat = tensor.reshape(-1).contiguous().cuda()
                del tensor  # Free original
                indices_cpu, absmax_cpu = encode_lut(flat, codebook_f32, block_size)
                del flat
                # Store: uint8 indices + bf16 absmax
                new_state[name.replace(".weight", ".q_weight")] = indices_cpu
                new_state[name.replace(".weight", ".q_absmax")] = absmax_cpu
                quant_stats["experts"] += 1
            elif is_expert and is_weight:
                new_state[name] = tensor.cpu()
            else:
                # Dense or non-weight tensor → copy as-is
                new_state[name] = tensor.cpu()
                quant_stats["dense"] += 1

        quant_stats["encode_time"] += time.time() - t0

        # Save codebook with model
        # codebook saved separately as .npy

        # Save to output
        out_name = fname.replace(".safetensors", "_lut.safetensors")
        out_path = os.path.join(output_path, out_name)
        save_file(new_state, out_path)

        orig_size = os.path.getsize(ckpt_path)
        new_size = os.path.getsize(out_path)
        total_orig += orig_size
        total_new += new_size
        print(f"  Saved: {out_name} ({orig_size/1e9:.1f}GB → {new_size/1e9:.1f}GB, "
              f"{new_size/orig_size*100:.0f}%)")

        del new_state

    # Copy config files
    for f in os.listdir(model_path):
        if any(f.endswith(ext) for ext in [".json", ".py", ".txt", ".model"]) and \
           not f.startswith("model-") and f != "model.safetensors":
            shutil.copy2(os.path.join(model_path, f), os.path.join(output_path, f))

    # Create index file
    index_src = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_src):
        with open(index_src) as f:
            index = json.load(f)
        new_index = {"metadata": index.get("metadata", {}), "weight_map": {}}
        for k, v in index.get("weight_map", {}).items():
            new_index["weight_map"][k] = v.replace(".safetensors", "_lut.safetensors")
        pass  # codebook not in safetensors
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Experts quantized: {quant_stats['experts']}")
    print(f"  Dense tensors copied: {quant_stats['dense']}")
    print(f"  Encode time: {quant_stats['encode_time']:.1f}s")
    print(f"  Size: {total_orig/1e9:.1f}GB → {total_new/1e9:.1f}GB "
          f"({(1-total_new/total_orig)*100:.0f}% reduction)")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")
    print(f"\nTo run with vLLM:")
    print(f'  llm = LLM(model="{output_path}", quantization="lut")')


if __name__ == "__main__":
    args = parse_args()
    convert_checkpoint(args)
