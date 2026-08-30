#!/usr/bin/env python3
"""Fast BlockLUT GGUF converter — streams tensor shards, uses precomputed LUT."""
import argparse, gc, json, os, sys, numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, "/home/hh/llama.cpp/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType

def is_expert(name):
    return any(p in name for p in ["experts.", ".experts.", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "gate_up_exps"]) \
        and not any(p in name for p in ["shared_expert", "shexp", "gate_inp", "router"])

def quantize_block(weight: np.ndarray, lut: np.ndarray):
    """BlockLUT quantize: 128-block, absmax normalize, LUT index search."""
    x = weight.ravel().astype(np.float32)
    n = x.size
    bs = 128
    nb = (n + bs - 1) // bs
    if nb * bs > n:
        x = np.pad(x, (0, nb * bs - n))
    blocks = x.reshape(nb, bs)
    absmax = np.maximum(np.max(np.abs(blocks), axis=1), 1e-12)
    normed = (blocks / absmax[:, np.newaxis]).ravel()
    midpoints = (lut[:-1] + lut[1:]) / 2.0
    indices = np.searchsorted(midpoints, normed).astype(np.uint8)[:n]
    absmax_bf16 = torch.from_numpy(absmax).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
    return indices, absmax_bf16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="/tmp/model_blocklut.gguf")
    parser.add_argument("--lut", default="")
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.model, "config.json")) as f:
        config = json.load(f)
    arch = config.get("architectures", ["Llama"])[0]
    print(f"[LUT-MoE] Arch: {arch}")

    # Load pre-computed LUT
    lut_path = args.lut or os.path.join(args.model, "blocklut_256.npy")
    lut = np.load(lut_path)
    print(f"[LUT-MoE] Loaded LUT: {len(lut)} entries, min={lut.min():.4f}, max={lut.max():.4f}")

    # Init GGUF writer
    w = GGUFWriter(args.out, arch)
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)):
            try: w.add_key_value(k, v)
            except: pass

    # Find shard files
    files = sorted(f for f in os.listdir(args.model) if f.endswith(".safetensors"))
    if not files:
        files = sorted(f for f in os.listdir(args.model) if f.endswith(".bin"))
    print(f"[LUT-MoE] Found {len(files)} shard files")

    # Process each shard
    expert_count = 0
    for sf in tqdm(files, desc="Processing shards"):
        with safe_open(os.path.join(args.model, sf), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if is_expert(name):
                    # BlockLUT quantize
                    indices, absmax = quantize_block(tensor.numpy(), lut)
                    packed = np.concatenate([indices, absmax.view(np.uint8)])
                    w.add_tensor(name, packed, raw_shape=(tensor.numel(),), raw_dtype=GGMLQuantizationType.BLOCKLUT8)
                    expert_count += 1
                else:
                    # BF16
                    t = tensor.to(torch.bfloat16).numpy().view(np.uint8)
                    w.add_tensor(name, t, raw_dtype=GGMLQuantizationType.BF16)
                del tensor
            gc.collect()

    # Write LUT table as metadata
    lut_bf16 = torch.from_numpy(lut).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
    w.add_array("lut-moe.lut_table", lut_bf16.tolist())
    w.add_bool("lut-moe.enabled", True)
    w.add_uint32("lut-moe.block_size", 128)
    w.add_uint32("lut-moe.k", 256)

    print(f"[LUT-MoE] Writing GGUF with {expert_count} BlockLUT experts ...")
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"[LUT-MoE] Done: {args.out}")

if __name__ == "__main__":
    main()
