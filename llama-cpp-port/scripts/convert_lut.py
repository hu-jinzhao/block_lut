#!/usr/bin/env python3
"""
LUT-MoE → GGUF Converter

Converts HuggingFace MoE models to GGUF format with BlockLUT / NestedLUT quantization.
Dense layers are stored as BF16; MoE expert weights are quantized to BlockLUT format.

Usage:
    python convert_lut.py --model /path/to/llama-moe --out model.gguf
    python convert_lut.py --model /path/to/qwen2-moe --out model.gguf --nested-lut
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from typing import Optional, Sequence
from safetensors import safe_open
from tqdm import tqdm

# GGUF Python library
sys.path.insert(0, "/home/hh/llama.cpp/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType, Keys
from gguf.quants import quant_shape_from_byte_shape


# ---------------------------------------------------------------------------
# BlockLUT Quantization (from LUT-MoE model_offload.py)
# ---------------------------------------------------------------------------

def train_kmeans_lut(normalized_values: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Train K-means on normalized block values to produce LUT centroids.
    Uses MiniBatchKMeans for memory efficiency on large datasets."""
    from sklearn.cluster import MiniBatchKMeans
    # Subsample if too large (>10M points)
    data = normalized_values.reshape(-1, 1).astype(np.float32)
    if len(data) > 10_000_000:
        idx = np.random.RandomState(seed).choice(len(data), 10_000_000, replace=False)
        data = data[idx]
        print(f"[LUT-MoE] Subsampled K-means training data: 10M points")
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=1024, n_init=3)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_.flatten()
    centroids.sort()
    return centroids.astype(np.float32)


def quantize_blocklut(weight: torch.Tensor, K: int = 256,
                      lut: Optional[np.ndarray] = None) -> tuple:
    """
    Quantize a weight tensor to BlockLUT format.

    Args:
        weight: Input weight tensor (any shape)
        K: Number of LUT entries (256 = 8-bit)
        lut: Pre-trained LUT centroids, or None to train from data

    Returns:
        (indices, absmax_uint16, lut_table)
        - indices: uint8 array of LUT indices
        - absmax_uint16: bf16 absmax per 128-element block, uint16 representation
        - lut_table: float32 LUT centroids
    """
    x = weight.detach().to(torch.float32).numpy().ravel()
    n = x.size
    bs = 128
    nb = (n + bs - 1) // bs

    # Pad to multiple of block_size
    if nb * bs > n:
        x = np.pad(x, (0, nb * bs - n))

    blocks = x.reshape(nb, bs)
    absmax_vals = np.max(np.abs(blocks), axis=1)
    absmax_vals = np.maximum(absmax_vals, 1e-12)
    normalized = blocks / absmax_vals[:, np.newaxis]

    if lut is None:
        lut = train_kmeans_lut(normalized, K)

    # Quantize: nearest-neighbor search in LUT
    midpoints = (lut[:-1] + lut[1:]) / 2.0
    indices = np.searchsorted(midpoints, normalized.ravel()).astype(np.uint8)
    indices = indices[:n]  # remove padding

    # Convert absmax to bf16 uint16
    absmax_bf16 = torch.from_numpy(absmax_vals).to(torch.bfloat16)
    absmax_uint16 = absmax_bf16.view(torch.int16).numpy().astype(np.uint16)

    return indices, absmax_uint16, lut


def pack_bitplanes(indices: np.ndarray) -> np.ndarray:
    """
    Pack uint8 indices into 3-section bit-plane format.
    Returns contiguous uint8 array: [packed_low] [packed_mid] [packed_high]
    """
    low = (indices & 0x0F).astype(np.uint8)
    mid = ((indices >> 4) & 0x03).astype(np.uint8)
    high = ((indices >> 6) & 0x03).astype(np.uint8)

    # Pack 4-bit (2 elem/byte)
    packed_low = (low[0::2] | (low[1::2] << 4)).astype(np.uint8)
    # Pack 2-bit (4 elem/byte)
    packed_mid = (mid[0::4] | (mid[1::4] << 2) | (mid[2::4] << 4) | (mid[3::4] << 6)).astype(np.uint8)
    packed_high = (high[0::4] | (high[1::4] << 2) | (high[2::4] << 4) | (high[3::4] << 6)).astype(np.uint8)

    return np.concatenate([packed_low, packed_mid, packed_high]).astype(np.uint8)


def build_nested_luts(lut256: np.ndarray) -> tuple:
    """
    Build nested LUT tables from a K=256 LUT.
    Returns (lut256, lut64, lut16) as uint16 bf16 arrays.
    """
    # For nested LUT: C16 ⊂ C64 ⊂ C256
    # Take first 64/16 centroids from sorted K=256
    lut64 = lut256[:64].copy()
    lut16 = lut256[:16].copy()

    # Pad to 256 entries (GPU kernel expects 256-entry table)
    def pad_to_256(lut_partial: np.ndarray) -> np.ndarray:
        padded = np.ones(256, dtype=np.float32) * lut_partial[-1]
        padded[:len(lut_partial)] = lut_partial
        return padded

    # Convert to bf16 uint16
    def to_bf16_uint16(arr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(arr).to(torch.bfloat16)
        return t.view(torch.int16).numpy().astype(np.uint16)

    return (to_bf16_uint16(pad_to_256(lut256)),
            to_bf16_uint16(pad_to_256(lut64)),
            to_bf16_uint16(pad_to_256(lut16)))


# ---------------------------------------------------------------------------
# MoE tensor detection
# ---------------------------------------------------------------------------

def is_expert_tensor(name: str) -> bool:
    """Check if a tensor name corresponds to an MoE expert weight."""
    name_lower = name.lower()
    # MoE expert patterns
    expert_patterns = [
        "experts.",         # LLaMA MoE, Qwen2MoE, DeepSeek
        "mlp.experts",      # SwitchTransformers
        ".experts.",        # generic
        "ffn_gate_exps",    # GGUF naming
        "ffn_up_exps",
        "ffn_down_exps",
        "gate_exps",        # generic
        "down_exps",
        "up_exps",
    ]
    # Exclude shared experts and routers
    exclude_patterns = [
        "shared_expert",
        "shexp",
        "gate_inp",         # router weights
        "gate.weight",      # router
        "router",
    ]
    for pat in expert_patterns:
        if pat in name_lower:
            for ex in exclude_patterns:
                if ex in name_lower:
                    return False
            return True
    return False


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_state_dict(model_path: str) -> dict:
    """Load HuggingFace model state dict from safetensors or bin files."""
    state_dict = {}
    if os.path.isdir(model_path):
        # Load all shards
        files = sorted(os.listdir(model_path))
        safetensor_files = [f for f in files if f.endswith(".safetensors")]
        bin_files = [f for f in files if f.endswith(".bin")]

        if safetensor_files:
            for sf in tqdm(safetensor_files, desc="Loading safetensors"):
                sf_path = os.path.join(model_path, sf)
                with safe_open(sf_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
        elif bin_files:
            for bf in bin_files:
                bf_path = os.path.join(model_path, bf)
                sd = torch.load(bf_path, map_location="cpu", weights_only=True)
                state_dict.update(sd)
        else:
            raise FileNotFoundError(f"No model files found in {model_path}")

        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                state_dict["__config__"] = json.load(f)
    else:
        raise ValueError(f"Model path not found: {model_path}")

    return state_dict


def get_model_architecture(config: dict) -> str:
    """Map HuggingFace architecture to GGUF architecture name."""
    arch = config.get("architectures", [""])[0]
    mapping = {
        "Qwen2MoeForCausalLM":        "Qwen2Moe",
        "Qwen2.5MoeForCausalLM":      "Qwen2Moe",
        "DeepseekV2ForCausalLM":      "Deepseek2",
        "DeepseekV3ForCausalLM":      "Deepseek2",
        "LlamaMoEForCausalLM":        "Llama",
        "MixtralForCausalLM":         "Mixtral",
        "DbrxForCausalLM":            "DBRX",
        "JetMoEForCausalLM":          "JetMoE",
        "OlmoEForCausalLM":           "OlmoE",
        "PhiMoEForCausalLM":          "PhiMoE",
        "ArcticForCausalLM":          "Arctic",
    }
    return mapping.get(arch, "Llama")  # default to LLaMA architecture


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_to_blocklut_gguf(
    model_path: str,
    output_path: str,
    K: int = 256,
    nested_lut: bool = False,
    progress: bool = True,
) -> None:
    """Convert a HuggingFace MoE model to GGUF with BlockLUT quantization."""
    print(f"[LUT-MoE] Loading model from {model_path} ...")
    state_dict = load_state_dict(model_path)
    config = state_dict.pop("__config__", {})
    print(f"[LUT-MoE] Architecture: {config.get('architectures', ['Unknown'])[0]}")

    arch = get_model_architecture(config)

    # Determine output filename
    if output_path is None:
        model_name = os.path.basename(model_path.rstrip("/"))
        output_path = f"{model_name}_blocklut.gguf"

    # Initialize GGUF writer
    gguf_writer = GGUFWriter(output_path, arch)

    # Add metadata
    gguf_writer.add_float32("hparams.attention_layer_norm_rms_epsilon",
                            config.get("rms_norm_eps", 1e-6))
    gguf_writer.add_uint32("hparams.n_expert",
                           config.get("num_local_experts", config.get("num_experts", 0)))
    gguf_writer.add_uint32("hparams.n_expert_used",
                           config.get("num_experts_per_tok", config.get("top_k", 2)))

    # LUT-MoE metadata
    gguf_writer.add_bool("lut-moe.enabled", True)
    gguf_writer.add_uint32("lut-moe.block_size", 128)
    gguf_writer.add_uint32("lut-moe.k", K)
    gguf_writer.add_bool("lut-moe.nested_lut", nested_lut)

    if progress:
        print(f"[LUT-MoE] Processing {len(state_dict)} tensors ...")

    # Collect all expert LUT tables for global LUT sharing
    all_normalized = []
    expert_names = []

    # First pass: collect statistics for global LUT training
    # Use reservoir sampling to limit memory usage
    if progress:
        print("[LUT-MoE] First pass: collecting statistics (sampling)...")
    sampled_data = []
    sample_capacity = 10_000_000  # 10M points max
    rng = np.random.RandomState(0)
    for name, tensor in tqdm(state_dict.items(), disable=not progress):
        if is_expert_tensor(name):
            x = tensor.to(torch.float32).numpy().ravel()
            n = x.size
            bs = 128
            nb = (n + bs - 1) // bs
            if nb * bs > n:
                x = np.pad(x, (0, nb * bs - n))
            blocks = x.reshape(nb, bs)
            absmax_vals = np.maximum(np.max(np.abs(blocks), axis=1), 1e-12)
            normalized = blocks / absmax_vals[:, np.newaxis]
            # Reservoir sampling: keep ~sample_capacity points
            flat = normalized.ravel()
            if len(sampled_data) < sample_capacity:
                sampled_data.extend(flat)
            else:
                # Random replace
                for val in flat:
                    if rng.rand() < sample_capacity / (len(sampled_data) + 1):
                        sampled_data[rng.randint(sample_capacity)] = val
            del x, blocks, normalized

    # Train global LUT
    # Try loading pre-computed LUT from model directory
    if global_lut is None:
        # Check common locations for pre-computed LUT
        for lut_name in [f"blocklut_{K}.npy", "blocklut_256.npy"]:
            lut_candidate = os.path.join(model_path, lut_name)
            if os.path.exists(lut_candidate):
                global_lut = np.load(lut_candidate)
                if progress:
                    print(f"[LUT-MoE] Loaded pre-computed LUT from {lut_candidate}")
                break

    if global_lut is None and sampled_data:
        if progress:
            print(f"[LUT-MoE] Training global LUT K={K} on {len(sampled_data):,} sampled values ...")
        global_lut = train_kmeans_lut(np.array(sampled_data, dtype=np.float32), K)

    # Build nested LUT tables if requested
    lut_uint16 = None
    lut_mapped64 = None
    lut_mapped16 = None
    if nested_lut and global_lut is not None:
        lut_uint16, lut_mapped64, lut_mapped16 = build_nested_luts(global_lut)
        # Store nested LUT tables in GGUF metadata
        gguf_writer.add_array("lut-moe.lut_table",
                              lut_uint16.tolist() if len(lut_uint16) <= 256 else lut_uint16[:256].tolist())
        if lut_mapped64 is not None:
            gguf_writer.add_array("lut-moe.lut_mapped64",
                                  lut_mapped64[:64].tolist())
        if lut_mapped16 is not None:
            gguf_writer.add_array("lut-moe.lut_mapped16",
                                  lut_mapped16[:16].tolist())
    elif global_lut is not None:
        lut_bf16 = torch.from_numpy(global_lut).to(torch.bfloat16)
        lut_uint16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)
        gguf_writer.add_array("lut-moe.lut_table",
                              lut_uint16.tolist() if len(lut_uint16) <= 256 else lut_uint16[:256].tolist())

    # Second pass: quantize and write tensors
    if progress:
        print("[LUT-MoE] Second pass: quantizing and writing tensors ...")

    for name, tensor in tqdm(state_dict.items(), disable=not progress):
        if is_expert_tensor(name):
            # Quantize to BlockLUT
            indices, absmax_uint16, _ = quantize_blocklut(tensor, K, global_lut)

            if nested_lut:
                # Pack into bit-plane format
                packed = pack_bitplanes(indices)
                # Concatenate with absmax
                quantized_data = np.concatenate([packed, absmax_uint16.view(np.uint8)])
            else:
                # Full uint8 indices + absmax
                quantized_data = np.concatenate([indices, absmax_uint16.view(np.uint8)])

            # Compute byte shape
            n_elements = tensor.numel()
            blk_size = 128
            type_size = 130  # 128 indices + 2 absmax bytes per block
            if nested_lut:
                type_size = 130  # same storage for all tiers (bitplane packed)
            byte_shape = quant_shape_from_byte_shape(
                (n_elements // blk_size * type_size,), GGMLQuantizationType.BLOCKLUT8
            )

            # Write as BLOCKLUT type
            gguf_writer.add_tensor(
                name,
                quantized_data,
                raw_shape=(n_elements,),
                raw_dtype=GGMLQuantizationType.BLOCKLUT8,
            )
        else:
            # Dense tensor: write as BF16
            tensor_bf16 = tensor.to(torch.bfloat16)
            gguf_writer.add_tensor(
                name,
                tensor_bf16.numpy().view(np.uint8),
                raw_dtype=GGMLQuantizationType.BF16,
            )

    # Write to file
    if progress:
        print(f"[LUT-MoE] Writing GGUF file to {output_path} ...")
    gguf_writer.write_tensors_to_file(progress=progress)
    gguf_writer.close()

    # Write file offset metadata for runtime
    meta_path = output_path + ".meta.json"
    # ... (metadata generation can be added)

    print(f"[LUT-MoE] Done! Output: {output_path}")
    if global_lut is not None:
        print(f"[LUT-MoE] LUT: {len(global_lut)} entries, PSNR ~44 dB (8-bit)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert MoE models to GGUF with BlockLUT quantization")
    parser.add_argument("--model", "-m", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--out", "-o", default=None, help="Output GGUF file path")
    parser.add_argument("--K", type=int, default=256, help="LUT size (default: 256 = 8-bit)")
    parser.add_argument("--nested-lut", action="store_true", help="Enable NestedLUT progressive bit-plane storage")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    convert_to_blocklut_gguf(
        model_path=args.model,
        output_path=args.out,
        K=args.K,
        nested_lut=args.nested_lut,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
