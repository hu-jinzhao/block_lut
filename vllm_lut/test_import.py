#!/usr/bin/env python3
"""
Quick test to verify LUT-MoE vLLM integration imports and basic functionality.
"""

import os
import sys

# Ensure CUDA runtime can be found
cuda_paths = [
    "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib",
]
for p in cuda_paths:
    if os.path.isdir(p) and p not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        print(f"[LUT-MoE] Added to LD_LIBRARY_PATH: {p}")

import torch

# Test 1: Imports
print("=== Test 1: Package Imports ===")
from vllm_lut.config import LUT_MoE_vLLMConfig
from vllm_lut.quantizer import LUTQuantizer, train_lut_codebook
from vllm_lut.progressive_cache import ProgressiveExpertCache
from vllm_lut.moe_layer import DeepseekV2MoE_LUT
from vllm_lut.patcher import patch_model, restore_model
from vllm_lut.engine import LUT_MoE_for_vLLM
print("[OK] All imports successful")

# Test 2: Quantizer
print("\n=== Test 2: LUT Quantizer ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

quantizer = LUTQuantizer(code_type="BLOCKLUT", device=device)

# Create test weights (simulate expert weights)
torch.manual_seed(42)
test_weights = torch.randn(4, 64, 128, dtype=torch.bfloat16, device=device)
print(f"Test weights shape: {test_weights.shape}")

# Train codebook
quantizer.train(test_weights)
print(f"Codebook trained: {quantizer.lut_codebook.shape}")
print(f"Codebook values: min={quantizer.lut_codebook.min().item():.4f}, "
      f"max={quantizer.lut_codebook.max().item():.4f}")

# Quantize
result_quant = quantizer.quantize(test_weights[0])
print(f"Quantized indices: {result_quant['indices'].shape}, "
      f"dtype={result_quant['indices'].dtype}")
print(f"Absmax blocks: {result_quant['absmax'].shape}")

# Decompress
decompressed = quantizer.decompress(
    result_quant["indices"].to(device),
    result_quant["absmax"].to(device),
    result_quant["codebook"].to(device),
    out_shape=test_weights[0].shape,
)
print(f"Decompressed shape: {decompressed.shape}")

# Calculate error
orig = test_weights[0].float()
decomp = decompressed.float()
mse = ((orig - decomp) ** 2).mean().item()
print(f"Decompression MSE: {mse:.6f}")
print(f"Compression ratio: "
      f"{result_quant['indices'].numel() * 1 + result_quant['absmax'].numel() * 2} / "
      f"{orig.numel() * 2} = "
      f"{(result_quant['indices'].numel() * 1 + result_quant['absmax'].numel() * 2) / (orig.numel() * 2):.3f}")

# Test 3: Progressive Cache
print("\n=== Test 3: Progressive Expert Cache ===")
cache = ProgressiveExpertCache(num_layers=3, num_experts=4, max_gpu_entries=4)
for layer in range(3):
    for expert in range(4):
        for _ in range(expert * 10 + layer * 5 + 1):
            tier = cache.record_access(layer, expert)
print(f"Cache stats: {cache.get_stats()}")

# Test 4: Config
print("\n=== Test 4: Configuration ===")
config = LUT_MoE_vLLMConfig(code_type="BLOCKLUT")
print(f"Config: code_type={config.code_type}, "
      f"enable_progressive_loading={config.enable_progressive_loading}")

config_nested = LUT_MoE_vLLMConfig(code_type="NESTEDLUT", lut_tier=0)
print(f"NestedLUT config: code_type={config_nested.code_type}, "
      f"lut_tier={config_nested.lut_tier}")

print("\n=== All tests passed! ===")
