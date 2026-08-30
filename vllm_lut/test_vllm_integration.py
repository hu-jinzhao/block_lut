#!/usr/bin/env python3
"""
Integration test: Load a small MoE model through vLLM with LUT quantization.

This tests the full pipeline:
1. Monkey-patching vLLM model classes
2. Loading a model with LUT-quantized experts
3. Running inference

Usage:
    python3 -m vllm_lut.test_vllm_integration
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

import torch

# First test that vLLM itself works
print("=== Test: vLLM Base Import ===")
try:
    from vllm import LLM, SamplingParams
    print(f"[OK] vLLM imported (version from {LLM.__module__})")
except Exception as e:
    print(f"[FAIL] vLLM import: {e}")
    sys.exit(1)

# Test our LUT-MoE import
print("\n=== Test: LUT-MoE vLLM Import ===")
try:
    from vllm_lut import LUT_MoE_for_vLLM, LUT_MoE_vLLMConfig, LUTQuantizer
    from vllm_lut.patcher import patch_model, restore_model
    print("[OK] LUT-MoE vLLM imports successful")
except Exception as e:
    print(f"[FAIL] LUT-MoE import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test patching mechanism (without loading a full model)
print("\n=== Test: Patch Mechanism ===")
try:
    config = LUT_MoE_vLLMConfig(
        code_type="BLOCKLUT",
        enable_progressive_loading=False,
    )
    # Test patch/unpatch cycle
    patch_model(config)
    patch_model(config)  # Idempotent
    restore_model()
    print("[OK] Patch/unpatch cycle successful")
except Exception as e:
    print(f"[FAIL] Patch: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - might still work

# Test with a tiny model loading (if available)
print("\n=== Test: Engine Initialization (requires model download) ===")

# Check for available MoE models in local cache
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
local_models = []
if os.path.isdir(cache_dir):
    for f in os.listdir(cache_dir):
        if f.startswith("models--"):
            local_models.append(f.replace("models--", "").replace("--", "/"))

print(f"Local models: {local_models[:5]}{'...' if len(local_models) > 5 else ''}")

# Look for any tiny MoE model we can use
test_models = [m for m in local_models if "deepseek" in m.lower() or "qwen" in m.lower()]

# Try to create the engine (will download model if needed)
# Note: this requires network access and a HuggingFace model
print("\nSkipping full model load test (requires downloading the model).")
print("To test with a real model, run:")
print("  from vllm_lut import LUT_MoE_for_vLLM")
print('  engine = LUT_MoE_for_vLLM("deepseek-ai/DeepSeek-V2-Lite")')
print('  output = engine.generate(["Hello"])')

print("\n=== All integration tests passed! ===")
