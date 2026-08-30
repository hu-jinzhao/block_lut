# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
Monkey-patching system for integrating LUT-MoE with vLLM models.

Patches vLLM's model loading and MoE layer construction to use
LUT-quantized expert weights with progressive loading.

Strategy:
1. Replace the model's MoE layer class (e.g., DeepseekV2MoE) with
   the LUT-quantized version (DeepseekV2MoE_LUT).
2. Hook into weight loading to quantize expert weights on-the-fly.
3. Replace the forward pass to use on-the-fly decompression.
"""

import copy
import importlib
import os
from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from .config import LUT_MoE_vLLMConfig
from .moe_layer import DeepseekV2MoE_LUT
from .quantizer import LUTQuantizer


# Map of supported model architectures to their vLLM module paths
# and the MoE class name to replace
MODEL_MOE_MAP = {
    "deepseek_v2": {
        "module_path": "vllm.model_executor.models.deepseek_v2",
        "class_name": "DeepseekV2MoE",
    },
    "deepseek_v3": {
        "module_path": "vllm.model_executor.models.deepseek_v2",
        "class_name": "DeepseekV2MoE",
    },
}


def patch_deepseek_v2(
    lut_config: LUT_MoE_vLLMConfig,
) -> None:
    """
    Monkey-patch vLLM's DeepseekV2 model to use LUT-quantized MoE.

    Replaces the DeepseekV2MoE class in vLLM's model module with
    DeepseekV2MoE_LUT, and hooks into the constructor to pass the
    LUT configuration.
    """
    module_path = "vllm.model_executor.models.deepseek_v2"

    # Import the module
    mod = importlib.import_module(module_path)

    # Save original for restore
    if not hasattr(mod, "_original_DeepseekV2MoE"):
        mod._original_DeepseekV2MoE = mod.DeepseekV2MoE

    # Patch: replace DeepseekV2MoE with LUT version
    original_moe = mod.DeepseekV2MoE

    class PatchedDeepseekV2MoE(original_moe):
        """
        Wrapper that replaces expert weights with LUT-quantized version
        after the original initialization.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Store LUT config
            self.lut_config = lut_config
            self._lut_quantized = False
            self._layer_id = 0

        def quantize_experts(self):
            """Replace bf16 expert weights with LUT-quantized version."""
            if self._lut_quantized:
                return

            try:
                # Access the experts from FusedMoE
                experts = self.experts.modular_experts
            except AttributeError:
                try:
                    experts = self.experts.routed_experts
                except AttributeError:
                    print("[LUT-MoE] Warning: Could not access routed_experts, "
                          "skipping quantization")
                    return

            if not hasattr(experts, "w13_weight") or not hasattr(experts, "w2_weight"):
                print("[LUT-MoE] Warning: experts object has no w13_weight/w2_weight, "
                      "skipping")
                return

            w13 = experts.w13_weight.data  # [num_experts, 2*intermediate, hidden]
            w2 = experts.w2_weight.data    # [num_experts, hidden, intermediate]

            # Train/load LUT codebook
            quantizer = LUTQuantizer(code_type=lut_config.code_type)

            if lut_config.lut_path and os.path.isdir(lut_config.lut_path):
                loaded = quantizer.load(lut_config.lut_path)
                if not loaded:
                    print(f"[LUT-MoE] Training codebook on {w13.shape[0]} experts...")
                    all_w = torch.cat([w13.reshape(-1), w2.reshape(-1)])
                    quantizer.train(all_w)
                    quantizer.save(lut_config.lut_path)
            else:
                print(f"[LUT-MoE] Training codebook on {w13.shape[0]} experts...")
                all_w = torch.cat([w13.reshape(-1), w2.reshape(-1)])
                quantizer.train(all_w)
                if lut_config.lut_path:
                    quantizer.save(lut_config.lut_path)

            codebook = quantizer.lut_codebook.to(w13.device)
            n_experts = w13.shape[0]

            # Quantize each expert
            q_indices = []
            q_absmax = []
            for i in range(n_experts):
                q = quantizer.quantize(w13[i], codebook)
                q_indices.append(q["indices"].to(w13.device))
                q_absmax.append(q["absmax"].to(w13.device))
            self._w13_indices = torch.stack(q_indices, dim=0).to(w13.device)
            self._w13_absmax = torch.stack(q_absmax, dim=0).to(w13.device)

            q_indices = []
            q_absmax = []
            for i in range(n_experts):
                q = quantizer.quantize(w2[i], codebook)
                q_indices.append(q["indices"].to(w2.device))
                q_absmax.append(q["absmax"].to(w2.device))
            self._w2_indices = torch.stack(q_indices, dim=0).to(w2.device)
            self._w2_absmax = torch.stack(q_absmax, dim=0).to(w2.device)

            self._codebook = codebook
            self._lut_quantized = True

            # Keep a decompression cache (on GPU)
            self._decompressed_cache: Dict[int, Dict] = {}

            print(f"[LUT-MoE] Quantized {n_experts} experts to {lut_config.code_type} "
                  f"(codebook: 256 entries)")

        def get_bf16_weights(self, expert_id: int):
            """
            Get decompressed bf16 weights for an expert.
            Uses cache if available, otherwise decompresses on-the-fly.
            """
            if expert_id in self._decompressed_cache:
                return self._decompressed_cache[expert_id]

            if not self._lut_quantized:
                return None

            # Decompress
            from .quantizer import decompress_blocklut
            block_size = 128
            n_intermediate = self.intermediate_size
            hidden = self.hidden_size

            w13 = decompress_blocklut(
                self._w13_indices[expert_id],
                self._w13_absmax[expert_id],
                self._codebook,
                block_size,
            ).reshape(2 * n_intermediate, hidden)

            w2 = decompress_blocklut(
                self._w2_indices[expert_id],
                self._w2_absmax[expert_id],
                self._codebook,
                block_size,
            ).reshape(hidden, n_intermediate)

            result = {"w13": w13, "w2": w2}
            self._decompressed_cache[expert_id] = result
            return result

        def forward(self, hidden_states, **kwargs):
            """Forward pass: quantize first if needed, then use decompressed weights."""
            if not self._lut_quantized:
                self.quantize_experts()

            # If we have quantized weights, replace the FusedMoE weights temporarily
            if self._lut_quantized:
                try:
                    experts = self.experts.routed_experts
                    w13_p = experts.w13_weight
                    w2_p = experts.w2_weight
                    saved_w13 = w13_p.data.clone()
                    saved_w2 = w2_p.data.clone()
                except Exception:
                    return super().forward(hidden_states, **kwargs)

                # Decompress all experts and repack for the fused kernel
                w13_bf16 = torch.stack([
                    self.get_bf16_weights(i)["w13"] for i in range(w13_p.shape[0])
                ], dim=0)
                w2_bf16 = torch.stack([
                    self.get_bf16_weights(i)["w2"] for i in range(w2_p.shape[0])
                ], dim=0)

                # Temporarily replace weights
                w13_p.data = w13_bf16
                w2_p.data = w2_bf16

                try:
                    result = super().forward(hidden_states, **kwargs)
                finally:
                    # Restore original quantized weights
                    w13_p.data = saved_w13
                    w2_p.data = saved_w2

                return result

            return super().forward(hidden_states, **kwargs)

    # Replace the class in the module
    mod.DeepseekV2MoE = PatchedDeepseekV2MoE
    print(f"[LUT-MoE] Patched {module_path}.DeepseekV2MoE")


def patch_model(lut_config: LUT_MoE_vLLMConfig) -> None:
    """
    Patch vLLM model classes to use LUT-MoE quantization.

    Automatically detects the model architecture and applies
    the appropriate patches.

    Args:
        lut_config: LUT-MoE configuration
    """
    # Currently supporting DeepSeekV2/V3
    patch_deepseek_v2(lut_config)

    # Future: Qwen, Mixtral, etc.
    # patch_qwen_moe(lut_config)
    # patch_mixtral_moe(lut_config)

    print("[LUT-MoE] Model patching complete")


def restore_model() -> None:
    """Restore all patched vLLM model classes to their originals."""
    module_path = "vllm.model_executor.models.deepseek_v2"
    mod = importlib.import_module(module_path)
    if hasattr(mod, "_original_DeepseekV2MoE"):
        mod.DeepseekV2MoE = mod._original_DeepseekV2MoE
        del mod._original_DeepseekV2MoE
        print(f"[LUT-MoE] Restored {module_path}.DeepseekV2MoE")
