# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.
"""
vLLM-LUT: LUT Clustering Quantization for vLLM.

Supports:
  - BlockLUT (8-bit, 50% memory savings)
  - NestedLUT (4/6/8-bit progressive, up to 75% memory savings)
  - Integration via vLLM quantization plugin system
"""

from .engine import LUT_MoE_for_vLLM
from .config import LUT_MoE_vLLMConfig
from .quantizer import LUTQuantizer, train_lut_codebook, quantize_expert_weights
from .quant_method import LUTConfig, LUTFusedMoEMethod

__all__ = [
    "LUT_MoE_for_vLLM",
    "LUT_MoE_vLLMConfig",
    "LUTQuantizer",
    "train_lut_codebook",
    "quantize_expert_weights",
    "LUTConfig",
    "LUTFusedMoEMethod",
]
