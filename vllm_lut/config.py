# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
LUT-MoE vLLM Configuration.

Adapts the original LUT-MoEConfig for the vLLM inference framework.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LUT_MoE_vLLMConfig:
    """
    Configuration for LUT-MoE integration with vLLM.

    Controls LUT quantization type, progressive loading tiers,
    cache sizes, and expert prediction/prefetching parameters.
    """

    # --- LUT Quantization ---
    code_type: str = field(
        default="BLOCKLUT",
        metadata={
            "help": "Quantization algorithm: 'BLOCKLUT' (8-bit), "
                    "'NESTEDLUT' (4/6/8-bit progressive), "
                    "'LUT' (basic 8-bit, no block-wise)"
        }
    )

    lut_path: str = field(
        default="",
        metadata={"help": "Path to pre-trained LUT codebook .npy file. "
                          "Auto-generated if empty."}
    )

    # --- Progressive Loading Tiers ---
    lut_tier: int = field(
        default=0,
        metadata={"help": "Initial LUT tier: 0=full256 (8-bit, best quality), "
                          "1=mapped64 (6-bit), 2=mapped16 (4-bit, fastest load)"}
    )

    nested_shallow_layers: int = field(
        default=8,
        metadata={"help": "Number of shallow layers always using 8-bit in static mode."}
    )

    enable_progressive_loading: bool = field(
        default=False,
        metadata={"help": "Enable SSD offload and progressive loading of experts. "
                          "When False, experts stay in GPU in LUT format."}
    )

    # --- Cache Settings ---
    gpu_cache_ratio: float = field(
        default=0.6,
        metadata={"help": "GPU cache ratio for decompressed expert weights. "
                          "0.0 = decompress on every use (no cache). "
                          "1.0 = decompress and keep all experts in bf16."}
    )

    device_memory_ratio: float = field(
        default=0.90,
        metadata={"help": "Maximum fraction of GPU memory to use for the expert cache."}
    )

    # --- Expert Prefetching ---
    prefetcher_topk: int = field(
        default=3,
        metadata={"help": "Number of experts to prefetch for the next layer. "
                          "0 = disable prefetching."}
    )

    expert_prediction_window: int = field(
        default=26 * 6 * 10,
        metadata={"help": "History window size for expert frequency prediction."}
    )

    # --- Offload Path ---
    offload_path: str = field(
        default="",
        metadata={"help": "Path to store/load compressed expert binaries. "
                          "Defaults to '{model_path}/lut_offload'."}
    )

    # --- Debug ---
    verbose: bool = field(
        default=False,
        metadata={"help": "Print detailed debug information."}
    )

    # --- vLLM-specific ---
    use_vllm_fused_kernel: bool = field(
        default=True,
        metadata={
            "help": "When True: decompress experts to bf16 then use vLLM's fused MoE kernel. "
                    "When False: use per-expert PyTorch matmul (no fused kernel, "
                    "supports progressive loading)."
        }
    )

    def __post_init__(self):
        if self.code_type not in ("LUT", "BLOCKLUT", "NESTEDLUT"):
            raise ValueError(
                f"Unsupported code_type: {self.code_type}. "
                f"Must be one of: LUT, BLOCKLUT, NESTEDLUT"
            )
        if self.lut_tier not in (0, 1, 2):
            raise ValueError(f"lut_tier must be 0, 1, or 2, got {self.lut_tier}")
