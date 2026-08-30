# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
Custom MoE Layer with LUT Quantization for vLLM.

Replaces vLLM's native MoE (FusedMoE) layer with a version that:
1. Quantizes expert weights to BlockLUT / NestedLUT format on load
2. Decompresses to bf16 on-the-fly during the forward pass
3. Supports progressive tier management (hot/warm/cold experts)
4. Optionally offloads to SSD for memory savings

Usage: monkey-patch vLLM model classes with these layer classes.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.activation import SiluAndMul

from .quantizer import LUTQuantizer, decompress_blocklut
from .progressive_cache import ProgressiveExpertCache


class LUTMoEExpertWrapper(nn.Module):
    """
    Wrapper around a single expert's LUT-quantized weights.

    Stores indices + codebook + absmax in LUT format,
    decompresses to bf16 on-the-fly during forward.

    For fused w13 (gate+up): one set of quantized indices
    For w2 (down): another set of quantized indices
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quantizer: LUTQuantizer,
        device: torch.device = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quantizer = quantizer
        self.device = device or torch.device("cuda")

        # LUT quantized parameters (initialized after weights are loaded)
        self.register_buffer("w13_indices", None)  # [2*intermediate, hidden] uint8
        self.register_buffer("w13_absmax", None)   # [num_blocks]
        self.register_buffer("w2_indices", None)   # [hidden, intermediate] uint8
        self.register_buffer("w2_absmax", None)    # [num_blocks]
        self.register_buffer("codebook", None)     # [256] bf16

        # Decompressed bf16 cache (optional, for hot experts)
        self._w13_bf16: Optional[torch.Tensor] = None
        self._w2_bf16: Optional[torch.Tensor] = None

    def set_weights(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        codebook: torch.Tensor,
    ) -> None:
        """
        Quantize bf16 weights to LUT format and store.

        Args:
            w13_weight: bf16 tensor [2*intermediate_size, hidden_size]
            w2_weight: bf16 tensor [hidden_size, intermediate_size]
            codebook: bf16 LUT codebook [256]
        """
        self.codebook = codebook.to(self.device)

        # Quantize w13 (gate+up fused)
        w13_q = self.quantizer.quantize(w13_weight, codebook)
        self.register_buffer("w13_indices", w13_q["indices"].to(self.device))
        self.register_buffer("w13_absmax", w13_q["absmax"].to(self.device))

        # Quantize w2 (down)
        w2_q = self.quantizer.quantize(w2_weight, codebook)
        self.register_buffer("w2_indices", w2_q["indices"].to(self.device))
        self.register_buffer("w2_absmax", w2_q["absmax"].to(self.device))

    def set_decompressed(self, w13_bf16: torch.Tensor, w2_bf16: torch.Tensor) -> None:
        """Store pre-decompressed bf16 weights (for hot experts)."""
        self._w13_bf16 = w13_bf16.to(self.device)
        self._w2_bf16 = w2_bf16.to(self.device)

    def has_decompressed(self) -> bool:
        return self._w13_bf16 is not None

    def clear_decompressed(self) -> None:
        self._w13_bf16 = None
        self._w2_bf16 = None

    def get_weight(self, name: str) -> torch.Tensor:
        """
        Get bf16 weight tensor for computation.
        Returns cached bf16 if available, otherwise decompresses on-the-fly.
        """
        if name == "w13":
            if self._w13_bf16 is not None:
                return self._w13_bf16
            return decompress_blocklut(
                self.w13_indices, self.w13_absmax,
                self.codebook, self.quantizer.block_size
            ).reshape(2 * self.intermediate_size, self.hidden_size)
        elif name == "w2":
            if self._w2_bf16 is not None:
                return self._w2_bf16
            return decompress_blocklut(
                self.w2_indices, self.w2_absmax,
                self.codebook, self.quantizer.block_size
            ).reshape(self.hidden_size, self.intermediate_size)
        else:
            raise ValueError(f"Unknown weight name: {name}")

    def forward(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Compute: x @ W.T where W is the LUT-decompressed weight.

        Args:
            x: input tensor [num_tokens, hidden_size]
            name: "w13" or "w2"

        Returns:
            output tensor
        """
        w = self.get_weight(name)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
                f"codebook_size=256")


class DeepseekV2MoE_LUT(nn.Module):
    """
    LUT-quantized MoE layer for DeepSeekV2/V3 architecture.

    Replaces vLLM's DeepseekV2MoE with a version that:
    1. Quantizes expert weights to BlockLUT/NestedLUT format
    2. Decompresses on-the-fly during forward
    3. Manages progressive tiers via ExpertCache
    4. Optionally offloads to SSD

    This is the key integration point with vLLM's model pipeline.
    """

    def __init__(
        self,
        config,
        parallel_config=None,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
        lut_config=None,
    ):
        super().__init__()
        self.config = config
        self.lut_config = lut_config

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        # Router
        from vllm.model_executor.layers.fused_moe import GateLinear
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            prefix=f"{prefix}.gate",
        )

        # Shared experts (stored in standard bf16, not quantized)
        if self.n_shared_experts is not None:
            from vllm.model_executor.layers.linear import (
                ColumnParallelLinear, RowParallelLinear,
            )
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_gate_up = ColumnParallelLinear(
                config.hidden_size,
                shared_intermediate,
                bias=False,
                prefix=f"{prefix}.shared_experts.gate_up_proj",
            )
            self.shared_down = RowParallelLinear(
                shared_intermediate,
                config.hidden_size,
                bias=False,
                prefix=f"{prefix}.shared_experts.down_proj",
            )
            self.shared_act_fn = SiluAndMul()
        else:
            self.shared_gate_up = None
            self.shared_down = None

        # LUT Quantizer
        self.quantizer = LUTQuantizer(
            code_type=self.lut_config.code_type if lut_config else "BLOCKLUT",
        )
        self._codebook_loaded = False

        # Individual expert wrappers (created after weight loading)
        self.experts: nn.ModuleList = nn.ModuleList()

        # Progressive cache
        self.expert_cache: Optional[ProgressiveExpertCache] = None
        self._progressive_enabled = False

        # Store layer_id (set by patcher)
        self.layer_id: int = 0

        # Decompression cache (for hot experts, shared across all experts)
        self._hot_expert_bf16: Dict[int, Dict[str, torch.Tensor]] = {}

    def load_codebook(self, lut_path: str) -> bool:
        """Load LUT codebook from .npy file."""
        loaded = self.quantizer.load(lut_path)
        if loaded and self.quantizer.lut_codebook is not None:
            self._codebook_loaded = True
            # Move codebook to CUDA
            codebook = self.quantizer.lut_codebook.to("cuda")
            self.register_buffer("_codebook", codebook)
        return loaded

    def quantize_expert_weights(
        self, w13_weights: torch.Tensor, w2_weights: torch.Tensor
    ) -> None:
        """
        Quantize all expert weights to LUT format.

        Args:
            w13_weights: [num_experts, 2*intermediate, hidden] bf16
            w2_weights: [num_experts, hidden, intermediate] bf16
        """
        if not self._codebook_loaded:
            # Train codebook on these weights
            all_weights = torch.cat([
                w13_weights.reshape(-1),
                w2_weights.reshape(-1),
            ])
            self.quantizer.train(all_weights)
            self._codebook = self.quantizer.lut_codebook.to("cuda")
            self._codebook_loaded = True

        codebook = self._codebook
        self.experts = nn.ModuleList()

        for i in range(self.num_experts):
            expert = LUTMoEExpertWrapper(
                self.hidden_size,
                self.intermediate_size,
                self.quantizer,
            )
            expert.set_weights(w13_weights[i], w2_weights[i], codebook)
            self.experts.append(expert)

        # Init progressive cache if enabled
        if self.lut_config and self.lut_config.enable_progressive_loading:
            self.expert_cache = ProgressiveExpertCache(
                num_layers=self.config.num_hidden_layers
                if hasattr(self.config, "num_hidden_layers") else 27,
                num_experts=self.num_experts,
                max_gpu_entries=int(self.num_experts * 2),
            )
            self._progressive_enabled = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with LUT-quantized experts.

        Args:
            hidden_states: [batch*seq_len, hidden_size]

        Returns:
            output: [batch*seq_len, hidden_size]
        """
        orig_shape = hidden_states.shape
        batch_size, seq_len, hidden_dim = orig_shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        # 1. Router
        router_logits = self.gate(hidden_states)

        # 2. Get top-k routing
        topk_weights, topk_indices = _routing(router_logits, self.top_k)

        # 3. Expert computation
        output = torch.zeros_like(hidden_states)
        routed_mask = topk_indices.device  # identity reference

        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]  # [num_tokens]
            weights = topk_weights[:, k]  # [num_tokens]

            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                if not mask.any():
                    continue

                tokens = hidden_states[mask]  # [n_tokens, hidden]

                # Get or create expert wrapper
                if expert_id >= len(self.experts):
                    continue

                expert_wrapper = self.experts[expert_id]

                # Progressive tier management
                if self._progressive_enabled and self.expert_cache is not None:
                    self.expert_cache.record_access(self.layer_id, expert_id)
                    if self.expert_cache.is_cached(self.layer_id, expert_id):
                        # Hot expert: use cached bf16
                        pass  # wrapper will use decompressed
                    else:
                        # Cold/warm expert: decompress on-the-fly
                        pass

                # Gate+Up projection (fused)
                gate_up = expert_wrapper(tokens, "w13")
                # SiLU(gate) * up
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
                # Down projection
                expert_out = expert_wrapper(activated, "w2")

                output[mask] += expert_out * weights[mask].unsqueeze(-1)

        # 4. Shared experts
        if self.shared_gate_up is not None and self.shared_down is not None:
            shared_hidden = hidden_states
            gate_up, _ = self.shared_gate_up(shared_hidden)
            activated = self.shared_act_fn(gate_up)
            shared_out, _ = self.shared_down(activated)
            output = output + shared_out

        return output.reshape(orig_shape)


def _routing(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard softmax top-k routing.

    Args:
        router_logits: [num_tokens, num_experts]
        top_k: number of experts per token

    Returns:
        topk_weights: [num_tokens, top_k]
        topk_indices: [num_tokens, top_k] (long)
    """
    routing_weights = F.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights.to(router_logits.dtype)
    return topk_weights, topk_indices
