"""
LUT Quantization Method for vLLM.

Stores expert weights as uint8 LUT indices + bf16 absmax + codebook.
Creates w13_weight/w2_weight as uint8 so the checkpoint's LUT data
loads directly with standard weight_loader.
"""

import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.layer import MoEActivation

from .quantizer import LUTQuantizer, decompress_blocklut
from .cuda_lut_gemv import lut_gemv as _lut_gemv


class LUTConfig(QuantizationConfig):
    """LUT quantization config for vLLM."""
    _layer_counter = 0  # Global counter for layer_id

    def __init__(self, code_type: str = "BLOCKLUT", lut_path: str = "",
                 block_size: int = 128, **kwargs):
        super().__init__()
        self.code_type = code_type
        self.lut_path = lut_path or kwargs.get('lut_path', '')
        self.block_size = block_size
        # Debug: log init
        import sys
        print(f"[LUT] LUTConfig init: code_type={code_type}, "
              f"lut_path={self.lut_path}, kwargs keys={list(kwargs.keys())}",
              file=sys.stderr)

    @classmethod
    def get_name(cls) -> str:
        return "lut"

    @classmethod
    def get_supported_act_dtypes(cls) -> list:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LUTConfig":
        return cls(**config)

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.linear import (
            LinearBase, UnquantizedLinearMethod,
        )
        if isinstance(layer, RoutedExperts):
            return LUTFusedMoEMethod(self)
        elif isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


class LUTFusedMoEMethod(FusedMoEMethodBase):
    """LUT quantization: uint8 indices + codebook instead of bf16 weights."""

    def __init__(self, quant_config: LUTConfig):
        # The actual moe config isn't needed since we handle things ourselves
        # Just create a minimal stub to satisfy parent class
        from types import SimpleNamespace
        dummy_cfg = SimpleNamespace()
        dummy_cfg.is_act_and_mul = True
        dummy_cfg.has_bias = False
        dummy_cfg.is_gated = False
        dummy_cfg.tp_size = 1
        dummy_cfg.ep_size = 1
        dummy_cfg.routing_method = 'default'
        dummy_cfg.hidden_dim = 1
        dummy_cfg.intermediate_size = 1
        dummy_cfg.num_experts = 1
        dummy_cfg.experts_per_token = 1
        dummy_cfg.num_local_experts = 1
        dummy_cfg.num_logical_experts = 1
        dummy_cfg.moe_parallel_config = SimpleNamespace()
        dummy_cfg.moe_parallel_config.tp_size = 1
        dummy_cfg.moe_parallel_config.ep_size = 1
        dummy_cfg.moe_parallel_config.tp_rank = 0
        dummy_cfg.moe_parallel_config.sp_size = 1
        super().__init__(dummy_cfg)
        self.quant_config = quant_config
        self.quantizer = LUTQuantizer(code_type=quant_config.code_type,
                                      block_size=quant_config.block_size)

    # ------------------------------------------------------------------
    # Create uint8 parameters (loads directly from LUT checkpoint)
    # ------------------------------------------------------------------

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra):
        """Create uint8 w13_weight/w2_weight (LUT indices) + bf16 absmax + codebook."""
        device = "cuda"

        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition

        # LUT indices stored as uint8 (8-bit index per element)
        w13 = torch.nn.Parameter(
            torch.empty(num_experts, w13_up_dim, hidden_size,
                        dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13)
        set_weight_attrs(w13, extra)

        w2 = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition,
                        dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2)
        set_weight_attrs(w2, extra)

        # Block absmax (one per 128 elements)
        bs = self.quant_config.block_size
        w13_blocks = math.ceil(w13.numel() / num_experts / bs)
        w2_blocks = math.ceil(w2.numel() / num_experts / bs)

        w13_a = torch.nn.Parameter(
            torch.empty(num_experts, w13_blocks, dtype=torch.bfloat16, device=device),
            requires_grad=False)
        layer.register_parameter("w13_absmax", w13_a)

        w2_a = torch.nn.Parameter(
            torch.empty(num_experts, w2_blocks, dtype=torch.bfloat16, device=device),
            requires_grad=False)
        layer.register_parameter("w2_absmax", w2_a)

        # Codebook (shared)
        cb = torch.nn.Parameter(
            torch.zeros(256, dtype=torch.bfloat16, device=device),
            requires_grad=False)
        layer.register_parameter("_codebook", cb)

    # ------------------------------------------------------------------
    # After loading: load codebook from file (if available)
    # ------------------------------------------------------------------

    def process_weights_after_loading(self, layer):
        """Load LUT codebook and absmax, prepare for forward."""
        import sys
        print(f"[LUT] process_weights_after_loading: has_w13={hasattr(layer, 'w13_weight')}, "
              f"type={type(layer).__name__}", file=sys.stderr, flush=True)
        if not hasattr(layer, 'w13_weight'):
            return  # Skip non-RoutedExperts modules
        if getattr(layer, "_lut_codebook_loaded", False):
            return

        # Load codebook
        cb_path = self.quant_config.lut_path or ""
        npz_path = os.path.join(cb_path, "lut_absmax.npz") if cb_path else ""
        print(f"[LUT] pw: cb_path='{cb_path}', npz_exists={os.path.exists(npz_path) if cb_path else False}",
              file=sys.stderr, flush=True)
        if cb_path and os.path.isdir(cb_path):
            self.quantizer.load(cb_path)
            if self.quantizer.lut_codebook is not None:
                layer._codebook.copy_(self.quantizer.lut_codebook)

        # Load absmax from npz
        if cb_path and os.path.exists(npz_path):
            import numpy as np, torch
            data = np.load(npz_path)
            expert_keys = [k for k in data.files if 'experts.' in k
                           and 'shared_expert' not in k and k.endswith('.weight')]
            layer_ids = sorted(set(
                parts[i+1] for k in expert_keys
                for parts in [k.split('.')]
                for i, p in enumerate(parts) if p == 'layers'
            ), key=int)
            if not layer_ids:
                layer._lut_codebook_loaded = True
                return

            # Count only RoutedExperts calls (skip LinearBase calls)
            if not hasattr(self, '_pw_moe_idx'):
                self._pw_moe_idx = 0
            lid = layer_ids[self._pw_moe_idx % len(layer_ids)]
            self._pw_moe_idx += 1

            n_exp = layer.w13_weight.shape[0]
            for e in range(n_exp):
                gk = f"model.layers.{lid}.mlp.experts.{e}.gate_proj.weight"
                uk = f"model.layers.{lid}.mlp.experts.{e}.up_proj.weight"
                dk = f"model.layers.{lid}.mlp.experts.{e}.down_proj.weight"
                if gk in data and uk in data:
                    gate_a = torch.from_numpy(data[gk])
                    up_a = torch.from_numpy(data[uk])
                    layer.w13_absmax.data[e] = torch.cat([gate_a, up_a]).to(
                        device=layer.w13_absmax.device, dtype=torch.bfloat16)
                if dk in data:
                    layer.w2_absmax.data[e] = torch.from_numpy(data[dk]).to(
                        device=layer.w2_absmax.device, dtype=torch.bfloat16)

        layer._lut_codebook_loaded = True

    # ------------------------------------------------------------------
    # Forward: decompress to bf16 + compute
    # ------------------------------------------------------------------

    def apply(self, layer, x, topk_weights, topk_ids, **kwargs):
        """MoE with LUT: cache unique IDs across layers, minimized overhead."""
        num_tokens, hidden_size = x.shape
        out = torch.zeros(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
        ni = layer.w13_weight.shape[1] // 2
        cb = layer._codebook.to(x.device)

        # Cache unique expert IDs across layers (topk_ids same for all layers)
        cache = self.__class__
        if not hasattr(cache, '_cached') or cache._cached.get('ptr') != id(topk_ids):
            cache._cached = {
                'ptr': id(topk_ids),
                'ids': torch.unique(topk_ids),
                'list': [],
            }
            for e in cache._cached['ids'].tolist():
                cache._cached['list'].append(e)
        unique_list = cache._cached['list']
        n_selected = len(unique_list)

        if num_tokens == 1:
            for i, eid in enumerate(unique_list):
                gate_up = _lut_gemv(
                    x[0], layer.w13_weight[eid], cb, layer.w13_absmax[eid])
                activated = F.silu(gate_up[:ni]) * gate_up[ni:]
                expert_out = _lut_gemv(
                    activated, layer.w2_weight[eid], cb, layer.w2_absmax[eid])
                for k in range(topk_ids.shape[-1]):
                    m = topk_ids[:, k] == eid
                    if not m.any(): continue
                    out[m] += expert_out * topk_weights[m, k].unsqueeze(-1)
        else:
            for eid in unique_list:
                flat13 = layer.w13_weight[eid].reshape(-1).long()
                w13 = (cb[flat13] * layer.w13_absmax[eid][
                    torch.arange(flat13.numel(), device=x.device)//128
                ].clamp(max=layer.w13_absmax.shape[1]-1)).reshape(2*ni, hidden_size)
                gate_up = F.linear(x, w13)
                activated = F.silu(gate_up[:, :ni]) * gate_up[:, ni:]
                flat2 = layer.w2_weight[eid].reshape(-1).long()
                w2 = (cb[flat2] * layer.w2_absmax[eid][
                    torch.arange(flat2.numel(), device=x.device)//128
                ].clamp(max=layer.w2_absmax.shape[1]-1)).reshape(hidden_size, ni)
                expert_out = F.linear(activated, w2)
                for k in range(topk_ids.shape[-1]):
                    m = topk_ids[:, k] == eid
                    if not m.any(): continue
                    out[m] += expert_out[m] * topk_weights[m, k].unsqueeze(-1)

        return out

    def _decompress_one(self, indices, absmax, codebook, block_size=128):
        """Decompress a single expert's LUT weights (single GPU gather)."""
        flat = indices.reshape(-1)
        codebook_d = codebook.to(flat.device)
        normalized = codebook_d[flat.long()]  # single gather
        N = flat.shape[0]
        block_id = torch.arange(N, device=flat.device) // block_size
        block_id = block_id.clamp(max=absmax.shape[0] - 1)
        scale = absmax[block_id]
        return normalized * scale

    def _batch_decompress(self, indices, absmax, codebook, block_size=128):
        """Decompress ALL experts in a single batched GPU operation."""
        n_exp = indices.shape[0]
        flat_idx = indices.reshape(n_exp, -1)  # [n_exp, N]
        flat_abs = absmax.reshape(n_exp, -1)   # [n_exp, num_blocks]

        # Single gather: all experts at once
        normalized = codebook.to(flat_idx.device)[flat_idx.long()]  # [n_exp, N]

        # Single broadcast multiply
        N = flat_idx.shape[1]
        block_id = torch.arange(N, device=flat_idx.device).reshape(1, -1) // block_size
        block_id = block_id.clamp(max=flat_abs.shape[1] - 1)
        scale = flat_abs.gather(1, block_id)  # [n_exp, N]
        result = normalized * scale  # [n_exp, N]

        return result.reshape(indices.shape)

    def get_fused_moe_quant_config(self, layer=None) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantDesc
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        desc = FusedMoEQuantDesc(dtype=torch.uint8)
        return FusedMoEQuantConfig(
            _a1=desc, _a2=desc, _w1=desc, _w2=desc,
        )

    def __repr__(self):
        return f"LUTFusedMoEMethod({self.quant_config.code_type})"


# Auto-register LUT with vLLM on import
try:
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    )
    if 'lut' not in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.append('lut')
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG['lut'] = LUTConfig
except ImportError:
    pass  # vLLM not yet available
