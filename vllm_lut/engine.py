# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
Main entry point for LUT-MoE vLLM integration.

Provides the LUT_MoE_for_vLLM class that wraps a vLLM engine
with LUT quantization and progressive loading of expert weights.

Usage:
    from vllm_lut import LUT_MoE_for_vLLM
    engine = LUT_MoE_for_vLLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        lut_config={"code_type": "BLOCKLUT", "lut_path": "/path/to/lut"}
    )
    output = engine.generate(["Hello, world!"])
"""

import os
from typing import Dict, List, Optional, Union

import torch

from vllm import LLM, SamplingParams

from .config import LUT_MoE_vLLMConfig
from .patcher import patch_model, restore_model


class LUT_MoE_for_vLLM:
    """
    LUT-MoE integrated vLLM engine.

    Loads a model through vLLM with LUT-quantized expert weights
    and optional progressive loading from SSD.

    Attributes:
        llm: The underlying vLLM LLM instance
        lut_config: LUT-MoE configuration
    """

    def __init__(
        self,
        model: str,
        lut_config: Optional[Union[str, Dict, LUT_MoE_vLLMConfig]] = None,
        **vllm_kwargs,
    ):
        """
        Initialize the LUT-MoE vLLM engine.

        Args:
            model: HuggingFace model name or local path
            lut_config: LUT-MoE configuration. Can be:
                - Path to a JSON config file
                - Dict of config parameters
                - LUT_MoE_vLLMConfig instance
                - None (uses defaults)
            **vllm_kwargs: Additional kwargs passed to vLLM LLM constructor
                           (e.g., tensor_parallel_size, max_model_len, etc.)
        """
        # Parse LUT config
        if isinstance(lut_config, str):
            import json
            with open(lut_config, "r") as f:
                config_dict = json.load(f)
            self.lut_config = LUT_MoE_vLLMConfig(**config_dict)
        elif isinstance(lut_config, dict):
            self.lut_config = LUT_MoE_vLLMConfig(**lut_config)
        elif isinstance(lut_config, LUT_MoE_vLLMConfig):
            self.lut_config = lut_config
        else:
            self.lut_config = LUT_MoE_vLLMConfig()

        # Setup offload path
        if not self.lut_config.offload_path:
            model_name_safe = model.replace("/", "_").replace("\\", "_")
            self.lut_config.offload_path = os.path.join(
                os.path.dirname(__file__) if "__file__" in dir() else ".",
                "..", "offload", model_name_safe
            )
        os.makedirs(self.lut_config.offload_path, exist_ok=True)

        if not self.lut_config.lut_path:
            self.lut_config.lut_path = self.lut_config.offload_path

        # Patch vLLM model classes
        print("[LUT-MoE] Patching vLLM model classes...")
        patch_model(self.lut_config)

        # Set default vLLM kwargs
        vllm_kwargs.setdefault("trust_remote_code", True)
        vllm_kwargs.setdefault("dtype", "bfloat16")

        # Initialize vLLM engine
        print(f"[LUT-MoE] Initializing vLLM with model: {model}")
        self.llm = LLM(model=model, **vllm_kwargs)

        self.model_name = model
        print("[LUT-MoE] Engine ready!")

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ):
        """
        Generate text using the LUT-MoE enhanced vLLM engine.

        Args:
            prompts: Input prompt(s)
            sampling_params: vLLM SamplingParams
            **kwargs: Additional kwargs

        Returns:
            vLLM RequestOutput(s)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        return self.llm.generate(prompts, sampling_params, **kwargs)

    def close(self):
        """Clean up resources."""
        restore_model()
        if hasattr(self.llm, '_engine'):
            # Graceful shutdown if possible
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
