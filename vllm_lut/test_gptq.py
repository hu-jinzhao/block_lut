#!/usr/bin/env python3
"""Test loading official GPTQ model in vLLM."""
import os
os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, '/home/hh/LUT-MoE')

from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
)
from vllm_lut.quant_method import LUTConfig
if 'lut' not in QUANTIZATION_METHODS:
    QUANTIZATION_METHODS.append('lut')
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG['lut'] = LUTConfig

from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(
        model='/home/hh/LUT-MoE/models/qwen_gptq',
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=64,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        quantization='moe_wna16',
    )
    print('MODEL LOADED!')
    sp = SamplingParams(temperature=0.7, max_tokens=10)
    out = llm.generate(['Explain AI'], sp)
    print(f'Output: {out[0].outputs[0].text}')
