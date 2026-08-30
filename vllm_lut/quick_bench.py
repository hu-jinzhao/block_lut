#!/usr/bin/env python3
"""Quick single-run benchmark for LUT 4-bit."""
import os, sys
os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] = '1'
sys.path.insert(0, '/home/hh/LUT-MoE')

import torch, time

# Register LUT
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
import vllm_lut.quant_method
from vllm_lut.quant_method import LUTConfig
if 'lut' not in QUANTIZATION_METHODS:
    QUANTIZATION_METHODS.append('lut')
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG['lut'] = LUTConfig

from vllm import LLM, SamplingParams

if __name__ == '__main__':
    print("Loading...", flush=True)
    t0 = time.time()
    llm = LLM(model='/home/hh/LUT-MoE/models/qwen_lut4bit', trust_remote_code=True,
              dtype='bfloat16', max_model_len=128, gpu_memory_utilization=0.9,
              enforce_eager=True, quantization='lut')
    print(f"Load: {time.time()-t0:.1f}s", flush=True)

    # Single TTFT measurement
    sp = SamplingParams(temperature=0.7, max_tokens=1)
    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(['Explain AI in simple terms'], sp)
    torch.cuda.synchronize()
    ttft = time.time() - t0
    print(f"TTFT (1 tok): {ttft:.2f}s", flush=True)

    # Single throughput measurement
    sp2 = SamplingParams(temperature=0.7, max_tokens=20)
    torch.cuda.synchronize()
    t0 = time.time()
    out2 = llm.generate(['Explain AI in simple terms'], sp2)
    torch.cuda.synchronize()
    total = time.time() - t0
    n = len(out2[0].outputs[0].token_ids)
    tpot = (total - ttft) / max(n - 1, 1)
    print(f"Generate {n} tokens: {total:.2f}s", flush=True)
    print(f"TPOT: {tpot*1000:.0f}ms", flush=True)
    print(f"Throughput: {n/total:.1f} tok/s", flush=True)
    print(f"Output: {out2[0].outputs[0].text!r}", flush=True)
    print("DONE", flush=True)
