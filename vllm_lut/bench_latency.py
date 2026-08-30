#!/usr/bin/env python3
"""Measure TTFT and TPOT for LUT 4-bit model in vLLM."""
import os, time
os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] = '1'

import torch

# Register LUT quant
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
)
import vllm_lut.quant_method
from vllm_lut.quant_method import LUTConfig
if 'lut' not in QUANTIZATION_METHODS:
    QUANTIZATION_METHODS.append('lut')
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG['lut'] = LUTConfig

PROMPT = "Explain what is artificial intelligence in simple terms:"

if __name__ == '__main__':
    from vllm import LLM, SamplingParams

    print("Loading LUT 4-bit Qwen model...")
    t0 = time.time()
    llm = LLM(
        model='/home/hh/LUT-MoE/models/qwen_lut4bit',
        trust_remote_code=True, dtype='bfloat16',
        max_model_len=128, gpu_memory_utilization=0.9,
        enforce_eager=True, quantization='lut',
    )
    print(f"Load time: {time.time()-t0:.1f}s")

    # Warm up
    print("Warming up...")
    _ = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print()

    # Measure TTFT (time to first token) - generate just 1 token
    print("=== TTFT (Time To First Token) ===")
    sp_ttft = SamplingParams(temperature=0.7, max_tokens=1)
    ttft_times = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([PROMPT], sp_ttft)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        ttft_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.0f}ms")

    avg_ttft = sum(ttft_times) / len(ttft_times)
    print(f"  Average TTFT: {avg_ttft*1000:.0f}ms")

    # Measure TPOT (time per output token)
    # Generate N tokens, then TPOT = total_time / (N-1) for decode tokens
    print("\n=== TPOT (Time Per Output Token) ===")
    sp_tpot = SamplingParams(temperature=0.7, max_tokens=50)
    tpot_times = []
    all_tokens = []
    for i in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([PROMPT], sp_tpot)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n = len(out[0].outputs[0].token_ids)
        # TPOT = (total - ttft) / (n-1) for decode phase
        decode_time = elapsed - avg_ttft
        tpot = decode_time / max(n - 1, 1)
        tpot_times.append(tpot)
        all_tokens.append(n)
        print(f"  Run {i+1}: {elapsed:.3f}s total, {n} tokens, "
              f"TPOT≈{tpot*1000:.0f}ms, {n/elapsed:.1f} tok/s")

    avg_tpot = sum(tpot_times) / len(tpot_times)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    print(f"\n  Average TPOT: {avg_tpot*1000:.0f}ms")
    print(f"  Average throughput: {avg_tokens/avg_tpot:.1f} tok/s")
    print()
    print("=" * 45)
    print(f"  TTFT:  {avg_ttft*1000:.0f} ms")
    print(f"  TPOT:  {avg_tpot*1000:.0f} ms")
    print(f"  Tok/s: {avg_tokens/avg_tpot:.1f}")
    print("=" * 45)
