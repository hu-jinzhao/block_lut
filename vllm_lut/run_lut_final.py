#!/usr/bin/env python3
"""Load and run LUT 4-bit model in vLLM."""
import os

os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] = '1'

from vllm_lut.quant_method import LUTConfig

if __name__ == '__main__':
    import time
    from vllm import LLM, SamplingParams

    print("Loading LUT 4-bit Qwen model...")
    t0 = time.time()
    llm = LLM(
        model='/home/hh/LUT-MoE/models/qwen_lut4bit',
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=64,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        quantization='lut',
    )
    print(f"Loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(temperature=0.7, max_tokens=10)
    out = llm.generate(['The future of AI is'], sp)
    print(f"Output: {out[0].outputs[0].text}")
    print("SUCCESS!")
