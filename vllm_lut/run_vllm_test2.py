#!/usr/bin/env python3
"""Run vLLM inference on DeepSeek model."""
import os, sys, time

# Must set env BEFORE importing vllm
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["CUDA_HOME"] = "/home/hh/miniconda3/envs/lut_moe_cu124"
os.environ["LD_LIBRARY_PATH"] = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib"

from vllm import LLM, SamplingParams

def main():
    print("Loading model...", flush=True)
    t0 = time.time()
    llm = LLM(
        model="/home/hh/LUT-MoE/models/deepseek",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.7, max_tokens=20)
    out = llm.generate(["The future of AI is"], sp)
    print(f"Output: {out[0].outputs[0].text}", flush=True)
    print("SUCCESS!", flush=True)

if __name__ == "__main__":
    main()
