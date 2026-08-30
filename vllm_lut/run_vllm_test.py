#!/usr/bin/env python3
"""Run vLLM inference test with UVA patch."""
import os, sys, time

# Environment setup
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("CUDA_HOME", "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13")
cuda_bin = os.environ["CUDA_HOME"] + "/bin"
if cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
cuda_lib = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
if cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from vllm import LLM, SamplingParams


def main():
    print("Loading model...")
    t0 = time.time()
    llm = LLM(
        model="/home/hh/LUT-MoE/models/deepseek",
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(temperature=0.7, max_tokens=20)
    out = llm.generate(["The future of AI is"], sp)
    text = out[0].outputs[0].text
    print(f"Output: {text}")
    print("SUCCESS!")


if __name__ == "__main__":
    main()
