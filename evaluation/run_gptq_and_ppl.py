#!/usr/bin/env python3
"""
GPTQ 4bit quantization (desc_act=True) + PPL evaluation on Qwen-MoE.
Uses gptqmodel with CPU offloading (245GB RAM).
"""
import os, sys, json, math, time, glob
sys.path.insert(0, '/home/hh/LUT-MoE')

import torch
import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
OUTPUT_DIR = "/tmp/qwen_gptq_4bit_new"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"

# ==============================================================
# Step 1: GPTQ Quantization
# ==============================================================

def quantize_gptq():
    print("=" * 60)
    print("GPTQ 4bit Quantization (desc_act=True, group_size=128)")
    print("=" * 60)

    from gptqmodel import GPTQModel, QuantizeConfig
    from datasets import load_dataset

    # Load calibration data (small subset: 128 samples)
    print("\nLoading calibration data...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = dataset["text"][:128]
    except:
        # Local fallback
        with open(DATASET) as f:
            texts = json.load(f)[:128]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # Quantize config with desc_act=True (real GPTQ)
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        damp_percent=0.01,
        true_sequential=True,
    )

    print(f"Loading model on CPU (this will take a moment, 245GB RAM)...")
    t0 = time.time()

    model = GPTQModel.load(
        MODEL_DIR,
        quantize_config=quantize_config,
        device="cpu",  # Load to CPU (we have 245GB)
        trust_remote_code=True,
        offload_to_disk=True,  # Layer-by-layer offloading
    )

    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print(f"\nQuantizing with {len(texts)} calibration samples...")
    print("  This will take 1-2 hours on CPU...")
    t1 = time.time()

    # Quantize
    model.quantize(
        tokenizer=tokenizer,
        calib_dataset=texts,
        batch_size=1,
        calib_batch_size=1,
    )

    print(f"  Quantization complete in {(time.time()-t1)/60:.1f}min")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_quantized(OUTPUT_DIR)
    print(f"  Model saved to {OUTPUT_DIR}")

    return OUTPUT_DIR


# ==============================================================
# Step 2: PPL Evaluation
# ==============================================================

def eval_ppl(model_path, method_name):
    """Evaluate PPL using vLLM's logprobs API."""
    print(f"\n{'='*60}")
    print(f"PPL Evaluation: {method_name}")
    print('='*60)

    # Register LUT method
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS, _CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    )
    from vllm_lut.quant_method import LUTConfig
    if 'lut' not in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.append('lut')
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG['lut'] = LUTConfig

    from vllm import LLM, SamplingParams

    # Load model
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    # Load dataset
    with open(DATASET) as f:
        texts = json.load(f)
    full_text = " ".join(texts)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    enc = tokenizer.encode(full_text)
    print(f"  WikiText-2: {len(enc)} tokens")

    # PPL with sliding window
    max_len = 2048
    stride = 1024
    nlls = []
    total_tok = 0
    t0 = time.time()

    for i in range(0, len(enc), stride):
        chunk = enc[i:i+max_len]
        if len(chunk) < 128:
            break

        prompt = tokenizer.decode(chunk[:-1])
        sp = SamplingParams(max_tokens=1, logprobs=1, prompt_logprobs=len(chunk)-1)
        out = llm.generate([prompt], sp)

        prompt_logprobs = out[0].prompt_logprobs
        if prompt_logprobs:
            for lp in prompt_logprobs:
                if lp is not None:
                    vals = list(lp.values())
                    if vals:
                        nlls.append(-vals[0].logprob)
                        total_tok += 1

        if (i // stride + 1) % 10 == 0:
            ppl = math.exp(sum(nlls) / max(total_tok, 1))
            print(f"  step {i//stride+1}: PPL={ppl:.4f}")

    ppl = math.exp(sum(nlls) / max(total_tok, 1))
    print(f"\n{method_name}: PPL={ppl:.4f} ({total_tok} tokens, {time.time()-t0:.1f}s)")
    return ppl


# ==============================================================
# Main
# ==============================================================

if __name__ == '__main__':
    results = {}

    # Step 1: Quantize
    gptq_path = quantize_gptq()

    # Step 2: Evaluate GPTQ PPL
    ppl_gptq = eval_ppl(gptq_path, "GPTQ 4bit")
    results['GPTQ_4bit'] = ppl_gptq

    # Step 3: Evaluate LUT PPL (using existing model)
    ppl_lut = eval_ppl("/home/hh/LUT-MoE/models/qwen_lut4bit", "LUT 4bit")
    results['LUT_4bit'] = ppl_lut

    # Final comparison
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)
    for name, ppl in results.items():
        print(f"  {name:20s}: PPL = {ppl:.4f}")
    if len(results) >= 2:
        v = list(results.values())
        diff = v[0] - v[1]
        better = "GPTQ" if diff < 0 else "LUT"
        print(f"  Difference: {abs(diff):.4f} ({better} better)")
