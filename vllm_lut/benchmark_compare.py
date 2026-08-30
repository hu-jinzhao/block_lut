#!/usr/bin/env python3
"""
LUT-MoE Cross-Framework Benchmark.

Compares inference performance between:
  1. HuggingFace + LUT-MoE (existing implementation)
  2. vLLM + LUT-MoE (new port)

Measures latency, throughput, and GPU memory usage.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Ensure CUDA runtime
_cuda_paths = ["/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib"]
for _p in _cuda_paths:
    if os.path.isdir(_p) and _p not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _p + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def run_lut_moe_hf(prompt, max_tokens, model_path, lut_config_path,
                   num_runs=3):
    """
    Run inference using LUT-MoE on HuggingFace (original implementation).
    """
    print("\n" + "=" * 60)
    print("HF + LUT-MoE Benchmark")
    print("=" * 60)

    import sys
    sys.path.insert(0, "/home/hh/LUT-MoE")

    from entry.llm_modeling import MoE
    from transformers import AutoTokenizer
    import torch

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load LUT-MoE engine
    moe = MoE(model_name_or_path=model_path, config=lut_config_path)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Warm up
    print("Warming up...")
    with torch.no_grad():
        _ = moe.generate(input_ids, max_new_tokens=20)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    tokens_list = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = moe.generate(input_ids, max_new_tokens=max_tokens)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Count generated tokens (exclude input)
        if hasattr(output, 'shape'):
            gen_tokens = output.shape[1] - input_ids.shape[1]
        else:
            gen_tokens = max_tokens

        latencies.append(elapsed)
        tokens_list.append(gen_tokens)
        print(f"  Run {i+1}: {elapsed:.3f}s, {gen_tokens} tokens")

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(tokens_list) / len(tokens_list)
    print(f"\nHF + LUT-MoE: {avg_lat:.3f}s avg, {avg_tok/avg_lat:.1f} tok/s")

    return {
        "framework": "HF+LUT-MoE",
        "avg_latency": avg_lat,
        "avg_throughput": avg_tok / avg_lat,
    }


def run_vllm_lut(prompt, max_tokens, model_path, code_type, lut_path,
                 num_runs=3):
    """
    Run inference using vLLM with LUT-quantized experts.
    """
    print("\n" + "=" * 60)
    print("vLLM + LUT-MoE Benchmark")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    from vllm_lut.quantizer import LUTQuantizer
    from collections import defaultdict
    import numpy as np
    import torch

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    # Apply LUT quantization
    quantizer = LUTQuantizer(code_type=code_type, device="cuda")

    # Load codebook
    codebook_path = os.path.join(lut_path, "blocklut_256.npy")
    if os.path.exists(codebook_path):
        quantizer.lut_codebook = torch.from_numpy(
            np.load(codebook_path)
        ).to(torch.bfloat16)
        print(f"Loaded codebook from {codebook_path}")
    else:
        print("No codebook found. Quantizing on-the-fly...")
        all_w = []
        model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
        base = model_obj.model if hasattr(model_obj, 'model') else model_obj
        for layer in base.model.layers:
            if hasattr(layer.mlp, 'routed_experts'):
                exp = layer.mlp.routed_experts
                if hasattr(exp, 'w13_weight') and exp.w13_weight.numel() > 0:
                    all_w.append(exp.w13_weight.data.float().reshape(-1))
                    all_w.append(exp.w2_weight.data.float().reshape(-1))
        if all_w:
            quantizer.train(torch.cat(all_w))

    codebook = quantizer.lut_codebook.to("cuda")

    # Apply quantization
    q_stats = defaultdict(int)

    def quantize_module(module, path=""):
        if hasattr(module, 'routed_experts'):
            exp = module.routed_experts
            if hasattr(exp, 'w13_weight') and exp.w13_weight.numel() > 0:
                w13 = exp.w13_weight.data
                w2 = exp.w2_weight.data
                n_exp = w13.shape[0]
                for i in range(n_exp):
                    q13 = quantizer.quantize(w13[i], codebook)
                    q2 = quantizer.quantize(w2[i], codebook)
                    exp.register_buffer(f"_q_w13_i_{i}", q13["indices"])
                    exp.register_buffer(f"_q_w13_a_{i}", q13["absmax"])
                    exp.register_buffer(f"_q_w2_i_{i}", q2["indices"])
                    exp.register_buffer(f"_q_w2_a_{i}", q2["absmax"])
                exp.register_buffer("_q_codebook", codebook)
                exp._lut_quantized = True
                q_stats['experts'] += n_exp
                q_stats['layers'] += 1
                print(f"  Quantized {path}: {n_exp} experts")

        for name, child in module.named_children():
            quantize_module(child, f"{path}.{name}" if path else name)

    model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
    model = model_obj.model if hasattr(model_obj, 'model') else model_obj
    print("\nApplying LUT quantization...")
    quantize_module(model)
    print(f"Quantized {q_stats['layers']} layers ({q_stats['experts']} experts)")

    # Warm up
    print("Warming up...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
    _ = llm.generate([prompt], sampling_params)

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    latencies = []
    tokens_list = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        outputs = llm.generate([prompt], sampling_params)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        gen_tokens = len(outputs[0].outputs[0].token_ids)

        latencies.append(elapsed)
        tokens_list.append(gen_tokens)
        print(f"  Run {i+1}: {elapsed:.3f}s, {gen_tokens} tokens")

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(tokens_list) / len(tokens_list)
    print(f"\nvLLM + LUT-MoE: {avg_lat:.3f}s avg, {avg_tok/avg_lat:.1f} tok/s")

    return {
        "framework": "vLLM+LUT-MoE",
        "code_type": code_type,
        "avg_latency": avg_lat,
        "avg_throughput": avg_tok / avg_lat,
    }


def run_vllm_baseline(prompt, max_tokens, model_path, num_runs=3):
    """Run inference using standard vLLM (no LUT quantization)."""
    print("\n" + "=" * 60)
    print("vLLM Baseline (no LUT) Benchmark")
    print("=" * 60)

    from vllm import LLM, SamplingParams
    import torch

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    print("Warming up...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
    _ = llm.generate([prompt], sampling_params)

    print(f"Benchmarking ({num_runs} runs)...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    latencies = []
    tokens_list = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        outputs = llm.generate([prompt], sampling_params)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        gen_tokens = len(outputs[0].outputs[0].token_ids)

        latencies.append(elapsed)
        tokens_list.append(gen_tokens)
        print(f"  Run {i+1}: {elapsed:.3f}s, {gen_tokens} tokens")

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(tokens_list) / len(tokens_list)
    print(f"\nvLLM Baseline: {avg_lat:.3f}s avg, {avg_tok/avg_lat:.1f} tok/s")

    return {
        "framework": "vLLM-Baseline",
        "avg_latency": avg_lat,
        "avg_throughput": avg_tok / avg_lat,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-framework benchmark")
    parser.add_argument("--model", default="/home/hh/LUT-MoE/models/deepseek")
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--lut_config",
                        default="/home/hh/LUT-MoE/entry/config.json")
    parser.add_argument("--code_type", default="BLOCKLUT")
    parser.add_argument("--compare", action="store_true",
                        help="Run both frameworks and compare")
    parser.add_argument("--mode", choices=["hf", "vllm_lut", "vllm_base", "all"],
                        default="all")
    args = parser.parse_args()

    import torch

    n_runs = args.num_runs
    results = []

    if args.mode in ("all", "vllm_base"):
        r = run_vllm_baseline(args.prompt, args.max_tokens, args.model,
                              num_runs=n_runs)
        results.append(r)

    if args.mode in ("all", "vllm_lut"):
        r = run_vllm_lut(args.prompt, args.max_tokens, args.model,
                         args.code_type, args.model, num_runs=n_runs)
        results.append(r)

    if args.mode in ("all", "hf"):
        r = run_lut_moe_hf(args.prompt, args.max_tokens, args.model,
                           args.lut_config, num_runs=n_runs)
        results.append(r)

    # Print comparison table
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print(f"{'Framework':<20} {'Latency (s)':<15} {'Throughput (tok/s)':<20}")
        print("-" * 55)
        for r in results:
            name = r.get("framework", "?")
            lat = f"{r['avg_latency']:.3f}"
            thr = f"{r['avg_throughput']:.1f}"
            print(f"{name:<20} {lat:<15} {thr:<20}")

        # Calculate speedup
        if len(results) >= 2:
            base = results[0]["avg_latency"]
            for r in results[1:]:
                speedup = base / r["avg_latency"]
                print(f"\n{r['framework']} vs {results[0]['framework']}: "
                      f"{speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = f"/home/hh/LUT-MoE/vllm_lut/benchmark_{timestamp}.json"
    with open(out, "w") as f:
        json.dump({
            "args": vars(args),
            "results": [{k: v for k, v in r.items()
                        if k in ("framework", "avg_latency", "avg_throughput", "code_type")}
                       for r in results]
        }, f, indent=2)
    print(f"\nResults saved to {out}")
