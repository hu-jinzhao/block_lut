#!/usr/bin/env python3
"""
Full LUT-MoE vLLM benchmark script.

Loads a DeepSeek-V2-Lite model through vLLM with LUT-quantized experts,
runs inference, and measures performance.

Usage:
    # With LUT quantization:
    python3 -m vllm_lut.run_lut_vllm --model /path/to/model --code_type BLOCKLUT

    # Without LUT quantization (baseline):
    python3 -m vllm_lut.run_lut_vllm --model /path/to/model --baseline
"""

import argparse
import json
import os
import sys
import time

# Add CUDA runtime to path
_cuda_paths = [
    "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib",
]
for _p in _cuda_paths:
    if os.path.isdir(_p) and _p not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = _p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="LUT-MoE vLLM Benchmark")
    parser.add_argument("--model", type=str, default="/home/hh/LUT-MoE/models/deepseek",
                        help="Model path or name")
    parser.add_argument("--code_type", type=str, default="BLOCKLUT",
                        choices=["BLOCKLUT", "NESTEDLUT", "RAW"],
                        help="Quantization type")
    parser.add_argument("--baseline", action="store_true",
                        help="Run without LUT quantization (baseline)")
    parser.add_argument("--lut_path", type=str, default="/home/hh/LUT-MoE/models/deepseek",
                        help="Path to LUT codebook .npy files")
    parser.add_argument("--prompt", type=str, default="The future of AI is",
                        help="Test prompt")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of benchmark runs")
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed profiling")
    return parser.parse_args()


def load_lut_codebooks(lut_path: str):
    """Load LUT codebooks from .npy files."""
    import numpy as np
    codebooks = {}

    # Load BlockLUT (256-entry)
    for fname, key in [("blocklut_256.npy", "blocklut"),
                        ("full256.npy", "blocklut")]:
        fp = os.path.join(lut_path, fname)
        if os.path.exists(fp):
            codebooks[key] = torch.from_numpy(np.load(fp)).to(torch.bfloat16)
            print(f"  Loaded {fname} -> {key}: {codebooks[key].shape}")
            break

    # Load NestedLUT codebooks
    for fname, key in [("nested_lut_mapped64.npy", "mapped64"),
                        ("nested_lut_mapped16.npy", "mapped16")]:
        fp = os.path.join(lut_path, fname)
        if os.path.exists(fp):
            codebooks[key] = torch.from_numpy(np.load(fp)).to(torch.bfloat16)
            print(f"  Loaded {fname} -> {key}: {codebooks[key].shape}")

    return codebooks


def quantize_and_save_lut_weights(model_path, lut_path, output_path):
    """
    Pre-quantize all expert weights to LUT format and save.
    This produces a vLLM-compatible weight file with quantized experts.
    """
    print("\n=== Pre-quantizing expert weights to LUT format ===")

    from vllm_lut.quantizer import LUTQuantizer
    from safetensors.torch import load_file, save_file

    # Load the checkpoint
    import glob
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not safetensors_files:
        print(f"No safetensors files found in {model_path}")
        return False

    print(f"Found {len(safetensors_files)} checkpoint files")

    # Load codebooks
    codebooks = load_lut_codebooks(lut_path)
    if "blocklut" not in codebooks:
        print("No LUT codebook found, will train on-the-fly")
        need_train = True
    else:
        need_train = False
        codebook = codebooks["blocklut"]

    quantizer = LUTQuantizer(code_type="BLOCKLUT", device="cuda")
    if not need_train:
        quantizer.lut_codebook = codebook

    # Process each checkpoint
    new_files = []
    for ckpt_path in safetensors_files:
        print(f"\nProcessing: {os.path.basename(ckpt_path)}")
        state_dict = load_file(ckpt_path, device="cuda")

        # Identify and quantize expert weights
        expert_keys = [k for k in state_dict.keys()
                      if "expert" in k and "shared_expert" not in k]
        print(f"  Found {len(expert_keys)} expert weight tensors")

        if need_train:
            # Collect all expert weights for codebook training
            print("  Collecting weights for codebook training...")
            all_weights = []
            for k in expert_keys:
                all_weights.append(state_dict[k].float().reshape(-1))
            all_weights_cat = torch.cat(all_weights)

            print(f"  Training {quantizer.lut_size}-entry codebook on "
                  f"{all_weights_cat.numel()/1e6:.1f}M elements...")
            quantizer.train(all_weights_cat)
            codebook = quantizer.lut_codebook

            # Save the trained codebook
            os.makedirs(lut_path, exist_ok=True)
            import numpy as np
            np.save(os.path.join(lut_path, "blocklut_256.npy"),
                    codebook.cpu().numpy())
            print(f"  Codebook saved to {lut_path}/blocklut_256.npy")
            need_train = False

        # Quantize each expert weight and store as new tensors
        quantized_count = 0
        for k in expert_keys:
            w = state_dict[k]  # [rows, cols] bf16
            q = quantizer.quantize(w, codebook)

            # Store quantized version back (replace original)
            # For now, we keep original weights and add quantized as metadata
            # The forward pass will decompress on-the-fly
            state_dict[k + ".q_indices"] = q["indices"].cpu()
            state_dict[k + ".q_absmax"] = q["absmax"].cpu()
            quantized_count += 1

        print(f"  Quantized {quantized_count} tensors")

        # Save modified checkpoint
        out_name = os.path.basename(ckpt_path).replace(".safetensors", "_lut.safetensors")
        out_path = os.path.join(output_path, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_file(state_dict, out_path)
        new_files.append(out_name)
        print(f"  Saved: {out_path}")

    # Create index file for the new quantized model
    if safetensors_files:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            # Create new index
            new_index = {"metadata": index.get("metadata", {}),
                        "weight_map": {}}
            for k, v in index["weight_map"].items():
                new_fname = v.replace(".safetensors", "_lut.safetensors")
                new_index["weight_map"][k] = new_fname
            with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
                json.dump(new_index, f, indent=2)
            print(f"\nCreated index file for quantized model")

    import numpy as np
    # Save codebook for runtime use
    np.save(os.path.join(output_path, "blocklut_256.npy"),
            codebook.cpu().numpy())
    print(f"Codebook saved to {output_path}/blocklut_256.npy")

    return True


def apply_lut_quantization_to_model(model, quantizer, codebook, lut_config):
    """
    Apply LUT quantization to a loaded vLLM model.

    Model structure (DeepSeekV2):
        model.layers[i].mlp.experts  -> FusedMoE
            .routed_experts -> RoutedExperts
                .w13_weight: [num_experts, 2*intermediate, hidden]
                .w2_weight: [num_experts, hidden, intermediate]
    """
    import torch.nn as nn
    from collections import defaultdict

    quant_stats = defaultdict(int)

    def _quantize_module(module, path=""):
        """Recursively find MoE layers and quantize expert weights."""
        # Check if this module has routed_experts (MoE layer)
        if hasattr(module, 'routed_experts'):
            experts = module.routed_experts
            if hasattr(experts, 'w13_weight') and hasattr(experts, 'w2_weight'):
                w13 = experts.w13_weight.data
                w2 = experts.w2_weight.data

                if w13.numel() == 0:
                    return

                print(f"  Quantizing {path}.routed_experts: "
                      f"w13={list(w13.shape)}, w2={list(w2.shape)}")

                # Quantize each expert
                n_experts = w13.shape[0]
                for i in range(n_experts):
                    q13 = quantizer.quantize(w13[i], codebook)
                    q2 = quantizer.quantize(w2[i], codebook)

                    # Store quantized versions as buffers
                    prefix = f"{path}.routed_experts"

                    # Use the module's register_buffer to store quantized data
                    # This avoids nn.Parameter overhead
                    experts.register_buffer(
                        f"_q_w13_indices_{i}", q13["indices"])
                    experts.register_buffer(
                        f"_q_w13_absmax_{i}", q13["absmax"])
                    experts.register_buffer(
                        f"_q_w2_indices_{i}", q2["indices"])
                    experts.register_buffer(
                        f"_q_w2_absmax_{i}", q2["absmax"])

                # Store codebook and config as buffers
                experts.register_buffer("_q_codebook", codebook.cpu())
                experts.register_buffer("_q_absmax_placeholder",
                                       torch.zeros(1, dtype=torch.bfloat16))

                # Flag this layer as quantized
                experts._lut_quantized = True
                quant_stats['experts'] += n_experts
                quant_stats['layers'] += 1

        # Recurse into children
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            _quantize_module(child, child_path)

    print("\n=== Applying LUT Quantization to Model ===")
    _quantize_module(model)
    print(f"Quantized {quant_stats['layers']} MoE layers "
          f"({quant_stats['experts']} experts total)")
    return quant_stats


def run_vllm_benchmark(args):
    """Run vLLM inference benchmark with optional LUT quantization."""
    print("=" * 60)
    print("LUT-MoE vLLM Benchmark")
    print("=" * 60)

    # Model path
    model_path = args.model
    if not os.path.isdir(model_path):
        print(f"Model path not found: {model_path}")
        print("Using HuggingFace model name instead (will download)...")

    print(f"\nModel: {model_path}")
    print(f"Code type: {args.code_type}")
    print(f"Baseline mode: {args.baseline}")

    # ---- Step 1: Load vLLM model ----
    print("\n=== Step 1: Loading model with vLLM ===")

    from vllm import LLM, SamplingParams

    # GPU memory config (leave headroom)
    max_model_len = 4096
    gpu_memory_utilization = 0.85

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(f"Model loaded successfully")

    # ---- Step 2: Apply LUT quantization (if enabled) ----
    if not args.baseline and args.code_type != "RAW":
        print("\n=== Step 2: Applying LUT Quantization ===")

        # Load or train codebook
        codebooks = load_lut_codebooks(args.lut_path)

        quantizer = LUTQuantizer(code_type=args.code_type, device="cuda")

        if "blocklut" in codebooks:
            quantizer.lut_codebook = codebooks["blocklut"]
        else:
            print("No codebook found, extracting weights from model...")
            # Get all expert weights from the model for training
            all_weights = []
            for layer in llm.llm_engine.model_executor.driver_worker.model_runner.model.\
                         model.layers:
                if hasattr(layer.mlp, 'routed_experts'):
                    experts = layer.mlp.routed_experts
                    if hasattr(experts, 'w13_weight'):
                        all_weights.append(experts.w13_weight.data.float().reshape(-1))
                        all_weights.append(experts.w2_weight.data.float().reshape(-1))

            if all_weights:
                all_cat = torch.cat(all_weights)
                print(f"Training codebook on {all_cat.numel()/1e6:.1f}M elements...")
                quantizer.train(all_cat)
                os.makedirs(args.lut_path, exist_ok=True)
                import numpy as np
                np.save(os.path.join(args.lut_path, "blocklut_256.npy"),
                        quantizer.lut_codebook.cpu().numpy())

        codebook = quantizer.lut_codebook.to("cuda")

        # Apply quantization to model
        model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
        stats = apply_lut_quantization_to_model(
            model_obj.model if hasattr(model_obj, 'model') else model_obj,
            quantizer, codebook, args
        )
        print(f"LUT quantization applied: {stats}")

    # ---- Step 3: Warm up ----
    print("\n=== Step 3: Warm up ===")
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=20,
    )
    _ = llm.generate([args.prompt], sampling_params)
    print("Warmup done")

    # ---- Step 4: Benchmark ----
    print(f"\n=== Step 4: Benchmark ({args.num_runs} runs) ===")
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_tokens,
    )

    latencies = []
    tokens_per_run = []

    for i in range(args.num_runs):
        torch.cuda.synchronize()
        start = time.time()

        outputs = llm.generate([args.prompt], sampling_params)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        generated_tokens = len(outputs[0].outputs[0].token_ids)
        latencies.append(elapsed)
        tokens_per_run.append(generated_tokens)

        print(f"  Run {i+1}: {elapsed:.3f}s, {generated_tokens} tokens, "
              f"{generated_tokens/elapsed:.1f} tok/s")

    # Stats
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_per_run) / len(tokens_per_run)
    avg_throughput = avg_tokens / avg_latency

    print(f"\n=== Results ===")
    print(f"  Average latency: {avg_latency:.3f}s")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Average throughput: {avg_throughput:.1f} tok/s")
    print(f"  Configuration: code_type={args.code_type}, "
          f"baseline={args.baseline}")

    return {
        "model": args.model,
        "code_type": args.code_type,
        "baseline": args.baseline,
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens,
        "avg_throughput": avg_throughput,
        "latencies": latencies,
        "tokens_per_run": tokens_per_run,
    }


if __name__ == "__main__":
    args = parse_args()
    result = run_vllm_benchmark(args)

    # Save results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"/home/hh/LUT-MoE/vllm_lut/benchmark_results_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "latencies"}, f, indent=2)
    print(f"\nResults saved to {result_file}")
