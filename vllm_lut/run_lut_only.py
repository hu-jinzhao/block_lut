#!/usr/bin/env python3
"""
LUT quantization for vLLM (no SSD offloading).

Loads a model through vLLM, quantizes expert weights to LUT format,
then runs inference. Uses vLLM's fused MoE kernel with on-the-fly
decompression.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/hh/LUT-MoE/models/deepseek")
    parser.add_argument("--code_type", default="BLOCKLUT", choices=["BLOCKLUT", "NESTEDLUT", "LUT"])
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--baseline", action="store_true", help="Skip quantization")
    return parser.parse_args()


def decompress_batch(indices, absmax, codebook, block_size=128):
    """Decompress one expert's LUT weights to bf16."""
    device = indices.device
    codebook = codebook.to(device)
    absmax = absmax.to(device)
    normalized = codebook[indices.long()]
    block_id = torch.arange(indices.shape[0], device=device) // block_size
    if block_id.max() >= absmax.shape[0]:
        block_id = block_id.clamp(max=absmax.shape[0] - 1)
    scale = absmax[block_id]
    return normalized * scale


def quantize_experts(module, quantizer, codebook, block_size=128):
    """Recursively find and quantize MoE expert weights in a vLLM model."""
    quantized_layers = 0
    quantized_experts = 0

    def _quantize(module, path=""):
        nonlocal quantized_layers, quantized_experts

        # Check if this is a RoutedExperts layer (has w13_weight)
        if hasattr(module, 'w13_weight') and module.w13_weight is not None:
            w13 = module.w13_weight.data
            w2 = module.w2_weight.data
            n_exp = w13.shape[0]

            print(f"  Quantizing {path}: {n_exp} experts "
                  f"(w13={list(w13.shape)}, w2={list(w2.shape)})...")

            # Quantize each expert
            q13_list, q13_a_list = [], []
            q2_list, q2_a_list = [], []

            for i in range(n_exp):
                q13 = quantizer.quantize(w13[i], codebook)
                q2 = quantizer.quantize(w2[i], codebook)
                q13_list.append(q13["indices"])
                q13_a_list.append(q13["absmax"])
                q2_list.append(q2["indices"])
                q2_a_list.append(q2["absmax"])

            # Store LUT format
            module.register_buffer("_q_w13_idx", torch.stack(q13_list, dim=0))
            module.register_buffer("_q_w13_abs", torch.stack(q13_a_list, dim=0).to(torch.bfloat16))
            module.register_buffer("_q_w2_idx", torch.stack(q2_list, dim=0))
            module.register_buffer("_q_w2_abs", torch.stack(q2_a_list, dim=0).to(torch.bfloat16))
            module.register_buffer("_q_codebook", codebook.cpu())

            # Store original weights (we'll replace during forward)
            module._q_orig_w13 = w13.clone()
            module._q_orig_w2 = w2.clone()

            # Flag
            module._lut_quantized = True
            module._q_block_size = block_size
            quantized_layers += 1
            quantized_experts += n_exp

        for name, child in module.named_children():
            _quantize(child, f"{path}.{name}" if path else name)

    _quantize(module)
    return quantized_layers, quantized_experts


def enable_lut_forward_hooks(module):
    """Register forward hooks to decompress LUT->bf16 before MoE computation."""
    hooks = []

    def make_hook(name):
        def _hook(module, input):
            if not getattr(module, '_lut_quantized', False):
                return
            # Decompress ALL experts to bf16
            block_size = module._q_block_size
            codebook = module._q_codebook.to(module._q_w13_idx.device)

            n_exp = module._q_w13_idx.shape[0]
            for i in range(n_exp):
                w13 = decompress_batch(module._q_w13_idx[i], module._q_w13_abs[i],
                                       codebook, block_size)
                w2 = decompress_batch(module._q_w2_idx[i], module._q_w2_abs[i],
                                      codebook, block_size)

                # Calculate expected shapes
                # w13: [2*intermediate, hidden], w2: [hidden, intermediate]
                int_size = w13.shape[0] // 2
                hid_size = w13.shape[1]

                # Store decompressed weights back
                module._q_orig_w13[i] = w13.reshape(2 * int_size, hid_size)
                module._q_orig_w2[i] = w2.reshape(hid_size, int_size)

            # Temporarily replace weight data for the forward pass
            # The actual replacement is done in the weight references
            saved_w13 = module.w13_weight.data.clone()
            saved_w2 = module.w2_weight.data.clone()

            module.w13_weight.data = module._q_orig_w13
            module.w2_weight.data = module._q_orig_w2

            # Store originals for post-hook restoration
            module._q_saved_w13 = saved_w13
            module._q_saved_w2 = saved_w2

        def _post_hook(module, input, output):
            if not getattr(module, '_lut_quantized', False):
                return
            # Restore original LUT weights
            if hasattr(module, '_q_saved_w13'):
                module.w13_weight.data = module._q_saved_w13
                del module._q_saved_w13
            if hasattr(module, '_q_saved_w2'):
                module.w2_weight.data = module._q_saved_w2
                del module._q_saved_w2

        return _hook, _post_hook

    def _register(module, path=""):
        if hasattr(module, 'w13_weight') and getattr(module, '_lut_quantized', False):
            pre_hook, post_hook = make_hook(path)
            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            hooks.extend([h1, h2])
        for name, child in module.named_children():
            _register(child, f"{path}.{name}" if path else name)

    _register(module)
    return hooks


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams
    from vllm_lut.quantizer import LUTQuantizer
    import numpy as np

    # Load LUT codebook if available
    quantizer = LUTQuantizer(code_type=args.code_type, device="cuda")
    codebook_path = os.path.join(args.model)
    loaded = quantizer.load(codebook_path)

    # ===== Load model =====
    print(f"Loading model from {args.model}...")
    t0 = time.time()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # ===== Apply LUT quantization =====
    if not args.baseline:
        print(f"\nApplying {args.code_type} quantization...")

        # Train codebook if not loaded
        if not loaded or quantizer.lut_codebook is None:
            print("  Training codebook on expert weights...")
            all_w = []
            model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
            model = model_obj.model if hasattr(model_obj, 'model') else model_obj
            for layer in model.model.layers:
                if hasattr(layer.mlp, 'routed_experts'):
                    exp = layer.mlp.routed_experts
                    if hasattr(exp, 'w13_weight') and exp.w13_weight.numel() > 0:
                        all_w.append(exp.w13_weight.data.float().reshape(-1))
                        all_w.append(exp.w2_weight.data.float().reshape(-1))
            if all_w:
                quantizer.train(torch.cat(all_w))
                quantizer.save(codebook_path)

        codebook = quantizer.lut_codebook.to("cuda")

        # Find the model and quantize
        model_obj = llm.llm_engine.model_executor.driver_worker.model_runner.model
        model = model_obj.model if hasattr(model_obj, 'model') else model_obj

        layers, experts = quantize_experts(model, quantizer, codebook)
        print(f"  Quantized {layers} layers ({experts} total experts)")

        # Enable forward hooks to decompress on-the-fly
        hooks = enable_lut_forward_hooks(model)
        print(f"  Registered {len(hooks)} forward hooks")
    else:
        print("  Running baseline (no quantization)")

    # ===== Warm up =====
    print("\nWarming up...")
    sp = SamplingParams(temperature=0.7, max_tokens=20)
    _ = llm.generate([args.prompt], sp)

    # ===== Benchmark =====
    print(f"\nBenchmarking ({args.num_runs} runs, max_tokens={args.max_tokens})...")
    sp = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    latencies = []
    tokens_list = []

    for i in range(args.num_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([args.prompt], sp)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        gen = len(out[0].outputs[0].token_ids)
        latencies.append(elapsed)
        tokens_list.append(gen)
        print(f"  Run {i+1}: {elapsed:.3f}s, {gen} tokens, {gen/elapsed:.1f} tok/s")

    avg_lat = sum(latencies) / len(latencies)
    avg_tok = sum(tokens_list) / len(tokens_list)
    label = "Baseline" if args.baseline else f"{args.code_type}"
    print(f"\n[{label}] Avg: {avg_lat:.3f}s, {avg_tok/avg_lat:.1f} tok/s")


if __name__ == "__main__":
    main()
