"""Phase 1: Collect MoE layer inputs from calibration data.

Hooks into each MoE layer to capture the input hidden states (before routing).
Saves pooled inputs per layer to disk for offline analysis.

Memory-efficient: processes one token at a time, pools periodically.
"""
import os, sys, time, json
import numpy as np
import torch
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OUTPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/expert_behavior"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("Loading model (CPU offload)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="/tmp/offload",
        max_memory={0: "6GB", "cpu": "32GB"},
    )
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Count MoE layers
    n_moe_layers = 0
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            n_moe_layers += 1
    print(f"MoE layers: {n_moe_layers}")

    # Storage: layer -> list of input vectors (as numpy)
    collected = {i: [] for i in range(n_moe_layers)}
    max_samples_per_layer = 5000  # cap to manage memory

    # Register hooks
    hooks = []

    def make_hook(moe_idx):
        def hook_fn(module, input, output):
            if len(collected[moe_idx]) >= max_samples_per_layer:
                return
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is None:
                return
            # x shape: (batch, seq, hidden) or (num_tokens, hidden)
            x_flat = x.detach().reshape(-1, x.shape[-1]).cpu().to(torch.float32)
            for i in range(x_flat.shape[0]):
                if len(collected[moe_idx]) >= max_samples_per_layer:
                    break
                collected[moe_idx].append(x_flat[i].numpy().copy())
        return hook_fn

    moe_idx = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'experts'):
            h = layer.mlp.register_forward_hook(make_hook(moe_idx))
            hooks.append(h)
            print(f"  Registered hook: layer {layer_idx} → MoE idx {moe_idx}")
            moe_idx += 1

    # Calibration prompts
    calibration_texts = [
        "The capital of France is",
        "Machine learning is a subset of artificial intelligence that",
        "The first law of thermodynamics states that energy",
        "In mathematics, the Pythagorean theorem describes the relationship",
        "The Python programming language was created by",
        "Deep learning models require large amounts of",
        "The transformer architecture introduced the concept of",
        "Quantum computing differs from classical computing in that",
        "The theory of evolution by natural selection was proposed by",
        "In computer science, a binary search tree supports",
    ]

    print(f"\nRunning calibration ({len(calibration_texts)} prompts)...")
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    for text in tqdm(calibration_texts, desc="Calibration"):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                _ = model(**inputs, max_new_tokens=8)  # generate a few tokens to get more activations
            except Exception as e:
                print(f"  Warning: {e}")
                # Try single forward pass
                try:
                    _ = model(**inputs)
                except Exception as e2:
                    print(f"  Skipping prompt: {e2}")
                    continue

    # Remove hooks
    for h in hooks:
        h.remove()

    # Report collection stats
    print(f"\nCollection summary:")
    for moe_idx in range(n_moe_layers):
        n = len(collected[moe_idx])
        if n > 0:
            arr = np.stack(collected[moe_idx])
            print(f"  MoE layer {moe_idx}: {n} samples, shape={arr.shape}, "
                  f"norm_mean={np.linalg.norm(arr, axis=1).mean():.2f}")

    # Save to disk
    for moe_idx in range(n_moe_layers):
        if collected[moe_idx]:
            arr = np.stack(collected[moe_idx])
            path = os.path.join(OUTPUT_DIR, f"inputs_layer_{moe_idx:02d}.npy")
            np.save(path, arr.astype(np.float16))
            print(f"  Saved: {path} ({arr.shape})")

    # Also save per-expert weight norms for reference
    print(f"\nSaving expert weight norms...")
    expert_info = {}
    moe_idx = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'experts'):
            e_info = {}
            for e_idx, expert in enumerate(layer.mlp.experts):
                w_norms = {}
                for ptype in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(expert, ptype):
                        w = getattr(expert, ptype).weight.detach().cpu().to(torch.float32)
                        w_norms[ptype] = float(torch.norm(w).item())
                e_info[str(e_idx)] = w_norms
            expert_info[str(moe_idx)] = e_info
            moe_idx += 1

    with open(os.path.join(OUTPUT_DIR, "expert_info.json"), "w") as f:
        json.dump(expert_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
