#!/usr/bin/env python3
"""Create BLOCKLUT offload directories for uniform6 and uniform4 LUT variants."""
import argparse, os, sys, time, gc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from evaluation.profile_tools import clear_model_cache
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/LUT-MoE/models/qwen"
OFFLOAD_BASE = "/home/hh/LUT-MoE/offload"


def create_offload(lut_name, offload_name):
    """Initialize MoE with specified LUT to trigger offload creation."""
    offload_path = os.path.join(OFFLOAD_BASE, offload_name)
    lut_path = os.path.join(CHECKPOINT, f"blocklut_{lut_name}.npy")

    print(f"\n{'='*60}")
    print(f"  Creating offload: {offload_name}")
    print(f"  LUT: {lut_path}")
    print(f"  Offload path: {offload_path}")
    print(f"{'='*60}")

    config = {
        "offload_path": offload_path,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95, "batch_size": 1,
        "code_type": "BLOCKLUT", "lut_path": lut_path,
        "hyperparam_state_margin": 0.1, "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/LUT-MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }

    t0 = time.perf_counter()
    model = MoE(CHECKPOINT, config)
    elapsed = time.perf_counter() - t0
    print(f"\nOffload creation completed in {elapsed:.1f}s")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    print(f"Cleaned up. Offload at: {offload_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=str, default="63:qwen_blocklut_uniform6,15:qwen_blocklut_uniform4",
                        help="Comma-separated lut_size:offload_name pairs")
    args = parser.parse_args()

    for variant in args.variants.split(","):
        lut_name, offload_name = variant.strip().split(":")
        create_offload(lut_name, offload_name)

    print("\nAll offload variants created.")


if __name__ == "__main__":
    main()
