"""Quick output quality test for pure 6-bit and pure 4-bit NESTEDLUT.

Tests: tier=0 (8-bit baseline), tier=1 (6-bit), tier=2 (4-bit)
Same prompt, compare generated text.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entry.llm_modeling import MoE
from utils.constants import (
    List_expert_topk, List_num_elements_per_expert, List_num_tensors_per_expert,
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)

CHECKPOINT = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OFFLOAD = "/home/hh/zip_Moe/LUT_MoE/offload/qwen_blocklut"
LUT_PATH = os.path.join(CHECKPOINT, "blocklut_256.npy")
PROMPT = "Hello!"


def test_tier(tier, code_type="NESTEDLUT"):
    from transformers import AutoTokenizer
    import torch

    config = {
        "offload_path": OFFLOAD,
        "caching_algorithm": "LFU", "prefetcher_topk": 4,
        "device_memory_ratio": 0.85, "gpu_pool_ratio": 0.95,
        "batch_size": 1,
        "code_type": code_type, "lut_path": LUT_PATH,
        "lut_tier": tier,
        "hyperparam_state_margin": 0.1,
        "num_file_chunks": 3, "num_compute_threads": 6,
        "trace_path": "/home/hh/zip_Moe/LUT_MoE/trace/qwen_trace.pt",
        "expert_topk": List_expert_topk["qwen"],
        "num_elements_per_expert": List_num_elements_per_expert["qwen"],
        "num_tensors_per_expert": List_num_tensors_per_expert["qwen"],
        "num_expert_layers": List_num_expert_layers["qwen"],
        "num_experts": List_num_experts["qwen"],
        "first_k_dense_replace": List_first_k_dense_replace["qwen"],
    }

    t0 = time.perf_counter()
    model = MoE(CHECKPOINT, config)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate
    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda:0")

    with torch.no_grad():
        t_start = time.perf_counter()
        output_ids = model.model.generate(
            input_ids, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_time = time.perf_counter() - t_start

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Input:  {PROMPT!r}")
    print(f"  Output: {output_text!r}")

    del model
    torch.cuda.empty_cache()
    return output_text


def main():
    print("=" * 60)
    print("TIER 0: Pure 8-bit (baseline)")
    print("=" * 60)
    out_8bit = test_tier(0)

    print("\n" + "=" * 60)
    print("TIER 1: Pure 6-bit (mapped64 K-means, ~50 dB)")
    print("=" * 60)
    out_6bit = test_tier(1)

    print("\n" + "=" * 60)
    print("TIER 2: Pure 4-bit (mapped16 K-means, ~38 dB)")
    print("=" * 60)
    out_4bit = test_tier(2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  8-bit: {out_8bit!r}")
    print(f"  6-bit: {out_6bit!r}")
    print(f"  4-bit: {out_4bit!r}")


if __name__ == "__main__":
    main()
