#!/usr/bin/env python3
"""Learn shared LUTs for different bit-widths and save to .npy files."""
import argparse, os, sys, time
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import (
    List_num_expert_layers, List_num_experts, List_first_k_dense_replace,
)
from utils.hf_config import parse_expert_id
from safetensors import safe_open
from transformers import AutoConfig

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
BLOCK_SIZE = 128
KMEANS_SAMPLE = 50000
KMEANS_BATCH = 2048
MAX_EXPERTS = 30  # experts to sample for LUT training


def block_normalize(tensor_f32):
    flat = tensor_f32.ravel()
    n = flat.numel()
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded = n_blocks * BLOCK_SIZE
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n)])
    blocks = flat.reshape(n_blocks, BLOCK_SIZE)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normalized = (blocks / absmax.unsqueeze(1)).ravel().numpy().astype(np.float32)
    return normalized


def learn_lut(samples_list, n_centroids):
    all_data = np.concatenate([s.ravel() for s in samples_list])
    if len(all_data) > KMEANS_SAMPLE:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_data), KMEANS_SAMPLE, replace=False)
        data = all_data[idx].reshape(-1, 1)
    else:
        data = all_data.reshape(-1, 1)

    km = MiniBatchKMeans(n_clusters=n_centroids, random_state=42,
                         batch_size=KMEANS_BATCH, n_init=1, max_iter=30)
    km.fit(data)
    centroids = km.cluster_centers_.ravel().astype(np.float32)
    centroids.sort()
    return centroids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--luts", type=str, default="63,15",
                        help="Comma-separated centroid counts")
    parser.add_argument("--output_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    centroid_counts = [int(c) for c in args.luts.split(",")]
    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    first_k_dense = List_first_k_dense_replace["qwen"]

    safetensor_files = sorted(
        f for f in os.listdir(args.model_dir) if f.endswith(".safetensors")
    )

    # Collect block-normalized samples from expert weights
    print(f"Collecting samples from up to {MAX_EXPERTS} experts...")
    samples = []
    expert_count = 0

    for sf in safetensor_files:
        if expert_count >= MAX_EXPERTS:
            break
        sf_path = os.path.join(args.model_dir, sf)
        with safe_open(sf_path, framework="pt", device="cpu") as fhandle:
            for k in fhandle.keys():
                if expert_count >= MAX_EXPERTS:
                    break
                if "expert" not in k or "shared_expert" in k:
                    continue
                model_layer, expert_id = parse_expert_id(k, model_config)
                if model_layer is None or expert_id is None:
                    continue
                t = fhandle.get_tensor(k).to(torch.float32)
                norm_np = block_normalize(t)
                samples.append(norm_np)
                expert_count += 1
                del t

    print(f"Collected {expert_count} experts, "
          f"{sum(s.size for s in samples):,} total values")

    for n_cent in centroid_counts:
        t0 = time.perf_counter()
        lut = learn_lut(samples, n_cent)
        elapsed = time.perf_counter() - t0

        out_path = os.path.join(args.output_dir, f"blocklut_{n_cent}.npy")
        np.save(out_path, lut)
        eff_bits = np.log2(n_cent + 1) + 16.0 / BLOCK_SIZE
        print(f"LUT-{n_cent} saved to {out_path} "
              f"({len(lut)} centroids, ~{eff_bits:.1f} bits, {elapsed:.1f}s)")

    print("Done.")


if __name__ == "__main__":
    main()
