#!/usr/bin/env python3
"""
Generate nested LUT subsets from 256 K-means centroids via greedy merging.

Usage: python evaluation/generate_nested_lut.py

Outputs:
  - blocklut_256.npy (if not exists)  — 256 K-means centroids
  - nested_lut_64.npy  — 64 nested subset centroids
  - nested_lut_16.npy  — 16 nested subset centroids
  - nested_map_256to64.npy — uint8[256] mapping table
  - nested_map_256to16.npy — uint8[256] mapping table
"""
import argparse
import os
import sys
import time
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
MAX_EXPERTS = 30
N_ASSIGN_SAMPLE = 200000  # samples for per-cluster weight estimation


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


def learn_256_lut(samples_list):
    """Train 256-centroid K-means LUT and return centroids + per-cluster weights."""
    all_data = np.concatenate([s.ravel() for s in samples_list])
    rng = np.random.RandomState(42)

    # Train K-means
    if len(all_data) > KMEANS_SAMPLE:
        idx = rng.choice(len(all_data), KMEANS_SAMPLE, replace=False)
        data = all_data[idx].reshape(-1, 1)
    else:
        data = all_data.reshape(-1, 1)

    km = MiniBatchKMeans(n_clusters=256, random_state=42,
                         batch_size=KMEANS_BATCH, n_init=1, max_iter=30)
    km.fit(data)
    centroids = km.cluster_centers_.ravel().astype(np.float32)
    sort_idx = np.argsort(centroids)
    centroids = centroids[sort_idx]

    # Assign samples to estimate per-cluster weights
    # Use a larger sample for stable weight estimation
    if len(all_data) > N_ASSIGN_SAMPLE:
        idx2 = rng.choice(len(all_data), N_ASSIGN_SAMPLE, replace=False)
        assign_data = all_data[idx2]
    else:
        assign_data = all_data

    # Batch assignment to avoid memory issues
    batch_size = 50000
    weights = np.zeros(256, dtype=np.int64)
    for start in range(0, len(assign_data), batch_size):
        batch = assign_data[start:start + batch_size]
        dists = np.abs(batch.reshape(-1, 1) - centroids.reshape(1, -1))
        labels = np.argmin(dists, axis=1)
        for lab in labels:
            weights[lab] += 1

    print(f"  Per-cluster sample counts: min={weights.min()}, max={weights.max()}, "
          f"median={np.median(weights):.0f}")
    return centroids, weights


def greedy_merge(centroids, weights, target_k):
    """Greedy 1D agglomerative clustering: merge adjacent pairs minimizing delta_MSE.

    Uses optimal 1D DP-like greedy: always merge the adjacent pair with
    lowest merge_cost = w_i*w_j*(c_i - c_j)^2 / (w_i + w_j).
    """
    import heapq

    K = len(centroids)
    if target_k >= K:
        return centroids, weights, np.arange(K, dtype=np.uint8)

    c = centroids.astype(np.float64).copy()
    w = weights.astype(np.float64).copy()

    # Doubly-linked list for active clusters
    prev = np.arange(K) - 1  # prev[i] = i-1, -1 for head
    next_arr = np.arange(K) + 1  # next[i] = i+1, K for tail
    active = np.ones(K, dtype=bool)

    # Heap of (merge_cost, i, j) where i < j are adjacent active clusters
    heap = []
    for i in range(K - 1):
        cost = w[i] * w[i + 1] * (c[i] - c[i + 1]) ** 2 / (w[i] + w[i + 1])
        heapq.heappush(heap, (cost, i, i + 1))

    merges = K - target_k
    for _ in range(merges):
        # Pop the cheapest valid merge
        while heap:
            cost, i, j = heapq.heappop(heap)
            if active[i] and active[j] and next_arr[i] == j:
                break
        else:
            break  # no valid merge (shouldn't happen)

        # Merge j into i
        new_c = (w[i] * c[i] + w[j] * c[j]) / (w[i] + w[j])
        new_w = w[i] + w[j]
        c[i] = new_c
        w[i] = new_w
        active[j] = False

        # Update linked list
        nxt = next_arr[j]
        next_arr[i] = nxt
        if nxt < K:
            prev[nxt] = i

        # Add new candidate merges
        # (prev[i], i)
        pi = prev[i]
        if pi >= 0 and active[pi]:
            new_cost = w[pi] * w[i] * (c[pi] - c[i]) ** 2 / (w[pi] + w[i])
            heapq.heappush(heap, (new_cost, pi, i))
        # (i, next[i])
        ni = next_arr[i]
        if ni < K and active[ni]:
            new_cost = w[i] * w[ni] * (c[i] - c[ni]) ** 2 / (w[i] + w[ni])
            heapq.heappush(heap, (new_cost, i, ni))

    # Extract surviving centroids and build mapping
    survivor_indices = np.where(active)[0]
    nested_c = c[survivor_indices].astype(np.float32)

    # Build mapping: original index -> survivor index
    # For merged clusters, map to the surviving centroid they were merged into
    mapping = np.zeros(K, dtype=np.uint8)
    # Trace merges backward: each surviving index represents a range
    # Simpler approach: for each original index, find nearest surviving centroid
    for i in range(K):
        # Walk right to find the surviving cluster that absorbed i
        cur = i
        while cur < K and not active[cur]:
            cur += 1
        if cur >= K:
            # Walk left as fallback
            cur = i
            while cur >= 0 and not active[cur]:
                cur -= 1
        # Map to the index in the nested array
        mapping[i] = np.searchsorted(survivor_indices, cur)

    return nested_c, w[survivor_indices].astype(np.int64), mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=MODEL_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    safetensor_files = sorted(
        f for f in os.listdir(args.model_dir) if f.endswith(".safetensors")
    )

    # ── Step 1: Collect block-normalized samples ──
    print("Collecting samples from expert weights...")
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
                samples.append(block_normalize(t))
                expert_count += 1
                del t

    print(f"Collected {expert_count} experts, "
          f"{sum(s.size for s in samples):,} total block-normalized values")

    # ── Step 2: Learn 256-centroid K-means + per-cluster weights ──
    lut256_path = os.path.join(output_dir, "blocklut_256.npy")
    if os.path.exists(lut256_path):
        print(f"Loading existing 256 LUT from {lut256_path}")
        centroids_256 = np.sort(np.load(lut256_path).astype(np.float32))
        # Re-compute weights by assigning samples to nearest centroid
        all_data = np.concatenate([s.ravel() for s in samples])
        rng = np.random.RandomState(42)
        if len(all_data) > N_ASSIGN_SAMPLE:
            idx = rng.choice(len(all_data), N_ASSIGN_SAMPLE, replace=False)
            assign_data = all_data[idx]
        else:
            assign_data = all_data

        weights_256 = np.zeros(256, dtype=np.int64)
        batch_size = 50000
        for start in range(0, len(assign_data), batch_size):
            batch = assign_data[start:start + batch_size]
            dists = np.abs(batch.reshape(-1, 1) - centroids_256.reshape(1, -1))
            labels = np.argmin(dists, axis=1)
            for lab in labels:
                weights_256[lab] += 1
        print(f"  Weights recomputed: min={weights_256.min()}, max={weights_256.max()}")
    else:
        print("Training 256-centroid K-means LUT...")
        t0 = time.perf_counter()
        centroids_256, weights_256 = learn_256_lut(samples)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
        np.save(lut256_path, centroids_256)
        print(f"  Saved to {lut256_path}")

    # ── Step 3: Greedy merge to 64 and 16 ──
    print("\nGreedy merging 256 -> 64...")
    t0 = time.perf_counter()
    nested_64, w64, map_256to64 = greedy_merge(centroids_256, weights_256, 64)
    print(f"  Done in {time.perf_counter() - t0:.1f}s, {len(nested_64)} centroids")

    print("Greedy merging 256 -> 16...")
    t0 = time.perf_counter()
    nested_16, w16, map_256to16 = greedy_merge(centroids_256, weights_256, 16)
    print(f"  Done in {time.perf_counter() - t0:.1f}s, {len(nested_16)} centroids")

    # Also generate 256->256 identity mapping for completeness
    map_256to256 = np.arange(256, dtype=np.uint8)

    # ── Step 4: Save nested LUTs and mapping tables ──
    np.save(os.path.join(output_dir, "nested_lut_64.npy"), nested_64)
    np.save(os.path.join(output_dir, "nested_lut_16.npy"), nested_16)
    np.save(os.path.join(output_dir, "nested_map_256to64.npy"), map_256to64)
    np.save(os.path.join(output_dir, "nested_map_256to16.npy"), map_256to16)
    np.save(os.path.join(output_dir, "nested_map_256to256.npy"), map_256to256)

    print(f"\nSaved to {output_dir}:")
    print(f"  nested_lut_64.npy  — {len(nested_64)} centroids")
    print(f"  nested_lut_16.npy  — {len(nested_16)} centroids")
    print(f"  nested_map_256to64.npy — {len(map_256to64)} entries, "
          f"unique={len(np.unique(map_256to64))}")
    print(f"  nested_map_256to16.npy — {len(map_256to16)} entries, "
          f"unique={len(np.unique(map_256to16))}")

    # ── Step 5: Quick PSNR validation ──
    print("\n--- Quick PSNR validation ---")
    all_data = np.concatenate([s.ravel() for s in samples])
    rng = np.random.RandomState(123)
    idx = rng.choice(len(all_data), min(50000, len(all_data)), replace=False)
    test_data = all_data[idx]

    def compute_psnr(original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(1.0 / mse))

    # Nest 64
    idx64 = np.searchsorted(
        (centroids_256[:-1] + centroids_256[1:]) / 2, test_data
    ).astype(np.uint8)
    mapped64 = map_256to64[idx64]
    recon64 = nested_64[mapped64]
    psnr_64_direct = compute_psnr(test_data, recon64)
    print(f"  Nested 64 PSNR: {psnr_64_direct:.2f} dB")

    # Nest 16
    mapped16 = map_256to16[idx64]
    recon16 = nested_16[mapped16]
    psnr_16_direct = compute_psnr(test_data, recon16)
    print(f"  Nested 16 PSNR: {psnr_16_direct:.2f} dB")

    # Full 256
    recon256 = centroids_256[idx64]
    psnr_256 = compute_psnr(test_data, recon256)
    print(f"  Full 256 PSNR:  {psnr_256:.2f} dB")

    print("\nDone.")


if __name__ == "__main__":
    main()
