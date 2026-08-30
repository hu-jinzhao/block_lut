"""Phase 2: Analyze expert output behavior clustering.

Loads collected MoE inputs and expert weights, then:
1. Computes per-expert outputs on shared inputs
2. Measures pairwise output cosine similarity
3. Clusters experts by functional behavior
4. Learns prototype weights and evaluates compression potential
"""
import os, sys, time, math, json
import numpy as np
import torch
from collections import defaultdict
from safetensors import safe_open
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
INPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/expert_behavior"


def build_weight_index(safetensor_files):
    """Build (layer, expert_idx, proj_type) -> (file_path, key) index."""
    index = defaultdict(dict)  # layer -> expert -> proj_type -> (file, key)
    for fp in safetensor_files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                parts = k.split(".")
                layer = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6]
                if expert_idx not in index[layer]:
                    index[layer][expert_idx] = {}
                index[layer][expert_idx][proj_type] = (fp, k)
    return index


def load_weight(fp, key):
    with safe_open(fp, framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(torch.float32)


def compute_expert_outputs(X, W):
    """X: (N, d_in), W: (d_out, d_in) → Y: (N, d_out)"""
    return X @ W.T


def main():
    t_start = time.perf_counter()

    safetensor_files = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")
    )
    print("Building weight index...")
    w_index = build_weight_index(safetensor_files)

    layers = sorted(w_index.keys())
    n_experts = max(w_index[layers[0]].keys()) + 1
    print(f"Layers: {len(layers)}, Experts: {n_experts}")

    # ========================================================================
    # 1. Per-layer: compute expert output similarity on shared inputs
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. EXPERT OUTPUT COSINE SIMILARITY (same inputs, per layer)")
    print("=" * 80)

    # For efficiency, sample 5 representative layers
    sample_layers = layers[::5]  # [0, 5, 10, 15, 20]
    # But also include last layer
    if layers[-1] not in sample_layers:
        sample_layers.append(layers[-1])

    ptype = "gate_proj"  # focus on one projection type

    layer_output_sim_stats = {}

    for layer in sample_layers:
        # Load collected inputs
        input_path = os.path.join(INPUT_DIR, f"inputs_layer_{layer:02d}.npy")
        if not os.path.exists(input_path):
            print(f"  Layer {layer}: no inputs found, skipping")
            continue
        X = torch.from_numpy(np.load(input_path).astype(np.float32))
        print(f"\n  Layer {layer}: X shape={X.shape}")

        # Compute output for all experts
        expert_outputs = {}  # expert_idx -> output tensor (N, d_out)
        for e in range(n_experts):
            if e in w_index[layer] and ptype in w_index[layer][e]:
                fp, k = w_index[layer][e][ptype]
                W = load_weight(fp, k)
                Y = compute_expert_outputs(X, W)
                expert_outputs[e] = Y

        n_found = len(expert_outputs)
        if n_found < 2:
            print(f"    Only {n_found} experts found, skipping")
            continue
        print(f"    {n_found} experts loaded")

        # Pairwise output cosine similarity
        # For efficiency, compute via normalized stack
        Y_stack = torch.stack([expert_outputs[e] for e in sorted(expert_outputs.keys())])
        # Y_stack: (E, N, d_out) → reshape to (E, N*d_out) for cosine
        E = Y_stack.shape[0]
        Y_flat = Y_stack.reshape(E, -1)  # (E, N * d_out)
        norms = torch.norm(Y_flat, dim=1, keepdim=True)
        Y_norm = Y_flat / (norms + 1e-12)
        sim = Y_norm @ Y_norm.T  # (E, E)

        off_diag_mask = ~torch.eye(E, dtype=torch.bool)
        off_diag = sim[off_diag_mask]

        print(f"    Pairwise output cosine similarity:")
        print(f"      mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
        print(f"      min={off_diag.min():.4f}, max={off_diag.max():.4f}")
        print(f"      >0.3: {(off_diag>0.3).float().mean()*100:.1f}%")
        print(f"      >0.5: {(off_diag>0.5).float().mean()*100:.1f}%")
        print(f"      >0.7: {(off_diag>0.7).float().mean()*100:.1f}%")
        print(f"      >0.9: {(off_diag>0.9).float().mean()*100:.1f}%")

        layer_output_sim_stats[layer] = {
            "mean": float(off_diag.mean()),
            "std": float(off_diag.std()),
            "min": float(off_diag.min()),
            "max": float(off_diag.max()),
            "pct_gt_0_3": float((off_diag > 0.3).float().mean() * 100),
            "pct_gt_0_5": float((off_diag > 0.5).float().mean() * 100),
            "pct_gt_0_9": float((off_diag > 0.9).float().mean() * 100),
        }

        # Save the full similarity matrix for one layer (for visualization)
        if layer == sample_layers[0]:
            np.save(os.path.join(INPUT_DIR, f"output_sim_layer_{layer:02d}.npy"),
                    sim.numpy())

    # ========================================================================
    # 2. K-means clustering of expert outputs
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. K-MEANS CLUSTERING OF EXPERT OUTPUTS (use 1st sample layer)")
    print("=" * 80)

    first_layer = sample_layers[0]
    input_path = os.path.join(INPUT_DIR, f"inputs_layer_{first_layer:02d}.npy")
    X = torch.from_numpy(np.load(input_path).astype(np.float32))

    # Collect normalized output vectors for clustering
    # Use output mean (averaged over samples) as expert "signature"
    expert_signatures = []
    for e in range(n_experts):
        if e in w_index[first_layer] and ptype in w_index[first_layer][e]:
            fp, k = w_index[first_layer][e][ptype]
            W = load_weight(fp, k)
            Y = compute_expert_outputs(X, W)  # (N, d_out)
            sig = Y.mean(dim=0)  # (d_out,)
            sig = sig / (torch.norm(sig) + 1e-12)
            expert_signatures.append(sig.numpy())

    expert_signatures = np.array(expert_signatures)  # (E, d_out)
    print(f"  Expert signatures: {expert_signatures.shape}")

    # Test different K
    results_k = {}
    for k in [2, 3, 5, 8, 10, 15, 20, 30]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(expert_signatures)
        inertia = km.inertia_
        # Mean intra-cluster distance
        intra_dists = []
        for c in range(k):
            cluster_members = expert_signatures[labels == c]
            if len(cluster_members) > 1:
                centroid = km.cluster_centers_[c]
                dists = np.linalg.norm(cluster_members - centroid, axis=1)
                intra_dists.extend(dists)
        mean_intra = np.mean(intra_dists) if intra_dists else 0
        results_k[k] = {"inertia": float(inertia), "mean_intra_dist": float(mean_intra)}
        cluster_sizes = [int(np.sum(labels == c)) for c in range(k)]
        print(f"    K={k:>3}: inertia={inertia:.2f}, intra_dist={mean_intra:.4f}, "
              f"cluster sizes={cluster_sizes}")

    # ========================================================================
    # 3. Prototype learning: what happens if we replace experts with cluster prototypes?
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. PROTOTYPE LEARNING — QUANTITATIVE COMPRESSION EVALUATION")
    print("=" * 80)

    # For each layer, for each projection type, test prototype-based compression
    test_layers = sample_layers[:3]  # first 3 sample layers
    all_psnr_results = []

    for layer in test_layers:
        input_path = os.path.join(INPUT_DIR, f"inputs_layer_{layer:02d}.npy")
        if not os.path.exists(input_path):
            continue
        X = torch.from_numpy(np.load(input_path).astype(np.float32))
        print(f"\n  Layer {layer}:")

        for ptype in ["gate_proj", "up_proj", "down_proj"]:
            # Load all expert weights
            weights = {}
            for e in range(n_experts):
                if e in w_index[layer] and ptype in w_index[layer][e]:
                    fp, k = w_index[layer][e][ptype]
                    W = load_weight(fp, k)
                    weights[e] = W

            if len(weights) < 2:
                continue

            # Stack weights: (E, d_out, d_in)
            W_stack = torch.stack([weights[e] for e in sorted(weights.keys())])
            E, d_out, d_in = W_stack.shape

            # Compute expert output signatures on real inputs
            signatures = []
            for e in sorted(weights.keys()):
                Y = compute_expert_outputs(X, weights[e])  # (N, d_out)
                sig = Y.mean(dim=0)
                signatures.append(sig.numpy())
            signatures = np.array(signatures)  # (E, d_out)

            # K-means clustering
            K_test = [3, 5, 8, 10, 15, 20]
            for K in K_test:
                if K >= E:
                    continue
                km = KMeans(n_clusters=K, random_state=42, n_init=10)
                labels = km.fit_predict(signatures)

                # Learn prototype: mean of cluster members
                prototypes = torch.zeros(K, d_out, d_in)
                prototype_weights = {}
                for c in range(K):
                    member_indices = [list(sorted(weights.keys()))[i]
                                      for i in range(E) if labels[i] == c]
                    if member_indices:
                        prototypes[c] = torch.stack([weights[idx]
                                                     for idx in member_indices]).mean(dim=0)

                # Simulate: each expert is represented as prototype + delta
                # Compression ratio:
                #   Original: E * d_out * d_in * 4 bytes (float32)
                #   Clustered: K * d_out * d_in * 4 + E * d_out * d_in * bits_per_delta/8
                # We measure: what quantization level is needed for deltas?

                # Compute deltas from prototypes
                deltas = {}
                for i, e in enumerate(sorted(weights.keys())):
                    c = labels[i]
                    deltas[e] = weights[e] - prototypes[c]

                # Measure delta stats
                delta_vals = torch.cat([d.ravel() for d in deltas.values()])
                orig_vals = torch.cat([w.ravel() for w in weights.values()])
                delta_std = float(delta_vals.std())
                orig_std = float(orig_vals.std())
                std_ratio = delta_std / orig_std

                # PSNR of reconstruction (reconstruct expert from prototype + delta)
                total_mse = 0.0
                total_elem = 0
                max_abs = max(float(orig_vals.abs().max()), 1e-12)
                for i, e in enumerate(sorted(weights.keys())):
                    c = labels[i]
                    # Simulate delta quantization at various bits
                    delta = deltas[e]
                    # Use actual delta (lossless) to measure upper bound
                    recon = prototypes[c] + delta
                    mse = float(((weights[e] - recon) ** 2).mean())
                    total_mse += mse * weights[e].numel()
                    total_elem += weights[e].numel()

                psnr = 20 * math.log10(max_abs / math.sqrt(total_mse / total_elem + 1e-12))

                result = {
                    "layer": layer, "ptype": ptype, "K": K,
                    "n_experts": E, "std_ratio": std_ratio,
                    "psnr": psnr,
                }
                all_psnr_results.append(result)

                # Compression ratio (delta quantized to int8)
                delta_bits = 8  # reasonable for deltas
                original_bytes = E * d_out * d_in * 4
                prototype_bytes = K * d_out * d_in * 4
                delta_bytes = E * d_out * d_in * delta_bits / 8
                total_bytes = prototype_bytes + delta_bytes
                compression = original_bytes / total_bytes

                print(f"    {ptype:>10} K={K:>2}: "
                      f"std_ratio={std_ratio:.3f}, "
                      f"prototype PSNR={psnr:.2f} dB (lossless delta), "
                      f"compression={compression:.2f}x")

    # ========================================================================
    # 4. Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n  Output cosine similarity (across all sampled layers):")
    mean_sims = [s["mean"] for s in layer_output_sim_stats.values()]
    max_sims = [s["max"] for s in layer_output_sim_stats.values()]
    print(f"    Mean cosine: {np.mean(mean_sims):.4f}")
    print(f"    Max cosine:  {np.max(max_sims):.4f}")

    if np.mean(mean_sims) > 0.5:
        print(f"\n  → Strong functional clustering! Prototype approach is VERY promising.")
    elif np.mean(mean_sims) > 0.3:
        print(f"\n  → Moderate functional clustering. Prototype + delta may work.")
    elif np.mean(mean_sims) > 0.1:
        print(f"\n  → Weak functional clustering. Prototype benefit is marginal.")
    else:
        print(f"\n  → Experts are functionally orthogonal — no behavioral redundancy.")
        print(f"    Prototype clustering will NOT provide meaningful compression.")

    # Save results
    output_json = os.path.join(INPUT_DIR, "behavior_clustering_results.json")
    output = {
        "output_cosine_similarity": {str(k): v for k, v in layer_output_sim_stats.items()},
        "prototype_psnr": all_psnr_results,
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_json}")

    elapsed = time.perf_counter() - t_start
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
