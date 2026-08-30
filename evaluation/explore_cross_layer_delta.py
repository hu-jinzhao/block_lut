"""Cross-layer same-index expert similarity analysis — memory-efficient version.

Processes one layer pair at a time to avoid OOM.
"""
import os, sys, math, time, json
import numpy as np
import torch
from collections import defaultdict
from safetensors import safe_open
from tqdm import tqdm

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"


def build_index(safetensor_files):
    """Build (layer, expert_idx, proj_type) -> (file_path, key) index."""
    index = defaultdict(list)  # (layer, expert_idx) -> [(file, key, proj_type)]
    for fp in safetensor_files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                parts = k.split(".")
                layer = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6]
                index[(layer, expert_idx)].append((fp, k, proj_type))
    return index


def load_tensor(fp, key):
    with safe_open(fp, framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(torch.float32)


def main():
    t_start = time.perf_counter()

    safetensor_files = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")
    )
    print(f"Found {len(safetensor_files)} safetensor files")

    print("Building index...")
    index = build_index(safetensor_files)

    layers = sorted(set(k[0] for k in index))
    experts_per_layer = sorted(set(k[1] for k in index))
    n_layers = len(layers)
    n_experts = len(experts_per_layer)
    print(f"Layers: {layers[0]}-{layers[-1]} ({n_layers}), Experts: {n_experts}")

    # ========================================================================
    # 1. Same-index cosine similarity (adjacent layers)
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. SAME-INDEX COSINE SIMILARITY (ADJACENT LAYERS)")
    print("=" * 80)

    for ptype in ["gate_proj", "up_proj", "down_proj"]:
        all_cos = []
        per_expert_cos = defaultdict(list)

        pbar = tqdm(range(n_layers - 1), desc=f"  {ptype}")
        for li in pbar:
            l_curr, l_next = layers[li], layers[li + 1]
            for e in experts_per_layer:
                # Find tensor keys for this (layer, expert, ptype)
                curr_items = [(fp, k) for fp, k, pt in index[(l_curr, e)] if pt == ptype]
                next_items = [(fp, k) for fp, k, pt in index[(l_next, e)] if pt == ptype]
                if not curr_items or not next_items:
                    continue

                v_curr = load_tensor(*curr_items[0]).ravel()
                v_next = load_tensor(*next_items[0]).ravel()

                cos = float(
                    torch.dot(v_curr, v_next)
                    / (torch.norm(v_curr) * torch.norm(v_next) + 1e-12)
                )
                all_cos.append(cos)
                per_expert_cos[e].append(cos)

        all_cos = np.array(all_cos)
        print(f"    N={len(all_cos)} pairs")
        print(f"    mean={all_cos.mean():.4f}, std={all_cos.std():.4f}")
        print(f"    min={all_cos.min():.4f}, max={all_cos.max():.4f}")
        print(f"    |cos|>0.3: {100*np.mean(np.abs(all_cos)>0.3):.1f}%")
        print(f"    |cos|>0.5: {100*np.mean(np.abs(all_cos)>0.5):.1f}%")
        print(f"    |cos|>0.7: {100*np.mean(np.abs(all_cos)>0.7):.1f}%")

        # Per-expert mean
        em = [np.mean(per_expert_cos[e]) for e in experts_per_layer]
        print(f"    per-expert mean range: [{min(em):.4f}, {max(em):.4f}]")

    # ========================================================================
    # 2. Best-match analysis (gate_proj only, sample layers)
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. BEST-MATCH CROSS-LAYER (gate_proj, all adjacent pairs)")
    print("=" * 80)

    ptype = "gate_proj"
    same_index_hits = 0
    total_pairs = 0

    for li in range(n_layers - 1):
        l_curr, l_next = layers[li], layers[li + 1]

        # Load all expert vecs for these 2 layers
        vecs_curr, vecs_next = [], []
        for e in experts_per_layer:
            ci = [(fp, k) for fp, k, pt in index[(l_curr, e)] if pt == ptype]
            ni = [(fp, k) for fp, k, pt in index[(l_next, e)] if pt == ptype]
            vc = load_tensor(*ci[0]).ravel() if ci else torch.zeros(1)
            vn = load_tensor(*ni[0]).ravel() if ni else torch.zeros(1)
            vecs_curr.append(vc / (torch.norm(vc) + 1e-12))
            vecs_next.append(vn / (torch.norm(vn) + 1e-12))
        sim = torch.stack(vecs_curr) @ torch.stack(vecs_next).T

        best_idx = sim.argmax(dim=1)
        best_vals = sim.max(dim=1).values
        same = (best_idx == torch.arange(n_experts)).float().mean().item()
        same_index_hits += int(same * n_experts)
        total_pairs += n_experts

        print(f"  L{layers[li]:>2}→L{layers[li+1]:>2}: "
              f"avg best={best_vals.mean():.4f}, same-idx={same*100:.0f}%")

    print(f"\n  Overall same-index rate: {same_index_hits}/{total_pairs} "
          f"({100*same_index_hits/total_pairs:.1f}%)")

    # ========================================================================
    # 3. Delta std ratio (gate_proj only)
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. DELTA STD RATIO (same-index, gate_proj)")
    print("=" * 80)

    ratios = []
    for li in range(n_layers - 1):
        l_curr, l_next = layers[li], layers[li + 1]
        for e in experts_per_layer:
            ci = [(fp, k) for fp, k, pt in index[(l_curr, e)] if pt == ptype]
            ni = [(fp, k) for fp, k, pt in index[(l_next, e)] if pt == ptype]
            if not ci or not ni:
                continue
            v_curr = load_tensor(*ci[0]).numpy().ravel()
            v_next = load_tensor(*ni[0]).numpy().ravel()
            delta = v_next - v_curr
            ratios.append(float(np.std(delta) / (np.std(v_curr) + 1e-12)))

    ratios = np.array(ratios)
    print(f"    N={len(ratios)} pairs")
    print(f"    mean={ratios.mean():.4f}, std={ratios.std():.4f}")
    print(f"    min={ratios.min():.4f}, max={ratios.max():.4f}")
    print(f"    ratio<0.5: {100*np.mean(ratios<0.5):.1f}%")
    print(f"    ratio<0.7: {100*np.mean(ratios<0.7):.1f}%")
    print(f"    ratio<0.9: {100*np.mean(ratios<0.9):.1f}%")

    if ratios.mean() < 0.5:
        print(f"\n    → DELTA IS SIGNIFICANTLY SMALLER — delta coding promising!")
    elif ratios.mean() < 0.9:
        print(f"\n    → Delta is somewhat smaller — marginal benefit")
    else:
        print(f"\n    → Delta ≈ original ({ratios.mean():.3f}x) — NO benefit")

    # ========================================================================
    # 4. Block128 entropy comparison (gate_proj, sample)
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. BLOCK128 ENTROPY: ORIGINAL vs DELTA (gate_proj, sample)")
    print("=" * 80)

    BLOCK_SIZE = 128
    sample_pairs = [(layers[0], layers[1]), (layers[11], layers[12])]
    orig_bits_list = []
    delta_bits_list = []

    for l_curr, l_next in sample_pairs:
        for e in [0, 30, 59]:
            ci = [(fp, k) for fp, k, pt in index[(l_curr, e)] if pt == ptype]
            ni = [(fp, k) for fp, k, pt in index[(l_next, e)] if pt == ptype]
            if not ci or not ni:
                continue
            v_curr = load_tensor(*ci[0]).numpy().ravel()
            v_next = load_tensor(*ni[0]).numpy().ravel()
            delta = v_next - v_curr

            for name, arr in [("orig", v_curr), ("delta", delta)]:
                n = arr.size
                nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                pad = nb * BLOCK_SIZE - n
                if pad > 0:
                    arr = np.pad(arr, (0, pad))
                absmax = np.max(np.abs(arr.reshape(nb, BLOCK_SIZE)), axis=1)
                blocks = arr.reshape(nb, BLOCK_SIZE)
                normed = blocks / (absmax[:, None] + 1e-12)
                # Uniform 8-bit quantization on normalized values
                q = np.clip(np.round(normed * 127), -128, 127).astype(np.int8)
                # Shannon entropy of quantized values
                from collections import Counter
                freq = Counter(q.ravel().tolist())
                total = sum(freq.values())
                ent = -sum((c/total) * math.log2(c/total) for c in freq.values())
                bits = ent + 16.0 / BLOCK_SIZE  # add absmax overhead
                if name == "orig":
                    orig_bits_list.append(bits)
                else:
                    delta_bits_list.append(bits)

            print(f"  L{l_curr}→L{l_next} E{e:>2}: "
                  f"orig={orig_bits_list[-1]:.2f}, delta={delta_bits_list[-1]:.2f} bits")

    orig_mean = np.mean(orig_bits_list)
    delta_mean = np.mean(delta_bits_list)
    print(f"\n  Mean: orig={orig_mean:.2f}, delta={delta_mean:.2f} bits")
    print(f"  Savings: {orig_mean - delta_mean:.2f} bits ({100*(orig_mean-delta_mean)/orig_mean:.1f}%)")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 80}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
