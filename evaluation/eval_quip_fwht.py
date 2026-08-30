"""QuIP-style 4-bit quantization with FWHT incoherence processing.

Compares: Direct 4-bit vs QuIP 4-bit vs Direct 6-bit (reference).

Key: randomized Hadamard transform makes weight distribution more "incoherent",
which improves 4-bit uniform quantization. FWHT = O(n log n), runs in minutes.
"""
import os, sys, math, time, json
import numpy as np
from safetensors import safe_open

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
INPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/expert_behavior"
OUTPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results"
GROUP_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Fast Walsh-Hadamard Transform (normalized, in-place)
# =============================================================================

def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())

def fwht(x):
    """Normalized FWHT on last axis. H @ x where H_ij ∈ {±1/sqrt(n)}."""
    n = x.shape[-1]
    h = 1
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i:i + h].copy()
            b = x[..., i + h:i + 2 * h]
            x[..., i:i + h] = (a + b) * inv_sqrt2
            x[..., i + h:i + 2 * h] = (a - b) * inv_sqrt2
        h *= 2
    return x


# =============================================================================
# QuIP transforms
# =============================================================================

def quip_forward(W, s_left, s_right):
    """W' = H_left @ diag(s_left) @ W @ diag(s_right) @ H_right."""
    # Right transform: W @ diag(s_right) @ H
    W = W * s_right[None, :]   # scale columns
    W = fwht(W)                # FWHT on rows
    # Left transform: H @ diag(s_left) @ W
    W = W * s_left[:, None]    # scale rows
    W = fwht(W.T).T            # FWHT on columns
    return W


def quip_inverse(W_q, s_left, s_right):
    """W = diag(s_left) @ H_left @ W_q @ H_right @ diag(s_right)."""
    # Left inverse: diag(s_left) @ H @ W
    W = fwht(W_q.T).T          # FWHT on columns
    W = W * s_left[:, None]    # scale rows
    # Right inverse: W @ H @ diag(s_right)
    W = fwht(W)                # FWHT on rows
    W = W * s_right[None, :]   # scale columns
    return W


def pad_to_pow2(W, d_out_pad, d_in_pad):
    """Pad W to (d_out_pad, d_in_pad) with zeros, centered for better FWHT."""
    d_out, d_in = W.shape
    W_pad = np.zeros((d_out_pad, d_in_pad), dtype=W.dtype)
    o_start = (d_out_pad - d_out) // 2
    i_start = (d_in_pad - d_in) // 2
    W_pad[o_start:o_start + d_out, i_start:i_start + d_in] = W
    return W_pad, (o_start, d_out, i_start, d_in)


def unpad(W_pad, o_start, d_out, i_start, d_in):
    return W_pad[o_start:o_start + d_out, i_start:i_start + d_in]


# =============================================================================
# Quantization
# =============================================================================

def blocklut_quantize(W, n_levels, block_size=128):
    """Block-wise uniform quantization. n_levels=16 for 4-bit, 64 for 6-bit."""
    flat = W.ravel().astype(np.float32)
    n = flat.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad:
        flat = np.pad(flat, (0, pad))
    blocks = flat.reshape(nb, block_size)
    absmax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-12)
    normed = blocks / absmax[:, np.newaxis]
    # Uniform quantization to n_levels in [-1, 1]
    step = 2.0 / (n_levels - 1)
    q = np.clip(np.round((normed + 1.0) / step) * step - 1.0, -1.0, 1.0)
    recon = (q * absmax[:, np.newaxis]).ravel()[:n]
    return recon.reshape(W.shape).astype(np.float32), absmax


def compute_psnr(orig, recon):
    mse = np.mean((orig.ravel() - recon.ravel()) ** 2)
    var = np.var(orig.ravel())
    return 99.0 if mse == 0 else float(10 * math.log10(var / mse))


def compute_output_cosine(Y1, Y2):
    y1 = Y1.reshape(-1, Y1.shape[-1])
    y2 = Y2.reshape(-1, Y2.shape[-1])
    n1 = np.linalg.norm(y1, axis=1) + 1e-12
    n2 = np.linalg.norm(y2, axis=1) + 1e-12
    cos = np.sum(y1 * y2, axis=1) / (n1 * n2)
    return float(np.mean(cos)), float(np.std(cos)), float(np.min(cos))


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.perf_counter()

    # Load calibration inputs
    print("Loading calibration inputs...")
    calib = {}
    for layer in range(24):
        path = os.path.join(INPUT_DIR, f"inputs_layer_{layer:02d}.npy")
        if os.path.exists(path):
            calib[layer] = np.load(path).astype(np.float32)
    print(f"  Loaded inputs for {len(calib)} layers")

    # Build weight index
    sft_files = sorted(
        os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")
    )
    from collections import defaultdict
    w_index = defaultdict(dict)
    for fp in sft_files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                parts = k.split(".")
                layer = int(parts[2])
                expert_idx = int(parts[5])
                proj_type = parts[6]
                if expert_idx not in w_index[layer]:
                    w_index[layer][expert_idx] = {}
                w_index[layer][expert_idx][proj_type] = (fp, k)

    layers_moe = sorted(w_index.keys())
    n_experts = max(w_index[layers_moe[0]].keys()) + 1

    # Test config
    eval_layers = [0, 12, 23]
    eval_experts = [0, 5, 10, 15, 20]  # subset of experts per layer
    eval_layers = [l for l in eval_layers if l in calib]

    # =========================================================================
    # Per-expert QuIP evaluation
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"QUIP 4-BIT EVALUATION (FWHT incoherence processing)")
    print(f"  {len(eval_layers)} layers x {len(eval_experts)} experts x 2 projections")
    print(f"  Methods: Direct4 | QuIP4 (pad2048) | QuIP4 (partial) | Direct6 (ref)")
    print(f"{'='*70}")

    rng = np.random.RandomState(42)
    all_results = []

    for layer in eval_layers:
        X = calib[layer].astype(np.float32)
        n_inputs = min(20, len(X))
        X = X[:n_inputs]
        print(f"\n  Layer {layer} ({n_inputs} inputs):")

        for e_idx in eval_experts:
            for ptype in ["gate_proj", "up_proj"]:
                if e_idx not in w_index[layer] or ptype not in w_index[layer][e_idx]:
                    continue

                fp, key = w_index[layer][e_idx][ptype]
                with safe_open(fp, framework="pt", device="cpu") as f:
                    W = f.get_tensor(key).float().numpy()
                d_out, d_in = W.shape
                W = W.astype(np.float32)

                # --- Reference output ---
                Y_ref = X @ W.T  # (N, d_out)

                # --- Method 1: Direct 4-bit ---
                W_d4, _ = blocklut_quantize(W, 16, GROUP_SIZE)
                Y_d4 = X @ W_d4.T
                psnr_d4 = compute_psnr(W, W_d4)
                cos_d4 = compute_output_cosine(Y_ref, Y_d4)

                # --- Method 2: QuIP 4-bit (pad to power-of-2) ---
                d_out_pad = _next_pow2(d_out)
                d_in_pad = _next_pow2(d_in)
                s_left = rng.choice([-1.0, 1.0], size=d_out_pad).astype(np.float32)
                s_right = rng.choice([-1.0, 1.0], size=d_in_pad).astype(np.float32)

                need_pad = (d_out != d_out_pad) or (d_in != d_in_pad)
                if need_pad:
                    W_pad, (os_, do_, is_, di_) = pad_to_pow2(W, d_out_pad, d_in_pad)
                else:
                    W_pad = W.copy()
                    os_, do_, is_, di_ = 0, d_out, 0, d_in

                # Forward: incoherence transform
                W_qf = quip_forward(W_pad, s_left, s_right)
                # Quantize in incoherent space
                W_qf_q, _ = blocklut_quantize(W_qf, 16, GROUP_SIZE)
                # Inverse: back to weight space
                W_quip_recon = quip_inverse(W_qf_q, s_left, s_right)
                if need_pad:
                    W_quip_recon = unpad(W_quip_recon, os_, do_, is_, di_)

                Y_quip = X @ W_quip_recon.T
                psnr_quip = compute_psnr(W, W_quip_recon)
                cos_quip = compute_output_cosine(Y_ref, Y_quip)

                # --- Method 3: QuIP partial (signs only on non-pow2 dims, no pad) ---
                # Apply Hadamard only on pow2 dims, signs only on others
                sL = rng.choice([-1.0, 1.0], size=d_out).astype(np.float32)
                sR = rng.choice([-1.0, 1.0], size=d_in).astype(np.float32)

                W_partial = W.copy()
                # Apply right transform (signs always, FWHT if pow2)
                W_partial = W_partial * sR[None, :]
                if d_in == _next_pow2(d_in):
                    W_partial = fwht(W_partial)
                # Apply left transform
                W_partial = W_partial * sL[:, None]
                if d_out == _next_pow2(d_out):
                    W_partial = fwht(W_partial.T).T

                # Quantize
                W_pq_q, _ = blocklut_quantize(W_partial, 16, GROUP_SIZE)

                # Inverse
                if d_out == _next_pow2(d_out):
                    W_pq_q = fwht(W_pq_q.T).T
                W_pq_q = W_pq_q * sL[:, None]
                if d_in == _next_pow2(d_in):
                    W_pq_q = fwht(W_pq_q)
                W_pq_q = W_pq_q * sR[None, :]

                Y_partial = X @ W_pq_q.T
                psnr_partial = compute_psnr(W, W_pq_q)
                cos_partial = compute_output_cosine(Y_ref, Y_partial)

                # --- Method 4: Direct 6-bit (reference) ---
                W_d6, _ = blocklut_quantize(W, 64, GROUP_SIZE)
                Y_d6 = X @ W_d6.T
                psnr_d6 = compute_psnr(W, W_d6)
                cos_d6 = compute_output_cosine(Y_ref, Y_d6)

                result = {
                    "layer": layer, "expert": e_idx, "ptype": ptype,
                    "shape": f"{d_out}x{d_in}",
                    "pow2_out": (d_out == _next_pow2(d_out)),
                    "pow2_in": (d_in == _next_pow2(d_in)),
                    "direct4_psnr": psnr_d4, "direct4_cos": cos_d4[0],
                    "quip4_psnr": psnr_quip, "quip4_cos": cos_quip[0],
                    "quip4p_psnr": psnr_partial, "quip4p_cos": cos_partial[0],
                    "direct6_psnr": psnr_d6, "direct6_cos": cos_d6[0],
                }
                all_results.append(result)

        # Per-layer summary
        lr = [r for r in all_results if r["layer"] == layer]
        for method, p_key, c_key in [
            ("Direct4", "direct4_psnr", "direct4_cos"),
            ("QuIP4", "quip4_psnr", "quip4_cos"),
            ("QuIP4p", "quip4p_psnr", "quip4p_cos"),
            ("Direct6", "direct6_psnr", "direct6_cos"),
        ]:
            ps = [r[p_key] for r in lr]
            cs = [r[c_key] for r in lr]
            print(f"    {method:10s}: PSNR={np.mean(ps):.2f}±{np.std(ps):.2f} dB, "
                  f"cos={np.mean(cs):.4f}±{np.std(cs):.4f}")

    # =========================================================================
    # Global summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    methods = [
        ("Direct4", "direct4_psnr", "direct4_cos"),
        ("QuIP4", "quip4_psnr", "quip4_cos"),
        ("QuIP4p", "quip4p_psnr", "quip4p_cos"),
        ("Direct6", "direct6_psnr", "direct6_cos"),
    ]

    for name, pk, ck in methods:
        ps = [r[pk] for r in all_results]
        cs = [r[ck] for r in all_results]
        print(f"  {name:10s}: PSNR={np.mean(ps):.2f}±{np.std(ps):.2f} dB, "
              f"cos={np.mean(cs):.4f}±{np.std(cs):.4f}")

    # Compute QuIP gain
    quip_gain = np.mean([r["quip4_psnr"] - r["direct4_psnr"] for r in all_results])
    quip_gain_cos = np.mean([r["quip4_cos"] - r["direct4_cos"] for r in all_results])
    print(f"\n  QuIP4 gain over Direct4: +{quip_gain:.2f} dB PSNR, +{quip_gain_cos:.4f} cos")
    print(f"  Gap to Direct6: "
          f"{np.mean([r['direct6_psnr'] for r in all_results]) - np.mean([r['quip4_psnr'] for r in all_results]):.2f} dB")

    # Verbose per-result table
    print(f"\n  Per-expert detail:")
    print(f"  {'Layer':>5s} {'Exp':>4s} {'Proj':>9s} {'Shape':>12s} "
          f"{'D4_PSNR':>8s} {'Q4_PSNR':>8s} {'Q4p_PSNR':>8s} {'D6_PSNR':>8s} "
          f"{'D4_cos':>8s} {'Q4_cos':>8s} {'Q4p_cos':>8s}")
    for r in sorted(all_results, key=lambda r: (r["layer"], r["expert"])):
        print(f"  {r['layer']:5d} {r['expert']:4d} {r['ptype']:>9s} {r['shape']:>12s} "
              f"{r['direct4_psnr']:8.2f} {r['quip4_psnr']:8.2f} {r['quip4p_psnr']:8.2f} "
              f"{r['direct6_psnr']:8.2f} "
              f"{r['direct4_cos']:8.4f} {r['quip4_cos']:8.4f} {r['quip4p_cos']:8.4f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "quip_fwht_4bit.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {"eval_layers": eval_layers, "eval_experts": eval_experts,
                       "group_size": GROUP_SIZE},
            "summary": {
                name: {"psnr_mean": float(np.mean([r[pk] for r in all_results])),
                       "psnr_std": float(np.std([r[pk] for r in all_results])),
                       "cos_mean": float(np.mean([r[ck] for r in all_results])),
                       "cos_std": float(np.std([r[ck] for r in all_results]))}
                for name, pk, ck in methods
            },
            "quip_gain_psnr": float(quip_gain),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    elapsed = time.perf_counter() - t_start
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
