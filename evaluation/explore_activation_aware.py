"""
激活感知量化实验

核心思路：对于线性层 y = Wx，输出 distortion 取决于输入激活分布
  E[||(W - W_q)x||²] = sum_i sum_j delta_ij² * E[x_j²]
所以激活方差大的输入通道需要更高的量化精度。

实验：
1. 分析 W 的列间 norm 差异（间接反映输入通道重要性）
2. 假设激活分布，评估激活感知量化 vs 均匀量化的输出 distortion
3. 尝试跑几轮推理收集真实激活统计
"""

import os, math, sys, time, collections
from dataclasses import dataclass
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
import heapq

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"

# ============================================================================
# Huffman estimate
# ============================================================================
def huffman_estimate(data: np.ndarray) -> float:
    freq = collections.Counter(data.ravel().tolist())
    total = sum(freq.values())
    if total == 0:
        return 0
    heap = []
    for sym, f in freq.items():
        heapq.heappush(heap, (f, id(sym), [sym, f]))
    if len(heap) <= 1:
        return 1.0
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, id(n1), [n1, n2]))
    def get_lengths(node, depth=0, lengths=None):
        if lengths is None:
            lengths = {}
        if isinstance(node, list):
            if len(node) == 2 and not isinstance(node[0], list):
                lengths[node[0]] = max(depth, 1)
            else:
                get_lengths(node[0], depth + 1, lengths)
                get_lengths(node[1], depth + 1, lengths)
        return lengths
    lengths = get_lengths(heap[0][2])
    return sum(freq[sym] * lengths.get(sym, 1) for sym in freq) / total

# ============================================================================
# Block-wise quant (generic bit width)
# ============================================================================
def blockwise_quantize(x: np.ndarray, block_size: int, nbits: int):
    """Block-wise absmax uniform quantization to nbits."""
    max_val = 2 ** (nbits - 1) - 1
    n = x.size
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    indices = np.zeros(nb * block_size, dtype=np.uint8)
    absmax_vals = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s, e = b * block_size, (b+1) * block_size
        block = x[s:e]
        amax = np.max(np.abs(block)) or 1e-12
        absmax_vals[b] = amax
        scale = amax / max_val
        q = np.clip(np.round(block / scale), -max_val-1, max_val).astype(np.int8)
        indices[s:e] = q.view(np.uint8)
    return indices, absmax_vals

def blockwise_dequantize(indices: np.ndarray, absmax_vals: np.ndarray,
                         block_size: int, nbits: int, orig_len: int):
    max_val = 2 ** (nbits - 1) - 1
    n = indices.size
    x = np.zeros(n, dtype=np.float32)
    for b in range((n + block_size - 1) // block_size):
        s, e = b * block_size, min((b+1) * block_size, n)
        q = indices[s:e].view(np.int8).astype(np.float32)
        scale = absmax_vals[b] / max_val
        x[s:e] = q * scale
    return x[:orig_len]

# ============================================================================
# 1. Column norm analysis
# ============================================================================
def analyze_column_norms():
    print("=" * 100)
    print("1. PER-COLUMN WEIGHT NORM ANALYSIS")
    print("=" * 100)
    print("If column norms vary significantly → activation-aware can help")
    print("If all columns have similar norm → activation-aware won't help\n")

    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])

    # Collect column norms across layers/experts
    import gc
    col_norm_cv_all = []  # CV = std/mean of column norms per matrix

    tensor_count = 0
    for path in tqdm(sft_files, desc="Loading safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                tensor = f.get_tensor(k)
                W = tensor.to(torch.float32).numpy()
                # W shape: (out_features, in_features)
                col_norms = np.linalg.norm(W, axis=0)  # per-input-channel norm
                cv = np.std(col_norms) / (np.mean(col_norms) + 1e-12)
                col_norm_cv_all.append(cv)
                tensor_count += 1
        gc.collect()

    print(f"Analyzed {tensor_count} expert tensors")
    print(f"\nColumn norm CV (std/mean):")
    print(f"  Mean: {np.mean(col_norm_cv_all):.4f}")
    print(f"  Median: {np.median(col_norm_cv_all):.4f}")
    print(f"  Min: {np.min(col_norm_cv_all):.4f}")
    print(f"  Max: {np.max(col_norm_cv_all):.4f}")
    print(f"  % with CV > 0.5: {np.mean(np.array(col_norm_cv_all) > 0.5) * 100:.1f}%")
    print(f"  % with CV > 1.0: {np.mean(np.array(col_norm_cv_all) > 1.0) * 100:.1f}%")

    # Interpretation
    if np.mean(col_norm_cv_all) < 0.2:
        print(f"\n  → Columns norms are very uniform (mean CV={np.mean(col_norm_cv_all):.3f}).")
        print(f"    Activation-aware per-channel scaling will NOT help significantly.")
    elif np.mean(col_norm_cv_all) < 0.5:
        print(f"\n  → Columns norms are somewhat uniform (mean CV={np.mean(col_norm_cv_all):.3f}).")
        print(f"    Activation-aware scaling may provide small benefits.")
    else:
        print(f"\n  → Column norms vary significantly (mean CV={np.mean(col_norm_cv_all):.3f}).")
        print(f"    Activation-aware quantization has high potential!")

    return col_norm_cv_all

# ============================================================================
# 2. Output distortion simulation
# ============================================================================
def simulate_output_distortion(num_inputs=50000):
    """
    For a sample expert matrix, simulate output distortion with:
    (a) Weight-only PSNR (uniform quantization)
    (b) Output distortion with uniformly distributed activations
    (c) Output distortion with skewed activations (some channels 10x larger)
    (d) Activation-scaled quantization (put more bits on high-activation channels)

    This shows the MAXIMUM BENEFIT of activation-aware quantization.
    """
    print("\n" + "=" * 100)
    print("2. OUTPUT DISTORTION SIMULATION")
    print("=" * 100)

    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])

    # Pick a representative expert tensor
    tensor_name = None
    W = None
    with safe_open(sft_files[0], framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" in k and "gate_proj" in k and "experts.0" in k:
                W = f.get_tensor(k).to(torch.float32).numpy()
                tensor_name = k
                break
        if W is None:
            # Fallback: pick any expert tensor
            for k in f.keys():
                if "expert" in k:
                    W = f.get_tensor(k).to(torch.float32).numpy()
                    tensor_name = k
                    break

    print(f"Using tensor: {tensor_name}, shape={W.shape} (out={W.shape[0]}, in={W.shape[1]})")

    out_dim, in_dim = W.shape

    # --- Generate synthetic activations ---
    rng = np.random.RandomState(42)

    # Case A: Uniform activations (all channels same variance)
    x_uniform = rng.randn(num_inputs, in_dim).astype(np.float32)  # N(in_dim, 1)

    # Case B: Skewed activations (some channels have larger variance)
    # log-space: variance ranges from 0.01 to 100
    channel_std = np.exp(rng.uniform(np.log(0.1), np.log(10), in_dim)).astype(np.float32)
    x_skewed = rng.randn(num_inputs, in_dim).astype(np.float32) * channel_std

    # --- Quantize W with block128 int8 ---
    W_flat = W.ravel()
    nbits = 8
    bs = 128
    indices, absmax_vals = blockwise_quantize(W_flat, bs, nbits)
    W_q_flat = blockwise_dequantize(indices, absmax_vals, bs, nbits, W_flat.size)
    W_q = W_q_flat.reshape(W.shape)

    # --- Weight PSNR ---
    mse_w = np.mean((W - W_q) ** 2)
    var_w = np.var(W)
    psnr_w = 10 * math.log10(var_w / mse_w) if mse_w > 0 else float('inf')

    # --- Output distortion ---
    def output_snr(W_orig, W_quant, X):
        """Output SNR: 10*log10(var(Wx) / mse(Wx, W_q x))"""
        Y_orig = X @ W_orig.T  # (N, out_dim) -- note: X is (N, in), W is (out, in)
        Y_quant = X @ W_quant.T
        var_y = np.var(Y_orig)
        mse_y = np.mean((Y_orig - Y_quant) ** 2)
        return 10 * math.log10(var_y / mse_y) if mse_y > 0 else float('inf')

    snr_uniform_act = output_snr(W, W_q, x_uniform)
    snr_skewed_act = output_snr(W, W_q, x_skewed)

    print(f"\n  Weight PSNR (int8 uniform): {psnr_w:.2f} dB")
    print(f"  Output SNR with uniform activations: {snr_uniform_act:.2f} dB")
    print(f"  Output SNR with skewed activations:  {snr_skewed_act:.2f} dB")

    if abs(snr_uniform_act - psnr_w) < 1.0:
        print(f"  → With uniform activations, output SNR ≈ weight PSNR (as expected)")
    if abs(snr_skewed_act - snr_uniform_act) > 2.0:
        delta = snr_skewed_act - snr_uniform_act
        print(f"  → Skewed activations cause {delta:+.1f} dB shift in output SNR")
        print(f"  → This is where activation-aware quantization can help!")
    else:
        print(f"  → Even skewed activations don't shift output SNR much")
        print(f"  → Because errors in W are random and average out over many input dims")

    # --- Now test activation-aware per-column scaling ---
    # Use channel_std as the "known" activation statistics
    # Apply per-column scaling: divide column j by channel_std[j] before quantizing
    # This puts more precision on high-activation columns

    # AWQ-style: W_scaled = W * diag(1/channel_std), quantize, then W_q = W_q_scaled * diag(channel_std)
    scale_factors = channel_std / np.mean(channel_std)  # normalize
    W_scaled = W / scale_factors[np.newaxis, :]  # broadcast over rows
    W_scaled_flat = W_scaled.ravel()

    indices_scaled, am_scaled = blockwise_quantize(W_scaled_flat, bs, nbits)
    W_q_scaled_flat = blockwise_dequantize(indices_scaled, am_scaled, bs, nbits, W_scaled_flat.size)
    W_q_scaled = W_q_scaled_flat.reshape(W.shape) * scale_factors[np.newaxis, :]

    # Weight PSNR of activation-scaled version
    mse_w_scaled = np.mean((W - W_q_scaled) ** 2)
    psnr_w_scaled = 10 * math.log10(var_w / mse_w_scaled) if mse_w_scaled > 0 else float('inf')

    # Output SNR of activation-scaled version
    snr_scaled = output_snr(W, W_q_scaled, x_skewed)

    print(f"\n  --- Activation-scaled quantization (using known activation stats) ---")
    print(f"  Weight PSNR (scaled int8): {psnr_w_scaled:.2f} dB")
    print(f"  Output SNR (scaled int8, skewed act): {snr_scaled:.2f} dB")
    print(f"  Improvement in output SNR: {snr_scaled - snr_skewed_act:+.2f} dB")

    # --- Per-channel bit allocation ---
    # What if we use different bit widths per column based on activation importance?
    # High-activation columns get 8 bits, low-activation get 4 bits
    print(f"\n  --- Per-channel mixed precision (high-act channels: 8bit, low-act: 4bit) ---")
    high_mask = channel_std > np.median(channel_std)
    print(f"  High-activation channels: {high_mask.sum()}/{in_dim} ({high_mask.sum()/in_dim*100:.0f}%)")

    # Quantize high-act channels at 8bit, low-act at 4bit
    W_flat = W.ravel()
    # Reshape to (out_dim, in_dim) for per-column access
    W_q_mixed = np.zeros_like(W)

    for j in range(in_dim):
        col = W[:, j]
        amax = np.max(np.abs(col)) or 1e-12
        col_nbits = 8 if high_mask[j] else 4
        max_val = 2 ** (col_nbits - 1) - 1
        scale = amax / max_val
        q = np.clip(np.round(col / scale), -max_val-1, max_val).astype(np.float32)
        W_q_mixed[:, j] = q * scale

    mse_w_mixed = np.mean((W - W_q_mixed) ** 2)
    psnr_w_mixed = 10 * math.log10(var_w / mse_w_mixed) if mse_w_mixed > 0 else float('inf')
    snr_mixed = output_snr(W, W_q_mixed, x_skewed)
    avg_bits_mixed = 8 * high_mask.sum()/in_dim + 4 * (1 - high_mask.sum()/in_dim)

    print(f"  Avg bits: {avg_bits_mixed:.1f} bits/elem")
    print(f"  Weight PSNR (mixed): {psnr_w_mixed:.2f} dB")
    print(f"  Output SNR (mixed): {snr_mixed:.2f} dB")

    # Compare: uniform int6 (same avg bits as mixed 4/8)
    nbits6 = 6
    indices6, am6 = blockwise_quantize(W_flat, bs, nbits6)
    W_q6_flat = blockwise_dequantize(indices6, am6, bs, nbits6, W_flat.size)
    W_q6 = W_q6_flat.reshape(W.shape)
    mse_w6 = np.mean((W - W_q6) ** 2)
    psnr_w6 = 10 * math.log10(var_w / mse_w6) if mse_w6 > 0 else float('inf')
    snr6 = output_snr(W, W_q6, x_skewed)

    print(f"\n  --- Same bit budget, uniform int6 (6 bits vs mixed {avg_bits_mixed:.1f}) ---")
    print(f"  Weight PSNR (int6 uniform): {psnr_w6:.2f} dB")
    print(f"  Output SNR (int6 uniform): {snr6:.2f} dB")

    if snr_mixed > snr6:
        print(f"\n  → Mixed precision wins! Output SNR +{snr_mixed - snr6:.1f} dB at same bit budget")
    else:
        print(f"\n  → Uniform int6 is actually better. Output SNR: {snr6:.2f} vs {snr_mixed:.2f}")

    # Also test: what if activations are COMPLETELY flat? (all columns same std)
    x_flat = rng.randn(num_inputs, in_dim).astype(np.float32)
    snr_flat = output_snr(W, W_q, x_flat)
    print(f"\n  Reference: Output SNR with truly flat activations: {snr_flat:.2f} dB")

# ============================================================================
# 3. Try to collect real activations via inference
# ============================================================================
def collect_real_activations():
    """Try to run a few forward passes and collect expert input activations."""
    print("\n" + "=" * 100)
    print("3. REAL ACTIVATION COLLECTION")
    print("=" * 100)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Try to load model with CPU offloading to fit in memory
        print("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            offload_folder="/tmp/offload",
            max_memory={0: "7GB", "cpu": "32GB"}
        )

        # Calibration data: a few short sentences
        calibration_texts = [
            "The capital of France is",
            "Machine learning is a subset of artificial intelligence that",
            "The first law of thermodynamics states that energy",
            "In mathematics, the Pythagorean theorem describes the relationship",
            "The Python programming language was created by",
        ]

        # We need to hook into expert inputs
        # For Qwen MoE, experts are in model.layers.L.mlp.experts.E
        # The expert takes input from the shared layer

        activation_stats = {}  # layer -> expert -> per-channel stats

        @torch.no_grad()
        def hook_fn(layer_idx, expert_idx, proj_type):
            def hook(module, input, output):
                # input is typically a tuple
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                if x is None:
                    return
                # x shape: (batch*seq, in_features)
                x_cpu = x.detach().cpu().to(torch.float32)
                per_channel_std = x_cpu.std(dim=0).numpy()  # (in_features,)
                # Store first and second moment per channel
                key = (layer_idx, expert_idx, proj_type)
                if key not in activation_stats:
                    activation_stats[key] = {'sum_sq': np.zeros_like(per_channel_std),
                                              'sum': np.zeros_like(per_channel_std),
                                              'count': 0}
                activation_stats[key]['sum_sq'] += (x_cpu ** 2).sum(dim=0).numpy()
                activation_stats[key]['sum'] += x_cpu.sum(dim=0).numpy()
                activation_stats[key]['count'] += x_cpu.shape[0]
            return hook

        # Register hooks on expert layers
        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'experts'):
                for expert_idx, expert in enumerate(layer.mlp.experts):
                    for proj_type in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(expert, proj_type):
                            h = getattr(expert, proj_type).register_forward_hook(
                                hook_fn(layer_idx, expert_idx, proj_type)
                            )
                            hooks.append(h)

        print(f"Registered {len(hooks)} hooks")

        # Run calibration
        model.eval()
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        for text in tqdm(calibration_texts, desc="Running calibration"):
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            try:
                _ = model(**inputs, max_new_tokens=1)
            except Exception as e:
                print(f"  Warning: {e}")
                continue

        # Remove hooks
        for h in hooks:
            h.remove()

        # Analyze collected statistics
        if activation_stats:
            print(f"\nCollected activation stats for {len(activation_stats)} expert/proj combinations")

            cvs = []
            for key, stats in activation_stats.items():
                layer, expert, proj = key
                # Per-channel std
                mean_sq = stats['sum_sq'] / stats['count']
                mean_val = stats['sum'] / stats['count']
                var = mean_sq - mean_val ** 2
                std = np.sqrt(np.maximum(var, 0))
                cv = np.std(std) / (np.mean(std) + 1e-12)
                cvs.append(cv)

            print(f"  Per-channel activation std CV (across channels):")
            print(f"    Mean:   {np.mean(cvs):.4f}")
            print(f"    Median: {np.median(cvs):.4f}")
            print(f"    Min:    {np.min(cvs):.4f}")
            print(f"    Max:    {np.max(cvs):.4f}")

            if np.mean(cvs) < 0.3:
                print(f"\n  → Activation std is nearly uniform across channels.")
                print(f"    Activation-aware per-channel quantization will NOT provide significant benefit.")
            elif np.mean(cvs) < 0.7:
                print(f"\n  → Moderate variation in activation std across channels.")
                print(f"    Activation-aware quantization may provide small benefit (~1-2 dB output SNR).")
            else:
                print(f"\n  → High variation! Activation-aware quantization could be very beneficial.")
        else:
            print("  No activation statistics collected (hooks may not have triggered)")

    except Exception as e:
        print(f"Failed to collect activations: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 4. Sensitivity analysis
# ============================================================================
def sensitivity_analysis():
    """
    If we KNOW the activation distribution perfectly, what's the MAXIMUM benefit
    of activation-aware quantization over uniform quantization?

    Theory: for per-channel quantization with optimal bit allocation,
    the output distortion is:
      D = sum_j (1/12) * (range_j^2 / (2^(2*b_j) - 1)) * E[x_j^2]
    where range_j is the weight range of column j, b_j is bits for column j.

    We want to minimize D subject to sum_j b_j = B_total.

    Using Lagrange multipliers: b_j ∝ log2(range_j^2 * E[x_j^2])
    """
    print("\n" + "=" * 100)
    print("4. THEORETICAL MAXIMUM BENEFIT ANALYSIS")
    print("=" * 100)

    sft_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])

    # Collect column statistics for all expert tensors
    # For each matrix, compute:
    # - Per-column weight range (max - min)
    # - Estimate per-column activation variance from column norm (proxy)

    col_range_cv_all = []
    total_cols = 0

    for path in tqdm(sft_files[:2], desc="Sampling tensors"):  # Just sample 2 files
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                W = f.get_tensor(k).to(torch.float32).numpy()  # (out, in)
                col_range = np.max(W, axis=0) - np.min(W, axis=0)  # per-column range
                col_range_cv = np.std(col_range) / (np.mean(col_range) + 1e-12)
                col_range_cv_all.append(col_range_cv)
                total_cols += W.shape[1]

    print(f"\n  Per-column weight range variation across {len(col_range_cv_all)} matrices:")
    print(f"    Mean CV:   {np.mean(col_range_cv_all):.4f}")
    print(f"    Median CV: {np.median(col_range_cv_all):.4f}")
    print(f"    Min CV:    {np.min(col_range_cv_all):.4f}")
    print(f"    Max CV:    {np.max(col_range_cv_all):.4f}")

    # If column ranges have CV=0, then all columns need same bits → no benefit
    # If CV=1.0, optimal bit allocation could give ~3dB improvement at same bit budget

    # Simulate optimal bit allocation
    mean_cv = np.mean(col_range_cv_all)

    # For a model case: what's the theoretical max SNR improvement?
    # Generate synthetic column ranges matching the observed CV
    n_cols = 2048  # typical in_features
    rng = np.random.RandomState(42)

    # Log-normal distribution with given CV
    sigma = np.sqrt(np.log(mean_cv ** 2 + 1))
    col_importance = np.exp(rng.randn(n_cols) * sigma)

    # Uniform allocation: 8 bits per column
    # Optimal allocation: b_j proportional to log(col_importance)
    # Subject to sum b_j = 8 * n_cols

    b_uniform = np.ones(n_cols) * 8
    log_imp = np.log2(col_importance + 1e-12)
    b_optimal = 8.0 + (log_imp - np.mean(log_imp)) * 0.5  # adjust to match total bits
    b_optimal = np.clip(b_optimal, 2, 10)  # practical limits
    # Adjust to match total bit budget
    b_optimal = b_optimal * (8.0 * n_cols / np.sum(b_optimal))

    # Distortion per column: (range^2 / 12) * (1/2^(2b)) * act_var
    # range^2 * act_var approximated by col_importance
    D_uniform = np.sum(col_importance / (2 ** (2 * b_uniform)))
    D_optimal = np.sum(col_importance / (2 ** (2 * b_optimal)))

    snr_gain = 10 * math.log10(D_uniform / D_optimal)

    print(f"\n  Theoretical analysis (CV={mean_cv:.3f}, {n_cols} columns):")
    print(f"    Uniform 8-bit output distortion: D_uniform")
    print(f"    Optimal bit allocation output distortion: D_optimal")
    print(f"    Max theoretical SNR gain: {snr_gain:.1f} dB")

    # For comparison, what if we used 7 bits uniformly? (saving 12.5% storage)
    D_7bit = np.sum(col_importance / (2 ** (2 * 7)))
    snr_8vs7 = 10 * math.log10(D_7bit / D_uniform)
    print(f"    Switching from 8-bit to 7-bit loses: {snr_8vs7:.1f} dB")
    print(f"    → Activation-aware could recover {min(snr_gain, abs(snr_8vs7)):.1f} dB of that loss")

    return col_range_cv_all

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    col_norm_cv = analyze_column_norms()
    simulate_output_distortion()
    sensitivity_analysis()

    # Try real activations only if we have enough VRAM and the model loads
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    print(f"""
    Column norm CV: {np.mean(col_norm_cv):.4f}

    Interpretation:
    - CV < 0.2: Activation-aware quantization provides negligible benefit
    - CV 0.2-0.5: Small benefit possible (~1-2 dB at same bit budget)
    - CV > 0.5: Significant benefit possible

    Based on the column norm analysis, the actual benefit depends on whether
    the real activation distribution is more or less skewed than the weight
    column norms suggest.
    """)

    # Ask if user wants to try real inference
    print("To collect real activation statistics, we need to run the model on GPU.")
    print("This requires loading the full model (~5.4GB) and running calibration prompts.")
    print("Estimated time: 5-10 minutes.")
