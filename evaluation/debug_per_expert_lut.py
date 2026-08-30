"""Per-expert learned LUT PSNR comparison vs global LUT / uniform / NF"""
import os, sys, time
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
import numpy as np
import torch
from safetensors import safe_open
from sklearn.cluster import MiniBatchKMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/models/qwen/blocklut_256.npy"
BLOCK_SIZE = 128
N_CENTROIDS = 255  # 255 reconstruction values + 0 (zero centered)
N_SAMPLES = 50     # number of experts to evaluate (spread across layers)

# ---- Load expert tensors ----
print("Loading expert tensors...")
files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.safetensors')])
all_experts = {}
for fname in files:
    fpath = os.path.join(MODEL_DIR, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" in k and "shared_expert" not in k:
                all_experts[k] = f.get_tensor(k).to(torch.bfloat16)

print(f"Loaded {len(all_experts)} expert tensors")

# ---- Sample experts across layers for evaluation ----
# Sort keys by layer number, then sample evenly
sorted_keys = sorted(all_experts.keys())
sample_step = max(1, len(sorted_keys) // N_SAMPLES)
sample_keys = sorted_keys[::sample_step][:N_SAMPLES]
print(f"Evaluating {len(sample_keys)} experts across layers")

# ---- Load global LUT ----
global_lut = np.load(LUT_PATH)
global_lut_f32 = np.sort(global_lut.astype(np.float32))
global_lut_bf16 = torch.from_numpy(global_lut_f32.copy()).to(torch.bfloat16)

def quantize_with_lut(flat_values_f32, lut_f32, lut_bf16):
    """Quantize flat_values (float32 in [-1,1]) using LUT.

    Matches the actual pipeline: searchsorted in float32 (CPU),
    recover using bf16 LUT (GPU kernel behavior).
    """
    midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
    idx = np.searchsorted(midpoints, flat_values_f32).astype(np.uint8)
    # Recover via bf16 LUT (same as CUDA kernel: bf16_lut[idx] * bf16_absmax)
    reco_f32 = lut_f32[idx]
    reco_bf16 = torch.from_numpy(reco_f32.astype(np.float32)).to(torch.bfloat16)
    return reco_bf16


def blockwise_uniform(t, block_size=128):
    """Block128 uniform quantization."""
    flat = t.ravel()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)
    normalized = blocks.float() / absmax.unsqueeze(1)
    idx = ((normalized + 1.0) / 2.0 * 255 + 0.5).clamp(0, 255).to(torch.uint8)
    dequant = (idx.float() / 255 * 2.0 - 1.0) * absmax.unsqueeze(1)
    reco = dequant.to(torch.bfloat16).ravel()[:n].reshape(t.shape)
    return reco


def blockwise_nf(t, block_size=128):
    """Block128 NF (pow 0.6) quantization."""
    flat = t.ravel()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)
    abs_blocks = blocks.abs().float() / absmax.unsqueeze(1)

    p = 0.6
    transformed = abs_blocks ** p
    idx = (transformed * 255 + 0.5).clamp(0, 255).to(torch.uint8)
    dequant = (idx.float() / 255) ** (1.0 / p)
    reco = (torch.sign(blocks.float()) * dequant * absmax.unsqueeze(1)).to(torch.bfloat16)
    reco = reco.ravel()[:n].reshape(t.shape)
    return reco


def blocklut_global(t, lut_f32, lut_bf16, block_size=128):
    """Block128 + global LUT quantization. lut_f32 for searchsorted, lut_bf16 for recovery."""
    flat = t.ravel()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)
    normalized = blocks.float() / absmax.unsqueeze(1)  # float32

    reco_normalized_bf16 = quantize_with_lut(normalized.ravel().numpy(), lut_f32, lut_bf16)
    reco_normalized = reco_normalized_bf16.reshape(n_blocks, block_size).float()

    reco = (reco_normalized * absmax.unsqueeze(1)).to(torch.bfloat16)
    reco = reco.ravel()[:n].reshape(t.shape)
    return reco


def blocklut_per_expert(t, lut_f32, lut_bf16, block_size=128):
    """Block128 + per-expert LUT quantization. lut_f32 for searchsorted, lut_bf16 for recovery."""
    flat = t.ravel()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n, dtype=flat.dtype)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.float().clamp(min=1e-12)
    normalized = blocks.float() / absmax.unsqueeze(1)  # float32

    reco_normalized_bf16 = quantize_with_lut(normalized.ravel().numpy(), lut_f32, lut_bf16)
    reco_normalized = reco_normalized_bf16.reshape(n_blocks, block_size).float()

    reco = (reco_normalized * absmax.unsqueeze(1)).to(torch.bfloat16)
    reco = reco.ravel()[:n].reshape(t.shape)
    return reco


def learn_per_expert_lut(tensor, block_size=128, n_centroids=255, sample_size=200000):
    """Learn a per-expert LUT from block-normalized values (float32).

    Returns (lut_f32, lut_bf16): sorted LUT for searchsorted (f32) and recovery (bf16).
    """
    flat = tensor.ravel().float()
    n = flat.numel()
    n_blocks = (n + block_size - 1) // block_size
    padded = n_blocks * block_size
    if padded > n:
        flat = torch.cat([flat, torch.zeros(padded - n)])

    blocks = flat.reshape(n_blocks, block_size)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    normalized = (blocks / absmax.unsqueeze(1)).ravel().numpy()

    # Sample for K-means
    if len(normalized) > sample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(normalized), sample_size, replace=False)
        data = normalized[indices].reshape(-1, 1)
    else:
        data = normalized.reshape(-1, 1)

    # K-means clustering
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=42,
                              batch_size=4096, n_init=1, max_iter=100)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_.ravel().astype(np.float32)
    centroids.sort()

    lut_bf16 = torch.from_numpy(centroids.copy()).to(torch.bfloat16)
    return centroids, lut_bf16


def compute_psnr(original, recovered):
    """Compute var-based PSNR."""
    o = original.float().numpy().ravel()
    r = recovered.float().numpy().ravel()
    mse = ((o - r) ** 2).mean()
    if mse == 0:
        return 99.0
    return 10 * np.log10(o.var() / mse)


# ---- Run comparison ----
print(f"\n{'Expert':<50} {'GlobalLUT':>10} {'PerExpLUT':>10} {'Uniform':>10} {'NF(pow0.6)':>10} {'Improve':>10}")
print("-" * 105)

results = {"global_lut": [], "per_expert_lut": [], "uniform": [], "nf": []}
timings = {"train": [], "eval": []}

for i, key in enumerate(sample_keys):
    t = all_experts[key]

    # Learn per-expert LUT
    t0 = time.perf_counter()
    per_lut_f32, per_lut_bf16 = learn_per_expert_lut(t)
    train_time = time.perf_counter() - t0
    timings["train"].append(train_time)

    # Evaluate all methods
    t0 = time.perf_counter()
    psnr_global = compute_psnr(t, blocklut_global(t, global_lut_f32, global_lut_bf16))
    psnr_per = compute_psnr(t, blocklut_per_expert(t, per_lut_f32, per_lut_bf16))
    psnr_uni = compute_psnr(t, blockwise_uniform(t))
    psnr_nf = compute_psnr(t, blockwise_nf(t))
    eval_time = time.perf_counter() - t0
    timings["eval"].append(eval_time)

    improvement = psnr_per - psnr_global
    results["global_lut"].append(psnr_global)
    results["per_expert_lut"].append(psnr_per)
    results["uniform"].append(psnr_uni)
    results["nf"].append(psnr_nf)

    short_name = key.split("model.layers.")[1] if "model.layers." in key else key
    print(f"{short_name:<50} {psnr_global:10.2f} {psnr_per:10.2f} {psnr_uni:10.2f} {psnr_nf:10.2f} {improvement:+10.2f}")

# ---- Summary ----
print("\n" + "=" * 105)
print("SUMMARY")
print("-" * 50)
for method in ["global_lut", "per_expert_lut", "uniform", "nf"]:
    vals = results[method]
    print(f"  {method:<20}: mean={np.mean(vals):.2f}  min={min(vals):.2f}  max={max(vals):.2f} dB")

print(f"\n  Per-expert improvement over global LUT: +{np.mean(results['per_expert_lut']) - np.mean(results['global_lut']):.2f} dB")
print(f"  Per-expert improvement over uniform:    +{np.mean(results['per_expert_lut']) - np.mean(results['uniform']):.2f} dB")
print(f"  Gap to NF (pow 0.6):                    {np.mean(results['nf']) - np.mean(results['per_expert_lut']):.2f} dB")
print(f"\n  Avg train time (per expert): {np.mean(timings['train']):.2f}s")
print(f"  Total train time ({len(sample_keys)} experts): {sum(timings['train']):.1f}s")
print(f"  Projected for all 4320 experts: {sum(timings['train']) / len(sample_keys) * 4320 / 60:.1f} min")
print("\nDone.")
