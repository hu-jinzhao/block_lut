"""K-means LUT 框架内优化: 加权采样 / NF初始化 / Per-layer LUT"""
import os, sys, time
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
import numpy as np
import torch
from safetensors import safe_open
from sklearn.cluster import MiniBatchKMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BLOCKLUT_PATH = "/home/hh/zip_Moe/LUT_MoE/models/qwen/blocklut_256.npy"
BS = 128
N_CENTROIDS = 255

# ---- Load experts ----
print("Loading experts...")
files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.safetensors')])
all_keys = []
for fname in files:
    fpath = os.path.join(MODEL_DIR, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" in k and "shared_expert" not in k:
                all_keys.append((k, fname))
all_keys = sorted(all_keys, key=lambda x: x[0])

# Load all weights
weights = {}
for fname in set(f[1] for f in all_keys):
    fpath = os.path.join(MODEL_DIR, fname)
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" in k and "shared_expert" not in k:
                weights[k] = f.get_tensor(k).to(torch.bfloat16)
print(f"Loaded {len(weights)} experts")

# ---- Prepare block-normalized data, grouped by layer ----
print("Preparing block-normalized data per layer...")
layer_data = {}  # layer_id -> normalized values
all_norm = []
for k in sorted(weights.keys()):
    parts = k.split(".")
    layer_id = None
    for i, p in enumerate(parts):
        if p == "layers":
            layer_id = int(parts[i+1])
            break
    if layer_id is None:
        continue

    w = weights[k].float().ravel()
    n = w.numel()
    nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad > 0:
        w = torch.cat([w, torch.zeros(pad)])
    blocks = w.reshape(nb, BS)
    absmax = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    norm = (blocks / absmax.unsqueeze(1)).ravel()

    step = max(1, norm.numel() // 50000)
    sample = norm[::step].numpy()
    all_norm.append(sample)

    if layer_id not in layer_data:
        layer_data[layer_id] = []
    layer_data[layer_id].append(sample)

all_norm = np.concatenate(all_norm).astype(np.float32)
print(f"Total training samples: {len(all_norm)}")
print(f"Layers: {sorted(layer_data.keys())}")

# ---- Baseline: current blocklut ----
blocklut = np.load(BLOCKLUT_PATH)
blocklut = np.sort(blocklut.astype(np.float32))

# ---- Training helpers ----
def train_kmeans(data, n_centroids=N_CENTROIDS, init=None, sample_weights=None, sample_size=500000):
    """Train K-means LUT on data."""
    n = len(data)
    if n > sample_size:
        if sample_weights is not None:
            # Weighted sampling
            sw = sample_weights / sample_weights.sum()
            idx = np.random.RandomState(42).choice(n, sample_size, replace=False, p=sw)
        else:
            idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
        data_subset = data[idx].reshape(-1, 1)
    else:
        data_subset = data.reshape(-1, 1)

    init_array = init.reshape(-1, 1) if init is not None else 'k-means++'
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, init=init_array,
                              random_state=42, batch_size=4096, n_init=1, max_iter=100)
    kmeans.fit(data_subset)
    return np.sort(kmeans.cluster_centers_.ravel().astype(np.float32))

def nf_reconstruction_levels(p, n_levels=N_CENTROIDS):
    """Generate NF reconstruction levels for a given power parameter."""
    # Uniformly spaced in transformed space, then inverse transform
    transformed = np.linspace(0, 1, n_levels, dtype=np.float64)
    levels = transformed ** (1.0 / p)
    # Map to [-1, 1]
    levels = np.concatenate([-levels[::-1], [0.0], levels])
    # Take the right half + 0, then mirror
    half = n_levels // 2 + 1
    pos_levels = transformed[1:] ** (1.0 / p)  # exclude 0
    # Symmetric: -pos_levels reversed, 0, pos_levels
    result = np.zeros(n_levels, dtype=np.float32)
    mid = n_levels // 2
    for i in range(mid):
        result[i] = -(pos_levels[mid - 1 - i])
    result[mid] = 0.0
    for i in range(mid + 1, n_levels):
        result[i] = pos_levels[i - mid - 1]
    return result

def compute_full_psnr(tensor, lut_f32):
    """Full pipeline PSNR for a given LUT."""
    w = tensor.float().ravel().numpy()
    n = w.size
    nb = (n + BS - 1) // BS
    pad = nb * BS - n
    if pad > 0:
        w = np.pad(w, (0, pad))
    blocks = w.reshape(nb, BS)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    normalized = blocks / absmax[:, np.newaxis]

    midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
    idx = np.searchsorted(midpoints, normalized.ravel()).clip(0, len(lut_f32)-1).astype(np.uint8)
    reco_norm = lut_f32[idx].reshape(nb, BS)

    # bf16 recovery (matching CUDA kernel)
    reco_bf16 = torch.from_numpy(reco_norm).to(torch.bfloat16).float()
    absmax_bf16 = torch.from_numpy(absmax).to(torch.bfloat16).float()
    reco = (reco_bf16 * absmax_bf16.unsqueeze(1)).to(torch.bfloat16)
    reco = reco.numpy().ravel()[:n]

    mse = ((tensor.float().numpy().ravel() - reco.astype(np.float32)) ** 2).mean()
    var = tensor.float().numpy().ravel().var()
    return 10 * np.log10(var / mse) if mse > 0 else 99.0

# ============================================================
# 策略1: 加权 K-means (0附近的数据点采样权重更高)
# ============================================================
print("\n" + "="*60)
print("策略1: 加权 K-means (0附近高权重)")
weights_exp = []  # different weight functions
labels = []

# Weight function: w(x) = 1 / (|x| + epsilon)
for eps, label in [(0.01, "w=1/(|x|+0.01)"), (0.05, "w=1/(|x|+0.05)"), (0.001, "w=1/(|x|+0.001)")]:
    w = 1.0 / (np.abs(all_norm) + eps)
    total_w = w.sum()
    w = w / total_w
    labels.append(label)
    weights_exp.append(w)

# Weight function: w(x) = exp(-|x|/sigma)
for sigma, label in [(0.1, "w=exp(-|x|/0.1)"), (0.2, "w=exp(-|x|/0.2)"), (0.05, "w=exp(-|x|/0.05)")]:
    w = np.exp(-np.abs(all_norm) / sigma)
    w = w / w.sum()
    labels.append(label)
    weights_exp.append(w)

for i, (w, label) in enumerate(zip(weights_exp, labels)):
    t0 = time.perf_counter()
    lut = train_kmeans(all_norm, sample_weights=w)
    t = time.perf_counter() - t0
    print(f"  [{label}] train={t:.1f}s  range=[{lut[0]:.4f},{lut[-1]:.4f}]", end="")
    # Quick PSNR test on 3 experts
    test_k = [sorted(weights.keys())[0], sorted(weights.keys())[len(weights)//2], sorted(weights.keys())[-1]]
    psnrs = [compute_full_psnr(weights[tk], lut) for tk in test_k]
    print(f"  PSNR={np.mean(psnrs):.2f} dB")

# ============================================================
# 策略2: NF 初始化 K-means
# ============================================================
print("\n" + "="*60)
print("策略2: NF初始化 K-means (NF重建值作为K-means初始质心)")

# Generate NF reconstruction levels for different pow parameters
for p in [0.4, 0.5, 0.6, 0.7]:
    nf_init = nf_reconstruction_levels(p)
    t0 = time.perf_counter()
    lut = train_kmeans(all_norm, init=nf_init)
    t = time.perf_counter() - t0
    print(f"  [NF p={p} init] train={t:.1f}s  range=[{lut[0]:.4f},{lut[-1]:.4f}]", end="")
    test_k = [sorted(weights.keys())[0], sorted(weights.keys())[len(weights)//2], sorted(weights.keys())[-1]]
    psnrs = [compute_full_psnr(weights[tk], lut) for tk in test_k]
    print(f"  PSNR={np.mean(psnrs):.2f} dB")

# ============================================================
# 策略3: Per-layer LUT
# ============================================================
print("\n" + "="*60)
print("策略3: Per-layer LUT")

# Pick test layers
test_layers = [0, 6, 12, 18, 23]
layer_psnrs_global = {lid: [] for lid in test_layers}
layer_psnrs_per_layer = {lid: [] for lid in test_layers}

for lid in test_layers:
    # Train per-layer LUT
    layer_norm = np.concatenate(layer_data[lid]).astype(np.float32)
    t0 = time.perf_counter()
    per_layer_lut = train_kmeans(layer_norm)
    t = time.perf_counter() - t0

    # Evaluate on experts from this layer
    prefix = f"model.layers.{lid}."
    layer_keys = [k for k in sorted(weights.keys()) if k.startswith(prefix)][:5]

    psnr_g = [compute_full_psnr(weights[k], blocklut) for k in layer_keys]
    psnr_pl = [compute_full_psnr(weights[k], per_layer_lut) for k in layer_keys]

    diff = np.mean(psnr_pl) - np.mean(psnr_g)
    print(f"  Layer {lid:2d}: global={np.mean(psnr_g):.2f}  per-layer={np.mean(psnr_pl):.2f}  diff={diff:+.2f} dB")

# ============================================================
# 最终对比: 最佳方案 vs baseline vs NF
# ============================================================
print("\n" + "="*60)
print("最终对比 (6 test experts across layers)")
test_layers_final = [0, 4, 8, 12, 16, 22]
test_keys_final = []
for lid in test_layers_final:
    prefix = f"model.layers.{lid}.mlp.experts.0."
    for k in sorted(weights.keys()):
        if k.startswith(prefix):
            test_keys_final.append(k)
            break

# Train best weighted K-means (w=exp(-|x|/0.1))
best_weight = np.exp(-np.abs(all_norm) / 0.1)
best_weight = best_weight / best_weight.sum()
best_weighted_lut = train_kmeans(all_norm, sample_weights=best_weight)

# Train best NF-init K-means (p=0.5 init)
nf_init_best = nf_reconstruction_levels(0.5)
best_nfinit_lut = train_kmeans(all_norm, init=nf_init_best)

methods = {
    "Uniform (linspace)": np.linspace(-1, 1, 256).astype(np.float32),
    "BlockLUT (K-means global)": blocklut,
    "K-means + 加权采样": best_weighted_lut,
    "K-means + NF初始化": best_nfinit_lut,
}
# NF pow scan
for p_val in [0.4, 0.5, 0.6, 0.7]:
    methods[f"NF pow={p_val}"] = None  # handled separately

print(f"\n{'Method':<30} {'PSNR_avg':>8} {'PSNR_min':>8}")
print("-" * 50)
for name, lut in methods.items():
    if "NF" in name:
        p_val = float(name.split("=")[-1])
        # NF: compute without LUT
        def nf_psnr(t, p=p_val):
            w = t.float().ravel().numpy()
            n = w.size
            nb = (n + BS - 1) // BS
            pad = nb * BS - n
            if pad > 0:
                w = np.pad(w, (0, pad))
            blocks = w.reshape(nb, BS)
            am = np.max(np.abs(blocks), axis=1)
            am = np.maximum(am, 1e-12)
            norm = blocks / am[:, np.newaxis]
            sign = np.sign(norm)
            abs_norm = np.abs(norm)
            idx = (abs_norm ** p * 255 + 0.5).clip(0, 255).astype(np.uint8)
            dequant = (idx.astype(np.float32) / 255.0) ** (1.0 / p)
            reco_norm = sign * dequant
            reco_bf16 = torch.from_numpy(reco_norm).to(torch.bfloat16).float()
            am_bf16 = torch.from_numpy(am).to(torch.bfloat16).float()
            reco = (reco_bf16 * am_bf16.unsqueeze(1)).to(torch.bfloat16).numpy().ravel()[:n]
            mse = ((t.float().numpy().ravel() - reco.astype(np.float32)) ** 2).mean()
            return 10 * np.log10(t.float().numpy().ravel().var() / mse)
        psnrs = [nf_psnr(weights[k]) for k in test_keys_final]
    else:
        psnrs = [compute_full_psnr(weights[k], lut) for k in test_keys_final]
    print(f"{name:<30} {np.mean(psnrs):8.2f} {min(psnrs):8.2f}")

print("\nDone.")
