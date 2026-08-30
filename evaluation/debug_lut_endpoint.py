"""
Block-normalized LUT 优化：强制 LUT 端点为 ±1，消除端点误差。
"""
import os, sys, time
import numpy as np
import torch
import faiss
from safetensors import safe_open

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"

safetensor_files = sorted([
    os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
    if f.endswith(".safetensors")
])

def load_all_expert_tensors():
    tensors = []
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    t = f.get_tensor(k).to(torch.bfloat16)
                    tensors.append((k, t))
    return tensors

def var_psnr(orig, reco):
    orig = orig.float().numpy().ravel()
    reco = reco.float().numpy().ravel()
    mse = ((orig - reco) ** 2).mean()
    var = orig.var()
    return 10 * np.log10(var / mse) if mse > 0 else float('inf')

def kmeans_lut_endpoint_fixed(values, K=256, seed=42):
    """
    K-means on 1D values, forcing endpoints at -1 and +1.
    Interior K-2 centroids learned via K-means on data excluding ±1.
    """
    data = np.ascontiguousarray(values.astype(np.float32)).reshape(-1, 1)
    km = faiss.Kmeans(d=1, k=K-2, seed=seed, niter=30, verbose=False)
    km.train(data)
    centroids = km.centroids.ravel()
    centroids.sort()
    # Add endpoints
    lut = np.concatenate([[-1.0], centroids, [1.0]])
    return lut.astype(np.float32)

def quantize_1d(values, lut):
    """Nearest-neighbor quantization with given LUT, returns int64 indices"""
    data = np.ascontiguousarray(values.astype(np.float32)).reshape(-1, 1)
    lut_data = np.ascontiguousarray(lut.astype(np.float32)).reshape(-1, 1)
    index = faiss.IndexFlatL2(1)
    index.add(lut_data)
    _, indices = index.search(data, 1)
    return indices.ravel()  # int64


def eval_block_lut_variants(weight, block_size=128):
    """测试多种 LUT 方案"""
    M, N = weight.shape
    flat = weight.float().numpy().ravel()
    n_elements = len(flat)

    pad = (block_size - n_elements % block_size) % block_size
    if pad > 0:
        flat_padded = np.pad(flat, (0, pad), mode='constant', constant_values=0)
    else:
        flat_padded = flat
    n_blocks = len(flat_padded) // block_size
    blocks = flat_padded.reshape(n_blocks, block_size)

    absmax = np.abs(blocks).max(axis=1, keepdims=True)
    absmax = np.maximum(absmax, 1e-8)
    normalized = blocks / absmax
    norm_flat = normalized.ravel()

    results = {}

    # A: Uniform quantization (baseline)
    # idx = round((v + 1) * 127.5), clamp to [0, 255]
    idx_uniform = np.clip(np.round((norm_flat + 1.0) * 127.5), 0, 255).astype(np.uint8)
    reco_uniform_norm = (idx_uniform.astype(np.float32) / 127.5 - 1.0)
    reco_uniform = reco_uniform_norm.reshape(n_blocks, block_size) * absmax
    reco_uniform_flat = reco_uniform.ravel()[:n_elements]
    reco = torch.from_numpy(reco_uniform_flat.reshape(M, N)).to(torch.bfloat16)
    psnr = var_psnr(weight, reco)
    total_bits = n_blocks * 16 + n_elements * 8
    results['uniform'] = {'psnr': psnr, 'bpe': total_bits / n_elements}

    # B1: free K-means LUT
    data = np.ascontiguousarray(norm_flat.reshape(-1, 1).astype(np.float32))
    km = faiss.Kmeans(d=1, k=256, seed=42, niter=30, verbose=False)
    km.train(data)
    lut_free = np.sort(km.centroids.ravel()).astype(np.float32)
    idx_free = quantize_1d(norm_flat, lut_free)
    reco_free_norm = lut_free[idx_free]
    reco_free = reco_free_norm.reshape(n_blocks, block_size) * absmax
    reco_free_flat = reco_free.ravel()[:n_elements]
    reco = torch.from_numpy(reco_free_flat.reshape(M, N)).to(torch.bfloat16)
    psnr = var_psnr(weight, reco)
    results['lut_free'] = {'psnr': psnr, 'bpe': (n_blocks * 16 + n_elements * 8) / n_elements}

    # B2: fixed endpoints
    km2 = faiss.Kmeans(d=1, k=254, seed=42, niter=30, verbose=False)
    km2.train(data)
    lut_fixed = np.sort(np.concatenate([[-1.0], km2.centroids.ravel(), [1.0]])).astype(np.float32)
    idx_fixed = quantize_1d(norm_flat, lut_fixed)
    reco_fixed_norm = lut_fixed[idx_fixed]
    reco_fixed = reco_fixed_norm.reshape(n_blocks, block_size) * absmax
    reco_fixed_flat = reco_fixed.ravel()[:n_elements]
    reco = torch.from_numpy(reco_fixed_flat.reshape(M, N)).to(torch.bfloat16)
    psnr = var_psnr(weight, reco)
    results['lut_fixed'] = {'psnr': psnr, 'bpe': (n_blocks * 16 + n_elements * 8) / n_elements}

    return results

def main():
    print("=" * 70)
    print("Block-normalized LUT: uniform vs free LUT vs fixed-endpoint LUT")
    print("=" * 70)

    print("\n加载 Qwen expert 权重...")
    all_tensors = load_all_expert_tensors()
    print(f"共 {len(all_tensors)} 个 expert tensor")

    n_sample = 30
    sampled_indices = np.linspace(0, len(all_tensors) - 1, n_sample, dtype=int)
    sampled = [all_tensors[i] for i in sampled_indices]

    all_results = {'uniform': [], 'lut_free': [], 'lut_fixed': []}

    for idx, (name, weight) in enumerate(sampled):
        results = eval_block_lut_variants(weight)
        for method, r in results.items():
            all_results[method].append(r)
        tag = name.split('.')[-3] + '.' + name.split('.')[-2] + '.' + name.split('.')[-1]
        print(f"  [{idx+1:2d}/30] {tag} shape={list(weight.shape)}: "
              f"uniform={results['uniform']['psnr']:.2f} dB, "
              f"LUT_free={results['lut_free']['psnr']:.2f} dB, "
              f"LUT_fixed={results['lut_fixed']['psnr']:.2f} dB")

    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'方案':<35} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10} {'bits/elem':>10}")
    print("-" * 77)
    for method, label in [('uniform', 'Block-uniform (baseline)'),
                           ('lut_free', 'Block + free LUT K=256'),
                           ('lut_fixed', 'Block + fixed-endpoint LUT K=256')]:
        psnrs = [r['psnr'] for r in all_results[method] if not np.isinf(r['psnr'])]
        bpe = np.mean([r['bpe'] for r in all_results[method]])
        print(f"{label:<35} {np.mean(psnrs):>10.2f} {np.min(psnrs):>10.2f} {np.max(psnrs):>10.2f} {bpe:>10.2f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
