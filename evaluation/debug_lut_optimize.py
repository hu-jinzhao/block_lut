"""
优化元素级 LUT 的三个方向对比：
1. Per-tensor LUT: 每个权重矩阵独立 256-entry K-means
2. Per-block normalized LUT: block 归一化后用全局 LUT 量化
3. 原始全局 LUT (baseline)
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

def kmeans_lut(values, K=256, seed=42):
    """对一维标量值做 K-means，返回 sorted LUT"""
    data = np.ascontiguousarray(values.ravel().astype(np.float32)).reshape(-1, 1)
    km = faiss.Kmeans(d=1, k=K, seed=seed, niter=30, verbose=False)
    km.train(data)
    centroids = km.centroids.ravel()
    centroids.sort()
    return centroids

def quantize_with_lut(values, lut):
    """用 LUT 量化标量值，返回 indices 和 recovered"""
    flat = values.float().numpy().ravel()
    # 对每个值找最近的 LUT entry
    indices = np.zeros(len(flat), dtype=np.uint8)
    recovered = np.zeros(len(flat), dtype=np.float32)
    for i in range(len(lut)):
        # 批量赋值太慢，用 searchsorted 的方式更高效
        pass
    # 高效方法：用 faiss search
    data = np.ascontiguousarray(flat.reshape(-1, 1).astype(np.float32))
    lut_data = np.ascontiguousarray(lut.reshape(-1, 1).astype(np.float32))
    index = faiss.IndexFlatL2(1)
    index.add(lut_data)
    _, indices = index.search(data, 1)
    indices = indices.ravel().astype(np.uint8)
    recovered = lut[indices]
    return indices, recovered

# ---- 方案 1: Per-tensor LUT ----
def eval_per_tensor_lut(weight, K=256):
    """每个 tensor 独立计算 LUT"""
    M, N = weight.shape
    flat = weight.float().numpy().ravel()
    lut = kmeans_lut(flat, K=K)
    data = np.ascontiguousarray(flat.reshape(-1, 1).astype(np.float32))
    lut_data = np.ascontiguousarray(lut.reshape(-1, 1).astype(np.float32))
    index = faiss.IndexFlatL2(1)
    index.add(lut_data)
    _, indices = index.search(data, 1)
    indices = indices.ravel().astype(np.uint8)
    recovered = torch.from_numpy(lut[indices].reshape(M, N)).to(torch.bfloat16)
    psnr = var_psnr(weight, recovered)
    # bits/elem: LUT itself is 256*16 bits, negligible per-element for large tensors
    lut_bits = K * 16
    idx_bits = M * N * 8
    bits_per_elem = (lut_bits + idx_bits) / (M * N)
    return psnr, bits_per_elem

# ---- 方案 2: Per-block normalized LUT ----
def eval_block_normalized_lut(weight, block_size=128, K=256):
    """
    每个 block 独立归一化(absmax)，然后用全局 LUT 量化归一化值。
    需要先算好全局 LUT（在归一化后的分布上训练）。
    """
    M, N = weight.shape
    flat = weight.float().numpy().ravel()
    n_elements = len(flat)

    # Pad to block_size multiple
    pad = (block_size - n_elements % block_size) % block_size
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode='constant', constant_values=0)

    n_blocks = len(flat) // block_size
    blocks = flat.reshape(n_blocks, block_size)

    # Per-block absmax
    absmax = np.abs(blocks).max(axis=1, keepdims=True)  # (n_blocks, 1)
    absmax = np.maximum(absmax, 1e-8)  # avoid div by zero
    normalized = blocks / absmax  # values in [-1, 1]

    # Train LUT on normalized data (this is "cheating" - would need to store or share LUT)
    # For fair comparison, train one LUT on all normalized blocks of this tensor
    norm_flat = normalized.ravel()
    lut = kmeans_lut(norm_flat, K=K)
    data = np.ascontiguousarray(norm_flat.reshape(-1, 1).astype(np.float32))
    lut_data = np.ascontiguousarray(lut.reshape(-1, 1).astype(np.float32))
    index = faiss.IndexFlatL2(1)
    index.add(lut_data)
    _, indices = index.search(data, 1)
    indices = indices.ravel().astype(np.uint8)

    recovered_norm = lut[indices].reshape(n_blocks, block_size)
    recovered_blocks = recovered_norm * absmax
    recovered_flat = recovered_blocks.ravel()[:n_elements]
    recovered = torch.from_numpy(recovered_flat.reshape(M, N)).to(torch.bfloat16)

    psnr = var_psnr(weight, recovered)
    # Storage: n_blocks * 16 bits (absmax) + n_elements * 8 bits (indices) + K*16 bits (LUT)
    total_bits = n_blocks * 16 + n_elements * 8 + K * 16
    bits_per_elem = total_bits / n_elements
    return psnr, bits_per_elem

# ---- 方案 3: Per-layer LUT (同一层所有 expert 共享一个 LUT) ----
# 这个在采样时做不了，需要在加载时按层分组

def main():
    print("=" * 70)
    print("元素级 LUT 优化方向对比")
    print("=" * 70)

    print("\n加载 Qwen expert 权重...")
    all_tensors = load_all_expert_tensors()
    print(f"共 {len(all_tensors)} 个 expert tensor")

    # 采样 30 个 tensor
    n_sample = 30
    sampled_indices = np.linspace(0, len(all_tensors) - 1, n_sample, dtype=int)
    sampled = [all_tensors[i] for i in sampled_indices]

    # 收集结果
    results_ptl = []  # per-tensor LUT
    results_bnl = []  # block-normalized LUT

    print("\n--- 方案 1: Per-tensor LUT (K=256) ---")
    for idx, (name, weight) in enumerate(sampled):
        psnr, bpe = eval_per_tensor_lut(weight, K=256)
        results_ptl.append({'name': name, 'psnr': psnr, 'bpe': bpe})
        print(f"  [{idx+1:2d}/30] {name.split('.')[-3]}.{name.split('.')[-2]}.{name.split('.')[-1]} "
              f"shape={list(weight.shape)}: PSNR={psnr:.2f} dB, {bpe:.2f} bits/elem")

    print("\n--- 方案 2: Block-normalized LUT (block=128, K=256) ---")
    for idx, (name, weight) in enumerate(sampled):
        psnr, bpe = eval_block_normalized_lut(weight, block_size=128, K=256)
        results_bnl.append({'name': name, 'psnr': psnr, 'bpe': bpe})
        print(f"  [{idx+1:2d}/30] {name.split('.')[-3]}.{name.split('.')[-2]}.{name.split('.')[-1]} "
              f"shape={list(weight.shape)}: PSNR={psnr:.2f} dB, {bpe:.2f} bits/elem")

    # 汇总对比
    print("\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'方案':<35} {'PSNR mean':>10} {'PSNR min':>10} {'PSNR max':>10} {'bits/elem':>10}")
    print("-" * 77)

    ptl_psnrs = [r['psnr'] for r in results_ptl if not np.isinf(r['psnr'])]
    bnl_psnrs = [r['psnr'] for r in results_bnl if not np.isinf(r['psnr'])]

    print(f"{'全局 LUT (256 centroids)':<35} {'~31.3':>10} {'~14.5':>10} {'~37.1':>10} {'8.00':>10}")
    print(f"{'Block128 uniform':<35} {'~43.5':>10} {'~42.5':>10} {'~44.3':>10} {'8.12':>10}")
    print(f"{'Per-tensor LUT K=256':<35} {np.mean(ptl_psnrs):>10.2f} {np.min(ptl_psnrs):>10.2f} {np.max(ptl_psnrs):>10.2f} {np.mean([r['bpe'] for r in results_ptl]):>10.2f}")
    print(f"{'Block-normalized LUT K=256':<35} {np.mean(bnl_psnrs):>10.2f} {np.min(bnl_psnrs):>10.2f} {np.max(bnl_psnrs):>10.2f} {np.mean([r['bpe'] for r in results_bnl]):>10.2f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
