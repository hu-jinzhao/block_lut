"""
行/列 K-means 聚类压缩对比：对权重矩阵的行向量或列向量做 K-means 聚类。
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

def compute_psnr(original, reconstructed):
    orig = original.float().numpy().ravel()
    reco = reconstructed.float().numpy().ravel()
    mse = ((orig - reco) ** 2).mean()
    var = orig.var()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(var / mse)

def evaluate_kmeans(weight, K, axis='row', random_state=42):
    """
    Args:
        weight: torch.Tensor, shape (M, N)
        K: 聚类数
        axis: 'row' 对行聚类, 'col' 对列聚类
    """
    M, N = weight.shape

    if axis == 'row':
        vectors = np.ascontiguousarray(weight.float().numpy())  # (M, N)
        dim = N
        n_vectors = M
    else:  # 'col'
        vectors = np.ascontiguousarray(weight.float().numpy().T)  # (N, M)
        dim = M
        n_vectors = N

    t0 = time.time()
    kmeans = faiss.Kmeans(d=dim, k=K, seed=random_state, niter=50, verbose=False)
    kmeans.train(vectors)
    centroids = kmeans.centroids  # (K, dim)
    _, indices = kmeans.index.search(vectors, 1)
    indices = indices.ravel().astype(np.int64)
    elapsed = time.time() - t0

    if axis == 'row':
        recovered = torch.from_numpy(centroids[indices]).to(torch.bfloat16)  # (M, N)
    else:
        recovered = torch.from_numpy(centroids[indices].T).to(torch.bfloat16)  # (N, M) -> (M, N)

    psnr = compute_psnr(weight, recovered)

    # bits_per_index rounds up to full bytes
    bits_per_index = np.ceil(np.log2(K))
    if axis == 'row':
        total_bits = K * N * 16 + M * bits_per_index
    else:
        total_bits = K * M * 16 + N * bits_per_index
    bits_per_elem = total_bits / (M * N)

    return {
        'psnr': psnr,
        'bits_per_elem': bits_per_elem,
        'K': K,
        'axis': axis,
        'shape': (M, N),
        'time_s': elapsed,
    }

def main():
    print("=" * 70)
    print("行/列 K-means 聚类压缩对比 (faiss)")
    print("=" * 70)

    print("\n加载 Qwen expert 权重...")
    all_tensors = load_all_expert_tensors()
    print(f"共 {len(all_tensors)} 个 expert tensor")

    n_sample = 30
    sampled_indices = np.linspace(0, len(all_tensors) - 1, n_sample, dtype=int)
    sampled = [all_tensors[i] for i in sampled_indices]

    K_values = [64, 128, 256, 512]
    axes = ['row', 'col']

    all_results = {(axis, K): [] for axis in axes for K in K_values}

    for idx, (name, weight) in enumerate(sampled):
        M, N = weight.shape
        print(f"\n[{idx+1}/{n_sample}] {name} shape=({M}, {N})")

        for axis in axes:
            for K in K_values:
                result = evaluate_kmeans(weight, K, axis=axis)
                all_results[(axis, K)].append(result)
                tag = "行" if axis == 'row' else "列"
                print(f"  {tag} K={K:4d}: PSNR={result['psnr']:.2f} dB, "
                      f"{result['bits_per_elem']:.2f} bits/elem, "
                      f"{result['time_s']:.1f}s")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    print(f"{'方向':<6} {'K':<6} {'PSNR mean':>10} {'PSNR median':>10} {'PSNR min':>10} {'PSNR max':>10} {'bits/elem':>10}")
    print("-" * 60)

    for axis in axes:
        for K in K_values:
            psnrs = [r['psnr'] for r in all_results[(axis, K)] if not np.isinf(r['psnr'])]
            bits = np.mean([r['bits_per_elem'] for r in all_results[(axis, K)]])
            tag = "行" if axis == 'row' else "列"
            print(f"{tag:<6} {K:<6} {np.mean(psnrs):>10.2f} {np.median(psnrs):>10.2f} "
                  f"{np.min(psnrs):>10.2f} {np.max(psnrs):>10.2f} {bits:>10.2f}")

    # 对比
    print("\n" + "=" * 70)
    print("与历史方案对比")
    print("=" * 70)
    print(f"{'方案':<30} {'PSNR':>8} {'bits/elem':>10}")
    print("-" * 50)
    print(f"{'全局 LUT (256 centroids)':<30} {'~31 dB':>8} {'8.00':>10}")
    print(f"{'Block128 uniform':<30} {'~43.5 dB':>8} {'8.12':>10}")
    for axis in axes:
        for K in K_values:
            psnrs = [r['psnr'] for r in all_results[(axis, K)] if not np.isinf(r['psnr'])]
            mean_psnr = np.mean(psnrs)
            mean_bits = np.mean([r['bits_per_elem'] for r in all_results[(axis, K)]])
            tag = "行" if axis == 'row' else "列"
            print(f"{tag + ' K-means K=' + str(K):<30} {mean_psnr:>8.2f} {mean_bits:>10.2f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
