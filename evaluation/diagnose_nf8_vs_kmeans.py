"""
诊断: 为什么 8-bit NF 比 8-bit 加权 K-means 效果好?

NF (NormalFloat) 方法:
  1. 对归一化值做 power transform: y = sign(x) * |x|^p  (p<1, 如 0.6)
  2. 在 y 空间做均匀量化
  3. 反变换: x' = sign(y_q) * |y_q|^(1/p)
  → 效果: 0 附近量化级别更密, 但范围保持 [-1, 1]

加权 K-means 方法:
  1. 按权重函数对样本做 sample replication
  2. 在加权样本上做 K-means 聚类
  → 效果: 质心集中在高密度区域, 但尾部质心可能偏离 ±1

假设: 加权 K-means 的尾部质心内缩导致大值量化误差爆炸,
     而 NF 通过 power transform 保持 [-1,1] 范围同时加密中心。
"""

import os, math, time
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from sklearn.cluster import KMeans

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
BS = 128
N_CENTROIDS = 256  # 8-bit
POW = 0.6


# ═══════════════════════════════════════════════════════════
# NF (NormalFloat) 量化 — 直接逐元素 power-transform 量化
#
# 步骤:
#   1. block absmax 归一化: x_norm = x / absmax ∈ [-1, 1]
#   2. power transform: y = sign(x_norm) * |x_norm|^p  (p<1, 压缩大值)
#   3. 在 y 空间均匀量化: y_q = round(y * max_val) / max_val
#   4. 逆变换: x' = sign(y_q) * |y_q|^(1/p)
#   5. 恢复: x_recon = x' * absmax
#
# 关键: 端点精确在 ±1, 因为 |±1|^p = 1, y_q_max = ±1, 1^(1/p) = 1
# ═══════════════════════════════════════════════════════════

def nf_quantize_block(tensor_flat, p, nbits=8, block_size=BS):
    """Direct per-element NF quantization."""
    max_val = 2**(nbits-1) - 1  # 127 for 8-bit

    n = len(tensor_flat)
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad:
        tensor_flat = np.pad(tensor_flat.astype(np.float32), (0, pad))
    else:
        tensor_flat = tensor_flat.astype(np.float32)

    blocks = tensor_flat.reshape(nb, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    norm = blocks / absmax[:, np.newaxis]

    # Power transform: compress large values toward 0 (in y-space)
    sign = np.sign(norm)
    y = sign * np.abs(norm) ** p

    # Uniform quantize in power-transformed space
    y_q = np.clip(np.round(y * max_val), -max_val, max_val).astype(np.int8)

    return y_q.view(np.uint8).ravel(), absmax.astype(np.float32)


def nf_dequantize_block(indices, absmax, p, nbits, block_size, orig_len):
    """NF dequantization via inverse power transform."""
    max_val = 2**(nbits-1) - 1

    nb = len(absmax)
    indices = indices.reshape(nb, block_size)
    y_q = indices.view(np.int8).astype(np.float32)

    # Inverse power transform
    sign = np.sign(y_q)
    norm_recon = sign * (np.abs(y_q) / max_val) ** (1.0 / p)

    x = (norm_recon * absmax[:, np.newaxis]).ravel()
    return x[:orig_len]


def nf_get_effective_lut(p, nbits=8):
    """获取 NF 的实际重构值 (用于对比分析).

    对于 y_q ∈ {-127..127}, 计算 x' = sign(y_q) * |y_q/127|^(1/p)
    这些是 NF 方法实际产生的所有可能重构值.
    """
    max_val = 2**(nbits-1) - 1
    y_pos = np.arange(max_val + 1) / max_val  # [0, 1/127, ..., 1]
    x_pos = y_pos ** (1.0 / p)
    return np.concatenate([-x_pos[-1:0:-1], x_pos]).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# K-means LUT
# ═══════════════════════════════════════════════════════════

def build_kmeans_lut(values, n_centroids, random_state=42):
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    km = KMeans(n_clusters=n_centroids, random_state=random_state,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(values.reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def build_weighted_kmeans_lut(values, n_centroids, weight_func):
    if len(values) > 500000:
        values = np.random.choice(values, 500000, replace=False)
    w = weight_func(values)
    w = np.maximum(w, 1e-10)
    probs = w / w.sum()
    idx = np.random.choice(len(values), size=200000, replace=True, p=probs)
    km = KMeans(n_clusters=n_centroids, random_state=42,
                n_init=3, max_iter=100, tol=1e-5)
    km.fit(values[idx].reshape(-1, 1).astype(np.float32))
    return np.sort(km.cluster_centers_.ravel()).astype(np.float32)


def find_nearest_vec(values, centroids):
    idx = np.searchsorted(centroids, values)
    idx = np.clip(idx, 1, len(centroids) - 1)
    left = np.abs(values - centroids[idx - 1])
    right = np.abs(values - centroids[idx])
    return np.where(left <= right, idx - 1, idx)


def lut_quantize_block(tensor_flat, lut, block_size=BS):
    n = len(tensor_flat)
    nb = (n + block_size - 1) // block_size
    pad = nb * block_size - n
    if pad:
        tensor_flat = np.pad(tensor_flat.astype(np.float32), (0, pad))
    else:
        tensor_flat = tensor_flat.astype(np.float32)
    blocks = tensor_flat.reshape(nb, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    norm = blocks / absmax[:, np.newaxis]
    idx = find_nearest_vec(norm.ravel(), lut)
    return idx.astype(np.uint8), absmax.astype(np.float32)


def lut_dequantize_block(indices, absmax, lut, block_size, orig_len):
    nb = len(absmax)
    indices = indices.reshape(nb, block_size)
    x = (lut[indices.astype(np.int32)] * absmax[:, np.newaxis]).ravel()
    return x[:orig_len]


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def psnr(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    var = np.var(orig)
    return float('inf') if mse == 0 else 10 * math.log10(var / mse)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    sft_files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
                        if f.startswith("model-") and f.endswith(".safetensors")])

    # ── 1. Collect training data ──
    print("=" * 90)
    print("[1/4] 收集归一化训练数据...")
    t0 = time.time()
    np.random.seed(42)
    all_vals = []
    for path in tqdm(sft_files[:2], desc="  Collecting"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" not in k or "shared_expert" in k:
                    continue
                t = f.get_tensor(k).to(torch.float32).numpy().ravel()
                nb = (len(t) + BS - 1) // BS
                pad = nb * BS - len(t)
                if pad:
                    t = np.pad(t, (0, pad))
                sample_nb = min(nb, 20)
                idx = np.random.choice(nb, sample_nb, replace=False)
                for b in idx:
                    s, e = b * BS, (b + 1) * BS
                    block = t[s:e]
                    amax = np.max(np.abs(block))
                    if amax < 1e-12:
                        continue
                    all_vals.append(block / amax)
    train_vals = np.concatenate(all_vals)
    sigma = train_vals.std()
    print(f"  {len(train_vals):,} values, std={sigma:.4f}, time={time.time()-t0:.1f}s")
    print(f"  分布: |x|<0.1: {np.mean(np.abs(train_vals)<0.1)*100:.1f}%, "
          f"|x|<0.25: {np.mean(np.abs(train_vals)<0.25)*100:.1f}%, "
          f"|x|>0.9: {np.mean(np.abs(train_vals)>0.9)*100:.1f}%")

    # ── 2. Build LUTs ──
    print("\n[2/4] 构建量化表...")

    # NF8 effective LUT (derived analytically from power transform, no data needed)
    nf8_lut = nf_get_effective_lut(POW)

    # Standard K-means
    t0 = time.time()
    km_lut = build_kmeans_lut(train_vals, N_CENTROIDS)
    print(f"  std-KM:     range=[{km_lut[0]:.4f}, {km_lut[-1]:.4f}], "
          f"near0(32)={(np.abs(km_lut)<0.1).sum()}, time={time.time()-t0:.1f}s")

    # Weighted K-means (Gaussian weight)
    t0 = time.time()
    w_gauss = lambda x: np.exp(-x**2 / (2 * sigma**2))
    wkm_gauss_lut = build_weighted_kmeans_lut(train_vals, N_CENTROIDS, w_gauss)
    print(f"  wKM-gauss:  range=[{wkm_gauss_lut[0]:.4f}, {wkm_gauss_lut[-1]:.4f}], "
          f"near0(32)={(np.abs(wkm_gauss_lut)<0.1).sum()}, time={time.time()-t0:.1f}s")

    # Weighted K-means (Laplace weight)
    t0 = time.time()
    w_laplace = lambda x: np.exp(-np.abs(x) / 0.25)
    wkm_laplace_lut = build_weighted_kmeans_lut(train_vals, N_CENTROIDS, w_laplace)
    print(f"  wKM-laplace: range=[{wkm_laplace_lut[0]:.4f}, {wkm_laplace_lut[-1]:.4f}], "
          f"near0(32)={(np.abs(wkm_laplace_lut)<0.1).sum()}, time={time.time()-t0:.1f}s")

    # Print LUT endpoint details
    print(f"\n  NF8 eff-LUT: range=[{nf8_lut[0]:.4f}, {nf8_lut[-1]:.4f}], "
          f"entries={len(nf8_lut)}, near0(32)={(np.abs(nf8_lut)<0.1).sum()}")

    # ── 3. Compare centroids spacing ──
    print("\n── 质心间距分析 (各方法重构值分布) ──")
    for name, lut in [("NF8 (eff-LUT)", nf8_lut), ("std-KM", km_lut),
                       ("wKM-gauss", wkm_gauss_lut), ("wKM-laplace", wkm_laplace_lut)]:
        h = len(lut) // 2
        pos_half = lut[h:]  # >=0 的质心
        diffs = np.diff(pos_half)
        print(f"\n  {name} ({len(lut)} entries):")
        print(f"    前10: {[f'{x:.4f}' for x in pos_half[:10]]}")
        print(f"    后10: {[f'{x:.4f}' for x in pos_half[-10:]]}")
        print(f"    最大端点: [{lut[0]:.6f}, {lut[-1]:.6f}]")
        print(f"    最小步长 (近0): {diffs[:5].min():.6f}")
        print(f"    最大步长 (尾部): {diffs[-5:].max():.4f}")
        print(f"    步长比 (max/min): {diffs[-5:].max()/diffs[:5].min():.1f}x")

    # ── 4. Evaluate on expert tensors ──
    print("\n[3/4] 在 expert tensors 上评估...")

    all_tensors = []
    for path in sft_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    all_tensors.append((k, f.get_tensor(k)))
    step = max(1, len(all_tensors) // 80)
    sampled = all_tensors[::step][:80]
    print(f"  采样 {len(sampled)} 个 tensor")

    methods = {
        'NF8 (pow=0.6)': {'type': 'nf', 'psnr': [], 'err_by_range': [],
                           'max_err': [], 'mse_tail': [], 'mse_center': []},
        'std-KM 256': {'type': 'lut', 'lut': km_lut, 'psnr': [], 'err_by_range': [],
                        'max_err': [], 'mse_tail': [], 'mse_center': []},
        'wKM-gauss 256': {'type': 'lut', 'lut': wkm_gauss_lut, 'psnr': [], 'err_by_range': [],
                           'max_err': [], 'mse_tail': [], 'mse_center': []},
        'wKM-laplace 256': {'type': 'lut', 'lut': wkm_laplace_lut, 'psnr': [], 'err_by_range': [],
                             'max_err': [], 'mse_tail': [], 'mse_center': []},
    }

    for name, W_raw in tqdm(sampled, desc="  Evaluating"):
        W = W_raw.to(torch.float32).numpy().ravel()
        orig_len = len(W)

        for method_name, data in methods.items():
            if data['type'] == 'nf':
                idx, am = nf_quantize_block(W, POW)
                recon = nf_dequantize_block(idx, am, POW, 8, BS, orig_len)
            else:
                idx, am = lut_quantize_block(W, data['lut'])
                recon = lut_dequantize_block(idx, am, data['lut'], BS, orig_len)

            data['psnr'].append(psnr(W, recon))

            # Per-element error analysis (in normalized space)
            abs_W = np.abs(W)
            abs_err = np.abs(W - recon)
            rel_err = abs_err / (abs_W + 1e-12)

            # MSE breakdown by value range (in original space, but binned by normalized magnitude)
            # Normalize W by per-block absmax for fair comparison
            n = orig_len
            nb = (n + BS - 1) // BS
            pad = nb * BS - n
            W_pad = np.pad(W, (0, pad)) if pad else W.copy()
            blocks = W_pad.reshape(nb, BS)
            am_block = np.max(np.abs(blocks), axis=1)
            am_block = np.maximum(am_block, 1e-12)
            norm_W = (blocks / am_block[:, np.newaxis]).ravel()[:orig_len]
            recon_pad = np.pad(recon, (0, pad)) if pad else recon.copy()
            recon_blocks = recon_pad.reshape(nb, BS)
            norm_recon = (recon_blocks / am_block[:, np.newaxis]).ravel()[:orig_len]

            # Bin by normalized value magnitude
            abs_norm = np.abs(norm_W)
            bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.0)]
            err_by_range = {}
            for lo, hi in bins:
                mask = (abs_norm >= lo) & (abs_norm < hi)
                if mask.sum() > 0:
                    mse = np.mean((norm_W[mask] - norm_recon[mask])**2)
                    err_by_range[f'[{lo},{hi})'] = (mse, mask.sum())
                else:
                    err_by_range[f'[{lo},{hi})'] = (0, 0)
            data['err_by_range'].append(err_by_range)

            # Tail vs center MSE ratio
            tail_mask = abs_norm > 0.9
            center_mask = abs_norm < 0.1
            if tail_mask.sum() > 0:
                data['mse_tail'].append(np.mean((norm_W[tail_mask] - norm_recon[tail_mask])**2))
            if center_mask.sum() > 0:
                data['mse_center'].append(np.mean((norm_W[center_mask] - norm_recon[center_mask])**2))

            data['max_err'].append(np.max(abs_err))

    # ── 5. Results ──
    print("\n" + "=" * 90)
    print("[4/4] 结果分析")
    print("=" * 90)

    print(f"\n{'Method':<22} {'PSNR':>8} {'MaxErr':>10} {'TailMSE':>10} {'CenterMSE':>12} {'Tail/Center':>12}")
    print("-" * 80)
    for name, data in methods.items():
        v = data['psnr']
        t = data['mse_tail']
        c = data['mse_center']
        avg_tail = np.mean(t) if t else 0
        avg_center = np.mean(c) if c else 0
        ratio = avg_tail / avg_center if avg_center > 0 else float('inf')
        print(f"{name:<22} {np.mean(v):>8.2f} {np.mean(data['max_err']):>10.6f} "
              f"{avg_tail:>10.6f} {avg_center:>12.8f} {ratio:>12.1f}x")

    # Per-range MSE breakdown
    print(f"\n── 归一化空间 MSE 按值范围分解 ──")
    ranges = ['[0,0.1)', '[0.1,0.3)', '[0.3,0.6)', '[0.6,0.9)', '[0.9,1.0)']
    header = f"{'Method':<22}"
    for r in ranges:
        header += f" {r:>14}"
    print(header)
    print("-" * 100)
    for name, data in methods.items():
        row = f"{name:<22}"
        for r in ranges:
            vals = [ebr[r][0] for ebr in data['err_by_range'] if ebr[r][1] > 0]
            mse = np.mean(vals) if vals else 0
            row += f" {mse:>14.8f}"
        print(row)

    # ── 6. Diagnosis ──
    print("\n" + "=" * 90)
    print("诊断结论")
    print("=" * 90)

    nf8_p = np.mean(methods['NF8 (pow=0.6)']['psnr'])
    km_p = np.mean(methods['std-KM 256']['psnr'])
    wkm_g_p = np.mean(methods['wKM-gauss 256']['psnr'])
    wkm_l_p = np.mean(methods['wKM-laplace 256']['psnr'])

    print(f"\n  1. NF8 (pow=0.6):  {nf8_p:.2f} dB  ← 最优")
    print(f"  2. std-KM 256:     {km_p:.2f} dB  (vs NF8: {km_p-nf8_p:+.2f})")
    print(f"  3. wKM-gauss 256:  {wkm_g_p:.2f} dB  (vs NF8: {wkm_g_p-nf8_p:+.2f}, vs std: {wkm_g_p-km_p:+.2f})")
    print(f"  4. wKM-laplace 256:{wkm_l_p:.2f} dB  (vs NF8: {wkm_l_p-nf8_p:+.2f}, vs std: {wkm_l_p-km_p:+.2f})")

    # Key insight
    nf8_tail = np.mean(methods['NF8 (pow=0.6)']['mse_tail'])
    km_tail = np.mean(methods['std-KM 256']['mse_tail'])
    wkm_g_tail = np.mean(methods['wKM-gauss 256']['mse_tail'])

    print(f"\n  尾部 MSE (|x|>0.9) 对比:")
    print(f"    NF8:            {nf8_tail:.6f}")
    print(f"    std-KM:         {km_tail:.6f}  ({km_tail/nf8_tail:.1f}x NF8)")
    print(f"    wKM-gauss:      {wkm_g_tail:.6f}  ({wkm_g_tail/nf8_tail:.1f}x NF8)")

    # Why
    print(f"\n  ── 为什么加权 K-means 不如 NF8 ──")
    print(f"  核心原因: 加权 K-means 通过 sample replication 改变聚类中心,")
    print(f"  但这导致尾部样本被稀释 → 尾部质心向内收缩 → 大值量化误差爆炸")
    print(f"")
    print(f"  NF8 的优势: power transform (x^p) 不改变值域范围 [-1,1],")
    print(f"  只是非线性地压缩大值、拉伸小值, 等效于在 [-1,1] 内非均匀分布量化级别")
    print(f"  端点始终精确在 ±1.0")
    print(f"")
    print(f"  如果你想让 K-means 追上 NF8, 可以考虑:")
    print(f"  1. 约束 K-means: 固定最外层质心在 ±1.0 (之前已试过 constrained wKM)")
    print(f"  2. 用 NF 的逆变换作为 K-means 的特征变换 (先做 power transform 再做均匀 K-means)")
    print(f"  3. 使用 Lloyd-Max 量化器 (最优非均匀量化器, 基于分布 PDF)")
    print(f"  4. 混合方案: center 用 K-means 质心, tail 用 NF 网格")

    # Bonus: verify NF8 actually keeps endpoints
    print(f"\n  ── 验证 ──")
    print(f"  NF8 eff-LUT 端点: [{nf8_lut[0]:.6f}, {nf8_lut[-1]:.6f}]  ← 精确 ±1")
    print(f"  NF8 实际重构范围: y_q=[-127,127] → x'=[{-(127/127)**(1/POW):.6f}, {(127/127)**(1/POW):.6f}]")
    print(f"  std-KM 端点: [{km_lut[0]:.4f}, {km_lut[-1]:.4f}]")
    print(f"  wKM-gauss 端点: [{wkm_gauss_lut[0]:.4f}, {wkm_gauss_lut[-1]:.4f}]")
    if abs(km_lut[0]) < 0.95 or abs(km_lut[-1]) < 0.95:
        print(f"  ⚠ K-means 端点偏离 ±1 → 尾部 5% 的值无法精确表示")
    if abs(wkm_gauss_lut[0]) < 0.95 or abs(wkm_gauss_lut[-1]) < 0.95:
        print(f"  ⚠ 加权 K-means 端点更偏离 ±1 → 尾部误差更大")

    print("\nDone.")


if __name__ == "__main__":
    main()
