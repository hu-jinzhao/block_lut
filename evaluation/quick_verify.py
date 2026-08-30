"""快速验证 5 个方向的可行性，仅采样分析"""
import os, math, numpy as np, torch
from safetensors import safe_open
from collections import defaultdict

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
sf_files = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])

# 加载所有 expert key
all_experts = []
for sf in sf_files:
    with safe_open(sf, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" in k and "shared_expert" not in k and ".weight" in k:
                parts = k.split(".")
                lid, eid = None, None
                for i, p in enumerate(parts):
                    if p == "layers" and i+1 < len(parts): lid = int(parts[i+1])
                    if p == "experts" and i+1 < len(parts): eid = int(parts[i+1])
                if lid is not None and eid is not None:
                    ttype = "gate" if "gate_proj" in k else ("up" if "up_proj" in k else ("down" if "down_proj" in k else None))
                    if ttype:
                        all_experts.append((lid, eid, ttype, k, sf))

print(f"Total expert tensors: {len(all_experts)}")
unique_layers = sorted(set(e[0] for e in all_experts))
unique_experts_per_layer = {l: sorted(set(e[1] for e in all_experts if e[0]==l)) for l in unique_layers}
print(f"Layers: {unique_layers}, experts per layer: {len(unique_experts_per_layer[unique_layers[0]])}")

# 采样: 取浅层(2)、中层(11)、深层(22)，每层 10 个专家
sampled_layers = [unique_layers[0], unique_layers[len(unique_layers)//2], unique_layers[-1]]
print(f"Sampled layers: {sampled_layers}")

sampled = [(l, e, t, k, sf) for (l, e, t, k, sf) in all_experts
           if l in sampled_layers and e < 10]
print(f"Sampled tensors: {len(sampled)}")

# 加载采样数据
matrices = {}
for (lid, eid, ttype, key, sf) in sampled:
    with safe_open(sf, framework="pt", device="cpu") as f:
        m = f.get_tensor(key).to(torch.float32).numpy()
    matrices[(lid, eid, ttype)] = m
print(f"Loaded {len(matrices)} matrices")

# =====================
# 1. SVD 低秩性
# =====================
print("\n=== 1. 低秩分解 (per layer) ===")
for lid in sampled_layers:
    for ttype in ["gate", "up", "down"]:
        ratios = []
        for (l, e, t), m in matrices.items():
            if l == lid and t == ttype:
                S = np.linalg.svd(m, full_matrices=False)[1]
                cumsum = np.cumsum(S) / np.sum(S)
                r95 = np.searchsorted(cumsum, 0.95) + 1
                ratios.append(r95 / min(m.shape))
        if ratios:
            print(f"  L{lid} {ttype}: r95/min_dim = {np.mean(ratios):.3f} (±{np.std(ratios):.3f})")
            # 估算压缩比: (M*N) / (M*r + r*N + r*r)
            M, N = 1408, 2048
            r = int(np.mean(ratios) * min(M, N))
            orig = M * N
            compressed = M * r + r * N + r * r
            print(f"         Tucker 压缩比: {orig/compressed:.2f}x (r={r})")

# =====================
# 2. 共享子空间
# =====================
print("\n=== 2. 共享子空间 (层内专家间相关性) ===")
for lid in sampled_layers:
    gate_vecs = []
    for eid in range(10):
        key = (lid, eid, "gate")
        if key in matrices:
            gate_vecs.append(matrices[key].ravel())
    if len(gate_vecs) > 1:
        # 专家间相关系数
        corrs = []
        for i in range(len(gate_vecs)):
            for j in range(i+1, len(gate_vecs)):
                corrs.append(np.corrcoef(gate_vecs[i], gate_vecs[j])[0, 1])
        print(f"  L{lid} gate_proj {len(gate_vecs)}专家: 平均互相关={np.mean(corrs):.4f}, "
              f"max={np.max(corrs):.4f}, |corr|>0.5比例={np.mean(np.abs(corrs)>0.5)*100:.1f}%")

        # 拼接矩阵的共享秩
        stacked = np.vstack([matrices[(lid, e, "gate")] for e in range(10) if (lid, e, "gate") in matrices])
        S = np.linalg.svd(stacked, full_matrices=False)[1]
        cumsum = np.cumsum(S) / np.sum(S)
        r50 = np.searchsorted(cumsum, 0.50) + 1
        r90 = np.searchsorted(cumsum, 0.90) + 1
        r95 = np.searchsorted(cumsum, 0.95) + 1
        print(f"         stacking (10 experts): 50%={r50}, 90%={r90}, 95%={r95},"
              f" 共享基压缩={10*min(stacked.shape)/r90:.1f}x vs 单专家")

# =====================
# 3. 稀疏性
# =====================
print("\n=== 3. 权重稀疏性 ===")
for ttype in ["gate", "up", "down"]:
    sp_vals = []
    for (l, e, t), m in matrices.items():
        if t == ttype:
            abs_m = np.abs(m)
            mx = np.max(abs_m)
            for thr_name, thr in [("1e-2", 1e-2), ("1e-3", 1e-3)]:
                sp_vals.append((thr_name, np.mean(abs_m < thr * mx)))
    for thr_name in ["1e-2", "1e-3"]:
        vals = [v for (tn, v) in sp_vals if tn == thr_name]
        print(f"  {ttype} thr={thr_name}: 稀疏度={np.mean(vals)*100:.2f}%")

# =====================
# 4. 查表量化
# =====================
print("\n=== 4. 查表量化 ===")
for (l, e, t), m in list(matrices.items())[:6]:
    flat = m.ravel()
    for n_bins in [64, 256, 1024]:
        q = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(flat, q)
        digitized = np.digitize(flat, edges[1:-1])
        recon = np.zeros_like(flat)
        for b in range(n_bins):
            mask = digitized == b
            if mask.any():
                recon[mask] = np.mean(flat[mask])
        mse = np.mean((flat - recon)**2)
        psnr = 10 * np.log10(np.max(np.abs(flat))**2 / mse) if mse > 0 else float("inf")
        bits = math.log2(n_bins)
        print(f"  L{l}E{e} {t}: {n_bins}bins → {bits:.0f}bit PSNR={psnr:.1f}dB "
              f"({16/bits:.1f}x vs bf16)")

# =====================
# 5. 块自相似性
# =====================
print("\n=== 5. 块自相似性 ===")
for (l, e, t), m in list(matrices.items())[:5]:
    h, w = m.shape
    h2, w2 = h//2, w//2
    ul = m[:h2, :w2].ravel()
    ur = m[:h2, w2:].ravel()
    ll = m[h2:, :w2].ravel()
    lr = m[h2:, w2:].ravel()
    print(f"  L{l}E{e} {t}: UL-UR corr={np.corrcoef(ul, ur)[0,1]:.4f}, "
          f"UL-LL corr={np.corrcoef(ul, ll)[0,1]:.4f}, "
          f"diag corr={np.corrcoef(ul, lr)[0,1]:.4f}")

print("\nDone.")
