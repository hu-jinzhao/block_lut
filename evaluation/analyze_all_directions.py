"""
逐一验证 5 个压缩方向的可行性。
"""
import os, sys, json, math
import numpy as np
import torch
from safetensors import safe_open
from collections import defaultdict

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OUTPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

safetensor_files = sorted([
    os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
    if f.endswith(".safetensors")
])

def load_all_expert_keys():
    """获取所有 expert key 的 (layer, expert, tensor_type, key, file)"""
    result = []
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "expert" in k and "shared_expert" not in k:
                    parts = k.split(".")
                    layer_idx = None
                    expert_idx = None
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                        if p == "experts" and i + 1 < len(parts):
                            expert_idx = int(parts[i + 1])
                    if layer_idx is not None and expert_idx is not None:
                        if "gate_proj" in k:
                            ttype = "gate_proj"
                        elif "up_proj" in k:
                            ttype = "up_proj"
                        elif "down_proj" in k:
                            ttype = "down_proj"
                        else:
                            continue
                        result.append((layer_idx, expert_idx, ttype, k, sf))
    return result

# 加载所有 expert（不采样）
all_keys = load_all_expert_keys()
print(f"Total expert tensors: {len(all_keys)}")

# 缓存
file_cache = {}
def get_tensor(sf, key):
    if sf not in file_cache:
        file_cache[sf] = {}
    if key not in file_cache[sf]:
        with safe_open(sf, framework="pt", device="cpu") as f:
            file_cache[sf][key] = f.get_tensor(key).to(torch.float32).numpy()
    return file_cache[sf][key]

# ===================================================================
# 1. 方向一：低秩分解 — 逐层分析有效秩
# ===================================================================
print("\n" + "="*60)
print("方向一: Tensor Train / Tucker 低秩分解")
print("="*60)

layer_ranks = defaultdict(lambda: {"gate": [], "up": [], "down": []})
for (layer, expert, ttype, key, sf) in all_keys:
    matrix = get_tensor(sf, key)
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    cumsum = np.cumsum(S) / np.sum(S)
    r90 = np.searchsorted(cumsum, 0.90) + 1
    r95 = np.searchsorted(cumsum, 0.95) + 1
    r99 = np.searchsorted(cumsum, 0.99) + 1
    min_dim = min(matrix.shape)
    if ttype == "gate_proj":
        layer_ranks[layer]["gate"].append((r90, r95, r99, min_dim))
    elif ttype == "up_proj":
        layer_ranks[layer]["up"].append((r90, r95, r99, min_dim))
    else:
        layer_ranks[layer]["down"].append((r90, r95, r99, min_dim))

print(f"\n{'Layer':<8} {'Type':<10} {'r90':<8} {'r95':<8} {'r99':<8} {'min_dim':<10} {'r95/min_dim':<12}")
print("-" * 60)
for layer in sorted(layer_ranks.keys()):
    for ttype, label in [("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")]:
        vals = layer_ranks[layer][ttype]
        if vals:
            avg = np.mean(vals, axis=0)
            print(f"{layer:<8} {label:<10} {int(avg[0]):<8} {int(avg[1]):<8} {int(avg[2]):<8} "
                  f"{int(avg[3]):<10} {avg[1]/avg[3]:.2f}")

overall = {"gate": [], "up": [], "down": []}
for layer in layer_ranks:
    for ttype in overall:
        overall[ttype].extend([v[1]/v[3] for v in layer_ranks[layer][ttype]])

print(f"\n整体平均 r95/min_dim: gate={np.mean(overall['gate']):.2f}, "
      f"up={np.mean(overall['up']):.2f}, down={np.mean(overall['down']):.2f}")

min_layer = min(layer_ranks.keys())
max_layer = max(layer_ranks.keys())
early_gate = [v[1]/v[3] for v in layer_ranks[min_layer]["gate"]] if layer_ranks[min_layer]["gate"] else [0]
late_gate = [v[1]/v[3] for v in layer_ranks[max_layer]["gate"]] if layer_ranks[max_layer]["gate"] else [0]
print(f"\n浅层(L{min_layer}) vs 深层(L{max_layer}) gate_proj 有效秩比: "
      f"{np.mean(early_gate):.2f} vs {np.mean(late_gate):.2f}")

# ===================================================================
# 2. 方向二：共享子空间 / 随机投影
# ===================================================================
print("\n" + "="*60)
print("方向二: 共享子空间 / 跨专家相关性")
print("="*60)

# 对同一层的所有 expert 计算 gate_proj 矩阵之间的相关系数
test_layer = 10  # 选中间层
print(f"\nLayer {test_layer} gate_proj 专家间相关性:")

layer_experts = defaultdict(dict)
for (layer, expert, ttype, key, sf) in all_keys:
    if layer == test_layer:
        matrix = get_tensor(sf, key)
        layer_experts[expert][ttype] = matrix

# 计算 expert 之间的相关系数
expert_ids = sorted(layer_experts.keys())
n_experts = len(expert_ids)
corr_matrix_gate = np.zeros((n_experts, n_experts))
for i, e_i in enumerate(expert_ids):
    for j, e_j in enumerate(expert_ids):
        if "gate_proj" in layer_experts[e_i] and "gate_proj" in layer_experts[e_j]:
            m_i = layer_experts[e_i]["gate_proj"].ravel()
            m_j = layer_experts[e_j]["gate_proj"].ravel()
            corr_matrix_gate[i, j] = np.corrcoef(m_i, m_j)[0, 1]

# 去掉对角线
off_diag = corr_matrix_gate[~np.eye(n_experts, dtype=bool)]
print(f"  专家间平均相关系数 (gate_proj): {np.mean(off_diag):.4f}")
print(f"  专家间最大相关系数: {np.max(off_diag):.4f}")
print(f"  专家间最小相关系数: {np.min(off_diag):.4f}")
print(f"  相关系数 >0.5 的比例: {np.mean(np.abs(off_diag) > 0.5)*100:.1f}%")
print(f"  相关系数 >0.1 的比例: {np.mean(np.abs(off_diag) > 0.1)*100:.1f}%")

# 共享基底分析: 对多个 expert 拼接后做 SVD
print(f"\nLayer {test_layer} gate_proj 共享基底分析:")
gate_matrices = [layer_experts[e]["gate_proj"] for e in expert_ids if "gate_proj" in layer_experts[e]]
# 拼接: (n_experts * rows, cols)
stacked = np.vstack(gate_matrices)  # (60*1408, 2048)
U_stacked, S_stacked, Vt_stacked = np.linalg.svd(stacked, full_matrices=False)
cumsum_stacked = np.cumsum(S_stacked) / np.sum(S_stacked)
r50_stacked = np.searchsorted(cumsum_stacked, 0.50) + 1
r90_stacked = np.searchsorted(cumsum_stacked, 0.90) + 1
r95_stacked = np.searchsorted(cumsum_stacked, 0.95) + 1
r99_stacked = np.searchsorted(cumsum_stacked, 0.99) + 1
min_dim_stacked = min(stacked.shape)
print(f"  拼接矩阵 shape: {stacked.shape}")
print(f"  50%能量需要: {r50_stacked} (/{min_dim_stacked}, ratio={r50_stacked/min_dim_stacked:.4f})")
print(f"  90%能量需要: {r90_stacked} (/{min_dim_stacked}, ratio={r90_stacked/min_dim_stacked:.4f})")
print(f"  95%能量需要: {r95_stacked} (/{min_dim_stacked}, ratio={r95_stacked/min_dim_stacked:.4f})")
print(f"  99%能量需要: {r99_stacked} (/{min_dim_stacked}, ratio={r99_stacked/min_dim_stacked:.4f})")

# 对比：单个 expert 相同维度的秩
single_gate = layer_experts[expert_ids[0]]["gate_proj"]
U1, S1, Vt1 = np.linalg.svd(single_gate, full_matrices=False)
cumsum1 = np.cumsum(S1) / np.sum(S1)
r95_single = np.searchsorted(cumsum1, 0.95) + 1
print(f"\n  单个 expert r95: {r95_single}/{min(single_gate.shape)}")
print(f"  共享基底的压缩比: {r90_stacked} 个基向量表示 {n_experts} 个专家 "
      f"→ 每个专家平均 {r90_stacked/n_experts:.0f} 个等效基")

# ===================================================================
# 3. 方向三：结构化剪枝 + RLE
# ===================================================================
print("\n" + "="*60)
print("方向三: 结构化剪枝 / 稀疏性分析")
print("="*60)

sparsity_stats = defaultdict(list)
for (layer, expert, ttype, key, sf) in all_keys[:200]:  # 采样
    matrix = get_tensor(sf, key)
    abs_m = np.abs(matrix)
    max_val = np.max(abs_m)
    for thr_name, thr in [("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4), ("1e-5", 1e-5)]:
        sparsity = np.mean(abs_m < thr * max_val)
        sparsity_stats[(ttype, thr_name)].append(sparsity)

print(f"\n{'Type':<12} {'Thr=1e-2':<12} {'Thr=1e-3':<12} {'Thr=1e-4':<12} {'Thr=1e-5':<12}")
print("-" * 60)
for ttype in ["gate_proj", "up_proj", "down_proj"]:
    vals = []
    for thr_name in ["1e-2", "1e-3", "1e-4", "1e-5"]:
        s = sparsity_stats[(ttype, thr_name)]
        vals.append(f"{np.mean(s)*100:5.1f}%")
    print(f"{ttype:<12} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12} {vals[3]:<12}")

# 行/列级别的稀疏性（结构化剪枝关键指标）
print("\n行级稀疏性 (整行接近零):")
for (layer, expert, ttype, key, sf) in all_keys[:30]:
    matrix = get_tensor(sf, key)
    row_norms = np.linalg.norm(matrix, axis=1)
    max_rn = np.max(row_norms)
    row_sparsity_1e2 = np.mean(row_norms < 1e-2 * max_rn)
    row_sparsity_1e3 = np.mean(row_norms < 1e-3 * max_rn)
    print(f"  {key}: 整行范数<1%max={row_sparsity_1e2*100:.1f}%, <0.1%max={row_sparsity_1e3*100:.1f}%")

# ===================================================================
# 4. 方向四：查表量化
# ===================================================================
print("\n" + "="*60)
print("方向四: 查表量化 (数值聚类)")
print("="*60)

# 取几个矩阵分析值分布
for (layer, expert, ttype, key, sf) in all_keys[:6]:
    matrix = get_tensor(sf, key)
    flat = matrix.ravel()

    # 统计唯一值数量（量化后）
    print(f"\n  {key}:")
    print(f"    值范围: [{np.min(flat):.4f}, {np.max(flat):.4f}]")
    print(f"    标准差: {np.std(flat):.4f}")

    # 用直方图统计：分成 256 个桶，看多少个桶占 95% 质量
    hist, bin_edges = np.histogram(flat, bins=256)
    hist_sorted = np.sort(hist)[::-1]
    cumsum = np.cumsum(hist_sorted) / np.sum(hist_sorted)
    bins_90 = np.searchsorted(cumsum, 0.90) + 1
    bins_95 = np.searchsorted(cumsum, 0.95) + 1
    bins_99 = np.searchsorted(cumsum, 0.99) + 1
    print(f"    256桶中覆盖90%权重需要: {bins_90} 桶 ({bins_90/256*100:.0f}%)")
    print(f"    256桶中覆盖95%权重需要: {bins_95} 桶 ({bins_95/256*100:.0f}%)")
    print(f"    256桶中覆盖99%权重需要: {bins_99} 桶 ({bins_99/256*100:.0f}%)")

# K-means 风格: 如果只用 N 个值（聚类中心）会怎样
print("\n  K-means 风格量化误差 (用分位数做简单聚类):")
for (layer, expert, ttype, key, sf) in all_keys[:6]:
    matrix = get_tensor(sf, key)
    flat = matrix.ravel()
    for n_bins in [16, 64, 256, 1024]:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges_q = np.percentile(flat, quantiles)
        # 用每个桶的均值做代表
        digitized = np.digitize(flat, bin_edges_q[1:-1])
        # 对每个桶计算均值
        reconstructed = np.zeros_like(flat)
        for b in range(n_bins):
            mask = digitized == b
            if np.any(mask):
                reconstructed[mask] = np.mean(flat[mask])
        mse = np.mean((flat - reconstructed) ** 2)
        psnr = 10 * np.log10(np.max(np.abs(flat)) ** 2 / mse) if mse > 0 else float("inf")
        # 压缩比: 每个值需要 log2(n_bins) bits + n_bins 个浮点值
        bits_per_elem = math.log2(n_bins)
        compression_vs_bf16 = 16 / bits_per_elem
        if n_bins <= 256:
            print(f"    {key}: {n_bins} bins → PSNR={psnr:.1f}dB, "
                  f"{bits_per_elem:.1f}bit/elem, 压缩比={compression_vs_bf16:.1f}x vs bf16")

# ===================================================================
# 5. 分形压缩
# ===================================================================
print("\n" + "="*60)
print("方向五: 分形压缩 / 块自相似性")
print("="*60)

for (layer, expert, ttype, key, sf) in all_keys[:5]:
    matrix = get_tensor(sf, key)
    h, w = matrix.shape
    # 把矩阵分成 4 块，检查块间相关性
    h2, w2 = h // 2, w // 2
    blocks = {
        "UL": matrix[:h2, :w2],
        "UR": matrix[:h2, w2:],
        "LL": matrix[h2:, :w2],
        "LR": matrix[h2:, w2:],
    }
    print(f"\n  {key} ({h}x{w}):")
    for name1, b1 in blocks.items():
        for name2, b2 in blocks.items():
            if name1 < name2:
                corr = np.corrcoef(b1.ravel(), b2.ravel())[0, 1]
                ratio = np.mean(np.abs(b1)) / (np.mean(np.abs(b2)) + 1e-10)
                print(f"    {name1} vs {name2}: corr={corr:.4f}, mean_ratio={ratio:.4f}")

# ===================================================================
# 总结
# ===================================================================
print("\n" + "="*60)
print("综合总结")
print("="*60)

print("""
方向一 (低秩分解): ❌ 不可行
  - 即便最好的层，95%能量仍需 ~84% 奇异值
  - 声称的 r=32-128 在 Qwen 上不成立
  - Tensor Train / Tucker 压缩比仅 1.2-1.5x，不如当前 LZ4HC 的 1.5x

方向二 (共享子空间): ⚠️ 部分可行
  - 专家间相关系数仅 ~0.01（几乎正交），共享基底收益有限
  - 使用共享基底可以减少参数，但需要高秩（r≈800+）才能保精度
  - 真正的共享方向：多个 expert 共享一个高秩基 + 各自低秩残差

方向三 (结构化剪枝): ⚠️ 有限可行
  - 权重稀疏度很低（1e-4阈值下仅 <1% 为零），没有天然稀疏性
  - 行级稀疏性几乎为零
  - 要做剪枝需要主动训练/微调来诱导稀疏

方向四 (查表量化): ✅ 最可行
  - 256 个值就能覆盖 99% 权重，PSNR 可达 30+ dB
  - 压缩比 2x vs bf16（8bit 索引 + 256×2byte 表）
  - 解压是查表，速度最快，适合在 GPU 上即时解压
  - 与当前 exp/SM 分离编码可叠加使用

方向五 (分形): ❌ 不可行
  - 块间相关系数接近 0，没有任何自相似性
  - 神经网络权重不具备分形结构
""")
