"""
分析 Qwen 专家矩阵的低秩/平滑特性，验证 DCT 压缩可行性。
检查维度：
  1. SVD 奇异值衰减 — 如果快速衰减则矩阵低秩
  2. 2D-DCT 能量集中度 — 如果大部分能量集中在低频，则 DCT 有效
  3. DCT 系数截断后的重建误差 — 模拟不同压缩率下的精度损失
"""
import os, sys, json, math, time
import numpy as np
import torch
from safetensors import safe_open
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = "/home/hh/zip_Moe/LUT_MoE/models/qwen"
OUTPUT_DIR = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- helpers ---

def load_expert_tensors(safetensor_files, num_layers=24, num_experts=60, max_samples=30):
    """从 safetensors 中采样专家矩阵。返回 [(layer, expert, tensor_type, matrix_np)]"""
    all_keys = []
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cpu") as f:
            all_keys.extend(f.keys())

    expert_keys = defaultdict(list)
    for k in all_keys:
        if "expert" in k and "shared_expert" not in k:
            # 提取 layer 和 expert 编号
            parts = k.split(".")
            layer_idx = None
            expert_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                if p == "experts" and i + 1 < len(parts):
                    expert_idx = int(parts[i + 1])
            if layer_idx is not None and expert_idx is not None:
                expert_keys[(layer_idx, expert_idx)].append(k)

    # 均匀采样
    all_experts = sorted(expert_keys.keys())
    rng = np.random.RandomState(42)
    if len(all_experts) > max_samples:
        sampled = [all_experts[i] for i in rng.choice(len(all_experts), max_samples, replace=False)]
    else:
        sampled = all_experts

    matrices = []
    # 缓存已加载的文件
    loaded = {}
    for sf in safetensor_files:
        loaded[sf] = {}

    for (layer, expert) in sampled:
        for k in expert_keys[(layer, expert)]:
            # 确定 tensor 类型
            if "gate_proj" in k:
                ttype = "gate_proj"
            elif "up_proj" in k:
                ttype = "up_proj"
            elif "down_proj" in k:
                ttype = "down_proj"
            else:
                continue

            # 找到对应的 safetensor 文件
            for sf in safetensor_files:
                if k not in loaded[sf]:
                    with safe_open(sf, framework="pt", device="cpu") as f:
                        if k in f.keys():
                            loaded[sf][k] = f.get_tensor(k)
                if k in loaded[sf]:
                    tensor = loaded[sf][k]
                    break
            else:
                continue

            matrix = tensor.to(torch.float32).numpy()
            matrices.append((layer, expert, ttype, k, matrix))

    print(f"Loaded {len(matrices)} expert matrices from {len(sampled)} experts")
    return matrices


def analyze_svd(matrix, name=""):
    """奇异值分析：衰减曲线、有效秩"""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    total = np.sum(S)
    cumsum = np.cumsum(S) / total

    # 有效秩：累积能量达 90%, 95%, 99% 需要的奇异值数量
    r90 = np.searchsorted(cumsum, 0.90) + 1
    r95 = np.searchsorted(cumsum, 0.95) + 1
    r99 = np.searchsorted(cumsum, 0.99) + 1

    return {
        "S": S,
        "cumsum": cumsum,
        "r90": r90, "r95": r95, "r99": r99,
        "total_singular_values": len(S),
        "ratio_90": r90 / len(S),
        "ratio_95": r95 / len(S),
        "ratio_99": r99 / len(S),
        "name": name,
    }


def analyze_dct2d(matrix, name=""):
    """2D DCT 能量集中度分析"""
    from scipy.fft import dctn
    coeffs = dctn(matrix, norm="ortho")
    # 按能量（幅度）排序
    flat = np.abs(coeffs).ravel()
    flat_sorted = np.sort(flat)[::-1]
    total_energy = np.sum(flat_sorted)
    cumsum = np.cumsum(flat_sorted) / total_energy

    total_coeffs = len(flat_sorted)
    # 保留 5%, 10%, 20%, 30% 系数时的能量百分比
    ratios = {}
    for pct in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(total_coeffs * pct))
        ratios[f"keep_{int(pct*100)}pct"] = float(cumsum[k - 1])

    return {
        "coeffs_sorted": flat_sorted,
        "cumsum": cumsum,
        "ratios": ratios,
        "total_coeffs": total_coeffs,
        "name": name,
    }


def dct_reconstruction_error(matrix, keep_fractions):
    """模拟 DCT 压缩重建误差"""
    from scipy.fft import dctn, idctn
    coeffs = dctn(matrix, norm="ortho")
    flat = coeffs.ravel()
    total = flat.size

    errors = {}
    for frac in keep_fractions:
        k = max(1, int(total * frac))
        # 阈值：保留 top-k 系数
        threshold = np.sort(np.abs(flat))[::-1][k - 1]
        mask = np.abs(coeffs) >= threshold
        truncated = coeffs * mask
        recon = idctn(truncated, norm="ortho")
        mse = np.mean((matrix - recon) ** 2)
        psnr = 10 * np.log10(np.max(np.abs(matrix)) ** 2 / mse) if mse > 0 else float("inf")
        errors[frac] = {"mse": float(mse), "psnr": float(psnr)}

    return errors


# --- main ---

def main():
    safetensor_files = sorted([
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".safetensors")
    ])
    print(f"Found {len(safetensor_files)} safetensor files")

    matrices = load_expert_tensors(safetensor_files, max_samples=30)

    # 按 tensor 类型分组分析
    svd_results = {"gate_proj": [], "up_proj": [], "down_proj": []}
    dct_results = {"gate_proj": [], "up_proj": [], "down_proj": []}

    print("\n=== SVD Analysis ===")
    for layer, expert, ttype, name, matrix in matrices:
        r = analyze_svd(matrix, f"L{layer}_E{expert}_{ttype}")
        svd_results[ttype].append(r)

    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        if not svd_results[ttype]:
            continue
        avg_r90 = np.mean([r["ratio_90"] for r in svd_results[ttype]])
        avg_r95 = np.mean([r["ratio_95"] for r in svd_results[ttype]])
        avg_r99 = np.mean([r["ratio_99"] for r in svd_results[ttype]])
        print(f"  {ttype}: 能量到 90% 需 {avg_r90*100:.1f}% 奇异值, "
              f"95% 需 {avg_r95*100:.1f}%, 99% 需 {avg_r99*100:.1f}%")

    print("\n=== 2D-DCT Energy Concentration ===")
    for layer, expert, ttype, name, matrix in matrices:
        r = analyze_dct2d(matrix, f"L{layer}_E{expert}_{ttype}")
        dct_results[ttype].append(r)

    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        if not dct_results[ttype]:
            continue
        for pct_label in ["keep_1pct", "keep_5pct", "keep_10pct", "keep_20pct", "keep_30pct", "keep_50pct"]:
            vals = [r["ratios"][pct_label] for r in dct_results[ttype]]
            avg = np.mean(vals)
            print(f"  {ttype} {pct_label}: 平均保留 {avg*100:.1f}% 能量")

    print("\n=== DCT Reconstruction Error ===")
    keep_fractions = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    reco_errors = {"gate_proj": defaultdict(list), "up_proj": defaultdict(list), "down_proj": defaultdict(list)}

    sample_count = 0
    for layer, expert, ttype, name, matrix in matrices:
        if sample_count >= 10:  # 只对前 10 个做重建（慢）
            break
        sample_count += 1
        errors = dct_reconstruction_error(matrix, keep_fractions)
        for frac, err in errors.items():
            reco_errors[ttype][frac].append(err)
        print(f"  [{sample_count}/10] {name}: keep=10% PSNR={errors[0.10]['psnr']:.1f}dB")

    print("\n=== Summary: DCT Reconstruction PSNR ===")
    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        print(f"  {ttype}:")
        for frac in keep_fractions:
            psnrs = [e["psnr"] for e in reco_errors[ttype][frac]]
            if psnrs:
                print(f"    keep={int(frac*100)}%: PSNR avg={np.mean(psnrs):.1f} dB, "
                      f"min={np.min(psnrs):.1f}, max={np.max(psnrs):.1f}")

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {"gate_proj": "#2196F3", "up_proj": "#4CAF50", "down_proj": "#FF9800"}

    # 图1: SVD 奇异值衰减（选 3 个典型矩阵）
    ax = axes[0, 0]
    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        for i, r in enumerate(svd_results[ttype][:3]):
            ax.semilogy(r["S"][:500], alpha=0.6, color=colors[ttype],
                        label=f"{ttype}" if i == 0 else "")
    ax.set_title("SVD: Singular Value Decay (first 500)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value (log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图2: SVD 累积能量
    ax = axes[0, 1]
    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        all_cumsum = np.array([r["cumsum"] for r in svd_results[ttype]])
        mean_cs = np.mean(all_cumsum, axis=0)
        std_cs = np.std(all_cumsum, axis=0)
        x = np.arange(len(mean_cs))
        ax.plot(x, mean_cs, color=colors[ttype], label=ttype)
        ax.fill_between(x, mean_cs - std_cs, mean_cs + std_cs, alpha=0.15, color=colors[ttype])
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax.set_title("SVD: Cumulative Energy (mean ± std)")
    ax.set_xlabel("Number of singular values")
    ax.set_ylabel("Cumulative energy fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图3: 有效秩分布
    ax = axes[0, 2]
    positions = []
    labels = []
    for i, ttype in enumerate(["gate_proj", "up_proj", "down_proj"]):
        for j, (label, key) in enumerate([("90%", "r90"), ("95%", "r95"), ("99%", "r99")]):
            vals = [r[key] / r["total_singular_values"] * 100 for r in svd_results[ttype]]
            pos = i * 3 + j
            positions.append(pos)
            bp = ax.boxplot(vals, positions=[pos], widths=0.6,
                            patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[ttype])
                patch.set_alpha(0.3 + j * 0.2)
            if i == 0:
                labels.append(f"{label}")
            else:
                labels.append("")
    ax.set_xticks(positions)
    ax.set_xticklabels(["90%\ngate", "95%\ngate", "99%\ngate",
                        "90%\nup", "95%\nup", "99%\nup",
                        "90%\ndown", "95%\ndown", "99%\ndown"], fontsize=7)
    ax.set_title("SVD: Effective Rank (% of total)")
    ax.set_ylabel("% of singular values needed")
    ax.grid(True, alpha=0.3)

    # 图4: DCT 能量集中
    ax = axes[1, 0]
    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        all_cumsum = np.array([r["cumsum"][:5000] for r in dct_results[ttype]])
        mean_cs = np.mean(all_cumsum, axis=0)
        x = np.arange(len(mean_cs))
        ax.plot(x, mean_cs, color=colors[ttype], label=ttype)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax.set_title("2D-DCT: Energy Concentration (first 5000 coeffs)")
    ax.set_xlabel("Number of DCT coefficients (sorted by magnitude)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图5: DCT 系数保留 vs 能量
    ax = axes[1, 1]
    x_positions = np.arange(len(keep_fractions))
    for i, ttype in enumerate(["gate_proj", "up_proj", "down_proj"]):
        means = []
        stds = []
        for frac in keep_fractions:
            label = f"keep_{int(frac*100)}pct"
            vals = [r["ratios"][label] for r in dct_results[ttype]]
            means.append(np.mean(vals) * 100)
            stds.append(np.std(vals) * 100)
        offset = (i - 1) * 0.1
        ax.bar(x_positions + offset, means, 0.08, yerr=stds, color=colors[ttype],
               label=ttype, capsize=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(f*100)}%" for f in keep_fractions])
    ax.set_title("2D-DCT: Energy Retained vs Coefficients Kept")
    ax.set_xlabel("Fraction of DCT coefficients kept")
    ax.set_ylabel("Energy retained (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 图6: DCT 重建 PSNR
    ax = axes[1, 2]
    for ttype in ["gate_proj", "up_proj", "down_proj"]:
        mean_psnrs = []
        for frac in keep_fractions:
            psnrs = [e["psnr"] for e in reco_errors[ttype][frac]]
            mean_psnrs.append(np.mean(psnrs) if psnrs else 0)
        ax.plot([f * 100 for f in keep_fractions], mean_psnrs, "o-",
                color=colors[ttype], label=ttype, markersize=6)
    ax.axhline(y=40, color="gray", linestyle="--", alpha=0.5, label="40 dB (high quality)")
    ax.axhline(y=30, color="gray", linestyle=":", alpha=0.5, label="30 dB (acceptable)")
    ax.set_title("DCT: Reconstruction PSNR vs Coefficients Retained")
    ax.set_xlabel("Coefficients retained (%)")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("LUT_MoE Qwen1.5-MoE-A2.7B — Expert Weight Matrix Analysis\n"
                 "SVD + 2D-DCT Energy Concentration + Reconstruction Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "expert_dct_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {output_path}")

    # 额外：检查矩阵的平滑度（相邻元素相关性）
    print("\n=== Matrix Smoothness (相邻元素相关性) ===")
    for layer, expert, ttype, name, matrix in matrices[:5]:
        # 水平相邻差
        h_diff = np.mean(np.abs(matrix[:, 1:] - matrix[:, :-1]))
        # 垂直相邻差
        v_diff = np.mean(np.abs(matrix[1:, :] - matrix[:-1, :]))
        # 归一化
        abs_mean = np.mean(np.abs(matrix))
        print(f"  {name}: h_grad={h_diff/abs_mean:.3f}, v_grad={v_diff/abs_mean:.3f} "
              f"(rel to mean abs)")

    # 保存数值结果
    summary = {
        "svd": {ttype: {
            "avg_ratio_90": float(np.mean([r["ratio_90"] for r in svd_results[ttype]])),
            "avg_ratio_95": float(np.mean([r["ratio_95"] for r in svd_results[ttype]])),
            "avg_ratio_99": float(np.mean([r["ratio_99"] for r in svd_results[ttype]])),
        } for ttype in svd_results},
        "dct_energy": {ttype: {
            label: float(np.mean([r["ratios"][label] for r in dct_results[ttype]]))
            for label in ["keep_1pct", "keep_5pct", "keep_10pct", "keep_20pct", "keep_30pct"]
        } for ttype in dct_results},
        "dct_psnr": {ttype: {
            str(int(frac*100)): float(np.mean([e["psnr"] for e in reco_errors[ttype][frac]]))
            for frac in keep_fractions
        } for ttype in reco_errors},
    }
    with open(os.path.join(OUTPUT_DIR, "dct_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 最终结论 ===")
    # 用 gate_proj 的代表性数据判断
    if svd_results["gate_proj"]:
        avg_r95 = np.mean([r["ratio_95"] for r in svd_results["gate_proj"]])
        if avg_r95 < 0.3:
            print(f"✅ SVD: 矩阵具有较强低秩性 (95%能量仅需{avg_r95*100:.0f}%奇异值)")
        elif avg_r95 < 0.5:
            print(f"⚠️  SVD: 矩阵具有中等低秩性 (95%能量需{avg_r95*100:.0f}%奇异值)")
        else:
            print(f"❌ SVD: 矩阵低秩性较弱 (95%能量需{avg_r95*100:.0f}%奇异值)")

    if dct_results["gate_proj"]:
        keep10_energy = np.mean([r["ratios"]["keep_10pct"] for r in dct_results["gate_proj"]])
        if keep10_energy > 0.95:
            print(f"✅ DCT: 能量高度集中 (10%系数保留{keep10_energy*100:.0f}%能量) -> DCT压缩有效")
        elif keep10_energy > 0.85:
            print(f"⚠️  DCT: 能量中等集中 (10%系数保留{keep10_energy*100:.0f}%能量) -> DCT有一定效果")
        else:
            print(f"❌ DCT: 能量分散 (10%系数仅保留{keep10_energy*100:.0f}%能量) -> DCT不适合")

    avg_psnr_10 = np.mean([e["psnr"] for e in reco_errors["gate_proj"][0.10]])
    print(f"   DCT保留10%系数时重建PSNR: {avg_psnr_10:.1f} dB "
          f"({'可用' if avg_psnr_10 > 30 else '不可用'})")

if __name__ == "__main__":
    main()
