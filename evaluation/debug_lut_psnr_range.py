"""检查不同 layer/expert 的 LUT 量化 PSNR 范围"""
import os, sys, struct
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch

LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
ORIG_CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"

# Load LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)

def quantize_py(tensor):
    flat_u16 = tensor.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
    flat_mono = np.where(flat_u16 & 0x8000, ~flat_u16, flat_u16 ^ np.uint16(0x8000)).astype(np.uint16)
    return np.searchsorted(thresholds, flat_mono).astype(np.uint8)

def recover_py(indices):
    reco_u16 = lut_u16[indices]
    return torch.from_numpy(reco_u16.astype(np.int16)).view(torch.bfloat16)

# Test across layers
from safetensors import safe_open
print("=== PSNR across all expert weight tensors ===\n")
psnr_values = []
for i in range(1, 9):
    ckpt = os.path.join(ORIG_CKPT, f"model-0000{i}-of-00008.safetensors")
    if not os.path.exists(ckpt):
        continue
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "expert" not in k or "shared_expert" in k:
                continue
            # model.layers.0.mlp.experts.0.gate_proj.weight
            parts = k.split(".")
            layer_idx = int(parts[2])
            expert_idx = int(parts[5])
            weight_type = parts[6]  # gate_proj, up_proj, down_proj

            t = f.get_tensor(k).to(torch.bfloat16)
            indices = quantize_py(t)
            recovered = recover_py(indices).reshape(t.shape)
            orig_f32 = t.float().numpy().ravel()
            reco_f32 = recovered.float().numpy().ravel()
            mse = ((orig_f32 - reco_f32) ** 2).mean()
            var_val = orig_f32.var()
            psnr_var = 10 * np.log10(var_val / mse) if mse > 0 else float('inf')
            psnr_max = 10 * np.log10((orig_f32.max()**2) / mse) if mse > 0 else float('inf')
            psnr_values.append({
                "layer": layer_idx, "expert": expert_idx, "type": weight_type,
                "psnr_var": psnr_var, "psnr_max": psnr_max,
                "var": var_val, "mse": mse, "max": np.abs(orig_f32).max()
            })

psnr_var_arr = np.array([x["psnr_var"] for x in psnr_values])
psnr_max_arr = np.array([x["psnr_max"] for x in psnr_values])
print(f"Total expert tensors: {len(psnr_values)}")
print(f"Var-based PSNR:  min={psnr_var_arr.min():.1f}, max={psnr_var_arr.max():.1f}, mean={psnr_var_arr.mean():.1f}, median={np.median(psnr_var_arr):.1f} dB")
print(f"Max-based PSNR:  min={psnr_max_arr.min():.1f}, max={psnr_max_arr.max():.1f}, mean={psnr_max_arr.mean():.1f}, median={np.median(psnr_max_arr):.1f} dB")
print(f"Var-based PSNR < 25: {(psnr_var_arr < 25).sum()}, < 30: {(psnr_var_arr < 30).sum()}, < 35: {(psnr_var_arr < 35).sum()}")

# By layer
for layer in sorted(set(x["layer"] for x in psnr_values)):
    layer_var = [x["psnr_var"] for x in psnr_values if x["layer"] == layer]
    layer_max = [x["psnr_max"] for x in psnr_values if x["layer"] == layer]
    layer_var_mean = np.mean(layer_var)
    layer_max_mean = np.mean(layer_max)
    # Var distribution
    layer_vars = [x["var"] for x in psnr_values if x["layer"] == layer]
    avg_var = np.mean(layer_vars)
    print(f"  L{layer:2d}: var-PSNR={layer_var_mean:.1f}dB, max-PSNR={layer_max_mean:.1f}dB, avg_var={avg_var:.6e}")

# By weight type
for wtype in sorted(set(x["type"] for x in psnr_values)):
    wt_var = [x["psnr_var"] for x in psnr_values if x["type"] == wtype]
    print(f"  {wtype}: var-PSNR mean={np.mean(wt_var):.1f} dB")

print("\nDone.")
