"""快速 partial offload 测试（只加载 1 个 safetensors），检查文件大小 scaling"""
import os, sys, shutil, time
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch
from safetensors import safe_open
from utils.config import LUT_MoEConfig
from utils.constants import DELAY_PROFILE, COMPRESSION_RATIO_PROFILE
import LUT_MoE

OFFLOAD_DIR = "/tmp/lut_moe_partial_test"
LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
CKPT = "/home/hh/zip_Moe/LUT_MoE/models/qwen/"
CHUNKS = 5

if os.path.exists(OFFLOAD_DIR):
    shutil.rmtree(OFFLOAD_DIR)
os.makedirs(OFFLOAD_DIR)

# 加载 LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32); lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16_arr = lut_bf16.view(torch.int16).numpy().astype(np.uint16)
midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)

def quantize_to_lut(t):
    flat = t.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
    mono = np.where(flat & 0x8000, ~flat, flat ^ np.uint16(0x8000)).astype(np.uint16)
    return np.searchsorted(thresholds, mono).astype(np.uint8)

config = LUT_MoEConfig(
    offload_path=OFFLOAD_DIR, code_type="LUT", lut_path=LUT_PATH,
    caching_algorithm="LFU", num_file_chunks=CHUNKS, num_compute_threads=6,
    prefetcher_topk=0, device_memory_ratio=0.34, gpu_pool_ratio=0.6,
    hyperparam_state_margin=0.1, num_elements_per_expert=1408*2048,
    num_tensors_per_expert=3, num_expert_layers=24, num_experts=60,
    expert_topk=4, first_k_dense_replace=0,
)
config.decompression_delay = DELAY_PROFILE[config.code_type] / config.num_file_chunks
config.compression_ratio = COMPRESSION_RATIO_PROFILE[config.code_type]
config.sm_io_delay = 900

engine = LUT_MoE.lut_moe_prefetch_handle(
    config.offload_path, config.offload_file_name, config.code_type,
    config.caching_algorithm, config.device_memory_ratio, config.gpu_pool_ratio,
    config.decompression_delay, config.sm_io_delay, config.num_compute_threads,
    config.num_file_chunks, config.prefetcher_topk, config.expert_topk,
    config.num_elements_per_expert, config.num_tensors_per_expert,
    config.num_expert_layers, config.num_experts,
    config.LZ4_accelerationLevel, config.LZ4HC_compressionLevel,
    config.ZSTD_compressionLevel, config.hyperparam_state_margin, config.bind_core,
)
engine.set_lut_table(lut_u16_arr)

DTYPE = torch.bfloat16
tensor_id = 1
stats = {"sparse": 0, "dense": 0, "sparse_bytes": 0, "dense_bytes": 0}

# 只加载第一个 safetensors 文件
ckpt_file = os.path.join(CKPT, "model-00001-of-00008.safetensors")
print(f"Loading {ckpt_file}...")
t0 = time.perf_counter()

with safe_open(ckpt_file, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    for k in keys:
        t = f.get_tensor(k).to(DTYPE).to("cpu")
        is_sparse = "expert" in k and "shared_expert" not in k

        if is_sparse:
            indices = quantize_to_lut(t)
            chunk_sz = (indices.size + CHUNKS - 1) // CHUNKS
            chunks = []
            for i in range(CHUNKS):
                s = i * chunk_sz
                e = min(indices.size, (i+1) * chunk_sz)
                chunks.append(np.ascontiguousarray(indices[s:e].copy()))
            empty_sm = np.zeros(0, dtype=np.uint8)
            engine.offload(tensor_id, t, chunks, empty_sm, True)
            stats["sparse"] += 1
            stats["sparse_bytes"] += indices.size
        else:
            engine.offload(tensor_id, t, [], np.zeros(0, dtype=np.uint8), False)
            stats["dense"] += 1
            stats["dense_bytes"] += t.numel() * 2

        tensor_id += 1

td = time.perf_counter() - t0
param_file = os.path.join(OFFLOAD_DIR, "lut_moe_param")
fsize = os.path.getsize(param_file)

# 精确计算预期：sparse per-tensor + dense per-tensor (with padding)
sparse_per_tensor = 2887680  # 705 pages
# dense: writes raw bf16, padded to 4096 per tensor
dense_pages = 0
dense_expected = 0
for k in keys:
    t_info = None  # We can't re-read without opening again
    # Estimate: dense bytes already tracked

expected_sparse = stats["sparse"] * sparse_per_tensor

print(f"\nFile 1 complete in {td:.1f}s")
print(f"Sparse tensors: {stats['sparse']}")
print(f"Dense tensors:  {stats['dense']}")
print(f"Raw sparse (indices):     {stats['sparse_bytes']/1e6:.2f} MB")
print(f"Raw dense (bf16):         {stats['dense_bytes']/1e6:.2f} MB")
print(f"Expected padded sparse:   {expected_sparse/1e9:.3f} GB")
print(f"Actual file size:         {fsize/1e9:.3f} GB")

# 验证 sparse 部分的文件大小
# 每个 sparse tensor = 705 pages = 2,887,680 bytes
# 每个 dense tensor 的 bf16 raw 加上 4096 padding
# 具体计算: 每个 dense tensor 单独 write_compressed_append, padded to 4096
# 但我们需要知道每个 dense tensor 的确切大小来计算 padding

# 简化: 比较 ratio
actual_minus_sparse = fsize - expected_sparse
dense_ratio = actual_minus_sparse / stats["dense_bytes"]
print(f"File minus sparse:        {actual_minus_sparse/1e6:.2f} MB")
print(f"Dense padding ratio:      {dense_ratio:.4f} (接近 1.0 = padding 小)")

# 预估全量
exp_full_sparse = 4320 * sparse_per_tensor
# 假设 dense 共 339 tensors, 3.72 GB raw
exp_full_dense = 3.72e9  # approximate
exp_full_total = (exp_full_sparse + exp_full_dense) / 1e9
print(f"\n预计全量: {exp_full_total:.1f} GB")
print(f"  (sparse: {exp_full_sparse/1e9:.2f} GB + dense: ~{exp_full_dense/1e9:.2f} GB)")

shutil.rmtree(OFFLOAD_DIR)
print("\nDone.")
