"""诊断 LUT offload 文件大小"""
import os, sys, shutil
sys.path.insert(0, "/home/hh/zip_Moe/LUT_MoE")
os.environ["LUT_MOE_TEST"] = "1"

import numpy as np
import torch
import LUT_MoE
from utils.config import LUT_MoEConfig
from utils.constants import DELAY_PROFILE, COMPRESSION_RATIO_PROFILE

LUT_PATH = "/home/hh/zip_Moe/LUT_MoE/evaluation/results/dct_analysis/lut_256.npy"
CHUNKS = 5

# 加载 LUT
lut = np.load(LUT_PATH)
lut_f32 = lut.astype(np.float32)
lut_f32.sort()
lut_bf16 = torch.from_numpy(lut_f32.copy()).to(torch.bfloat16)
lut_u16_arr = lut_bf16.view(torch.int16).numpy().astype(np.uint16)

midpoints = (lut_f32[:-1] + lut_f32[1:]) / 2.0
mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)

def quantize_and_chunk(t):
    flat = t.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
    mono = np.where(flat & 0x8000, ~flat, flat ^ np.uint16(0x8000)).astype(np.uint16)
    indices = np.searchsorted(thresholds, mono).astype(np.uint8)
    chunk_size = (indices.size + CHUNKS - 1) // CHUNKS
    chunks = []
    for i in range(CHUNKS):
        s = i * chunk_size
        e = min(indices.size, (i+1) * chunk_size)
        chunks.append(np.ascontiguousarray(indices[s:e].copy()))
    empty_sm = np.zeros(0, dtype=np.uint8)
    return chunks, empty_sm

def make_config(offload_path, code_type="LUT"):
    c = LUT_MoEConfig(
        offload_path=offload_path, code_type=code_type,
        lut_path=LUT_PATH if code_type == "LUT" else "",
        caching_algorithm="LFU", num_file_chunks=CHUNKS, num_compute_threads=6,
        prefetcher_topk=0, device_memory_ratio=0.34, gpu_pool_ratio=0.6,
        hyperparam_state_margin=0.1, num_elements_per_expert=1408*2048,
        num_tensors_per_expert=3, num_expert_layers=24, num_experts=60,
        expert_topk=4, first_k_dense_replace=0,
    )
    c.decompression_delay = DELAY_PROFILE[c.code_type] / c.num_file_chunks
    c.compression_ratio = COMPRESSION_RATIO_PROFILE[c.code_type]
    c.sm_io_delay = 900
    return c

def make_engine(c, set_lut=True):
    e = LUT_MoE.lut_moe_prefetch_handle(
        c.offload_path, c.offload_file_name, c.code_type,
        c.caching_algorithm, c.device_memory_ratio, c.gpu_pool_ratio,
        c.decompression_delay, c.sm_io_delay, c.num_compute_threads,
        c.num_file_chunks, c.prefetcher_topk, c.expert_topk,
        c.num_elements_per_expert, c.num_tensors_per_expert,
        c.num_expert_layers, c.num_experts,
        c.LZ4_accelerationLevel, c.LZ4HC_compressionLevel,
        c.ZSTD_compressionLevel, c.hyperparam_state_margin, c.bind_core,
    )
    if set_lut:
        e.set_lut_table(lut_u16_arr)
    return e

def make_tensor():
    return torch.randn(1408, 2048, dtype=torch.bfloat16)

# ========== Test 1: Pipeline (individual) path ==========
print("=" * 60)
print("Test 1: LUT Pipeline (individual offload)")
print("=" * 60)

DIR1 = "/tmp/lut_moe_d1"
shutil.rmtree(DIR1, ignore_errors=True); os.makedirs(DIR1)
e1 = make_engine(make_config(DIR1))

t1 = make_tensor()
chunks1, sm1 = quantize_and_chunk(t1)

print(f"Tensor numel: {t1.numel()}, nbytes: {t1.numel() * 2}")
print(f"Indices total: {sum(c.size for c in chunks1)} bytes")
print(f"SM size: {sm1.size} bytes")
chunk_size = (t1.numel() + CHUNKS - 1) // CHUNKS

# offload 会自动建立 index entry
t0 = __import__('time').perf_counter()
e1.offload(1, t1, chunks1, sm1, True)
td = __import__('time').perf_counter() - t0

fsize1 = os.path.getsize(os.path.join(DIR1, "lut_moe_param"))
exp_pages = CHUNKS * ((chunk_size + 4095) // 4096)
print(f"File size: {fsize1} bytes = {fsize1/4096:.1f} pages ({fsize1/1024:.1f} KB)")
print(f"Expected: {exp_pages * 4096} bytes = {exp_pages} pages")
print(f"Time: {td:.2f}s")
print(f"Size per element: {fsize1 / t1.numel():.3f} bytes")

# ========== Test 2: Batch path ==========
print("\n" + "=" * 60)
print("Test 2: LUT Batch (single tensor)")
print("=" * 60)

DIR2 = "/tmp/lut_moe_d2"
shutil.rmtree(DIR2, ignore_errors=True); os.makedirs(DIR2)
e2 = make_engine(make_config(DIR2))

t2 = make_tensor()
chunks2, sm2 = quantize_and_chunk(t2)

t0 = __import__('time').perf_counter()
e2.batch_offload([2], [t2], [chunks2], [sm2])
td = __import__('time').perf_counter() - t0

fsize2 = os.path.getsize(os.path.join(DIR2, "lut_moe_param"))
print(f"File size: {fsize2} bytes = {fsize2/4096:.1f} pages")
print(f"Time: {td:.2f}s")
print(f"Pipeline vs Batch: {'MATCH' if fsize1 == fsize2 else 'DIFFER! ' + str(fsize2 - fsize1)}")

# ========== Test 3: 连续 3 个 tensor via Batch ==========
print("\n" + "=" * 60)
print("Test 3: LUT Batch 连续 3 个 tensor")
print("=" * 60)

tid = 3
for j in range(2):
    tj = make_tensor()
    chunks_j, sm_j = quantize_and_chunk(tj)
    e2.batch_offload([tid + j], [tj], [chunks_j], [sm_j])

fsize3 = os.path.getsize(os.path.join(DIR2, "lut_moe_param"))
print(f"After 3 tensors (batch): {fsize3} bytes = {fsize3/4096:.1f} pages")
print(f"Expected: {fsize1 * 3} bytes")
print(f"Match: {'YES' if fsize3 == fsize1 * 3 else 'NO - ratio=' + str(fsize3/(fsize1*3))}")

# ========== Test 4: LZ4HC 对比 ==========
print("\n" + "=" * 60)
print("Test 4: LZ4HC (original) for comparison")
print("=" * 60)

DIR4 = "/tmp/lut_moe_d4"
shutil.rmtree(DIR4, ignore_errors=True); os.makedirs(DIR4)
e4 = make_engine(make_config(DIR4, "LZ4HC"), set_lut=False)

t4 = make_tensor()
w16 = t4.detach().view(torch.int16).numpy()
exponents = ((w16 >> 7) & 0xFF).astype(np.uint8)
sign_mantissa = (((w16 >> 15) & 0x1) << 7 | (w16 & 0x7F)).astype(np.uint8)
exponents = np.ascontiguousarray(exponents.ravel())
sm4 = sign_mantissa.ravel().copy()

exp_chunks4 = []
for i in range(CHUNKS):
    s = i * chunk_size
    e = min(exponents.size, (i+1) * chunk_size)
    exp_chunks4.append(exponents[s:e].copy())

e4.offload(1, t4, exp_chunks4, sm4, True)

fsize4 = os.path.getsize(os.path.join(DIR4, "lut_moe_param"))
print(f"LZ4HC 1 tensor: {fsize4} bytes = {fsize4/4096:.1f} pages ({fsize4/1024:.1f} KB)")
print(f"LUT 1 tensor:   {fsize1} bytes = {fsize1/4096:.1f} pages ({fsize1/1024:.1f} KB)")
print(f"Size ratio LUT/LZ4HC: {fsize1/fsize4:.3f}")

# ========== Test 5: 10 个 tensor, 验证无异常 ==========
print("\n" + "=" * 60)
print("Test 5: 10 个 tensor via batch (验证扩展性)")
print("=" * 60)

DIR5 = "/tmp/lut_moe_d5"
shutil.rmtree(DIR5, ignore_errors=True); os.makedirs(DIR5)
e5 = make_engine(make_config(DIR5))

for j in range(10):
    tj = make_tensor()
    chunks_j, sm_j = quantize_and_chunk(tj)
    e5.batch_offload([100 + j], [tj], [chunks_j], [sm_j])

fsize5 = os.path.getsize(os.path.join(DIR5, "lut_moe_param"))
print(f"After 10 tensors: {fsize5} bytes = {fsize5/1e6:.3f} MB")
print(f"Expected: {fsize1 * 10} bytes = {fsize1 * 10 / 1e6:.3f} MB")
print(f"Match: {'YES' if fsize5 == fsize1 * 10 else 'NO - ratio=' + str(fsize5/(fsize1*10))}")

# 清理
for d in [DIR1, DIR2, DIR4, DIR5]:
    shutil.rmtree(d, ignore_errors=True)

# 总结
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Per-tensor LUT size: {fsize1} bytes ({fsize1/1024/1024:.2f} MB)")
print(f"Expected for 4320 tensors: {4320 * fsize1 / 1e9:.2f} GB")
print(f"Expected for dense (339 tensors): {339 * 5767168 / 1e9:.2f} GB (raw bf16)")
print(f"Expected TOTAL: {(4320 * fsize1 + 339 * 5767168) / 1e9:.2f} GB")
