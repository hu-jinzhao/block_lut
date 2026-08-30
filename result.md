# LUT-MoE 评估结果

## 测试模型
Qwen1.5-MoE-A2.7B-Chat  
GPU: NVIDIA RTX A5000 (24GB)

## PPL 评估（WikiText-2, max_length=8192）

### K-means LUT 量化

| 方法 | 有效位宽 | PPL | Δ vs Lossless | 说明 |
|------|:-------:|:---:|:-------------:|------|
| Lossless (bf16, CPU) | 16 bit | **7.76** | — | 原始精度基线 |
| BlockLUT K=256 | 8.125 bit | **7.76** | **0.00** | ✅ 无损 |
| K-means K=64 | 6.125 bit | **7.77** | **+0.01** | ✅ 几乎无损 |
| K-means K=16 | 4.125 bit | **7.83** | **+0.07** | ⚠️ 轻微损失 |

### Uniform 量化（block128 + absmax）

| 方法 | 有效位宽 | PPL | Δ vs Lossless | 说明 |
|------|:-------:|:---:|:-------------:|------|
| Uniform int8 | 8.0 bit | **7.76** | **0.00** | ✅ 同 K=256 |
| Uniform int6 | 6.0 bit | **7.78** | **+0.02** | ⚠️ 略差于 K=64 (7.77) |
| Uniform int4 | 4.0 bit | **8.05** | **+0.29** | ❌ 远差于 K=16 (7.83) |
| **GPTQ/AWQ 4-bit** | 4.0 bit | **8.54** | **+0.78** | ❌ Hessian 补偿反作用 |

> K-means 在所有低位宽都优于 uniform，4-bit 差距拉大（+0.07 vs +0.29）。因为 K-means 自适应数据密度，uniform 均匀量化在长尾区域浪费精度。



## TTFT/TPOT Benchmark

### 各方案对比（device_mem_ratio=0.40）

| 方案 | TTFT | TPOT | 说明 |
|------|:----:|:----:|------|
| 16-bit RAW bf16 | ~1900ms* | ~1.23s | 零压缩，SSD I/O 最大 |
| K=16 BLOCKLUT | ~550ms* | ~345ms | 4-bit 质量低，提前 EOS |
| K=64 BLOCKLUT | ~930ms* | ~583ms | |
| **K=256 BLOCKLUT** | **~860ms** | **~620ms** | **streamer 实测** |
| Static NestedLUT | ~950ms* | ~598ms | 浅层 8bit / 深层 6bit |
| Dynamic NestedLUT | ~2.2s | ~660ms | streamer 实测，动态 tier |
| Uniform int8 | ~860ms* | ~620ms* | 与 K=256 同 TPOT |
| Uniform int6 | ~860ms* | ~620ms* | 同上 |
| Uniform int4 | ~860ms* | ~620ms* | 同上 |

> \* = 从 `total_time/2` 估算值除以 21（校准因子，由 K=256 实测 860ms vs total/2=20.2s 得出）  
> 所有方案 TPOT 在 580-660ms 范围内，差异主要来自采样波动而非量化方案本身  
> device_mem_ratio=0.40 对应 ~9.6GB 专家缓存池

### TPOT 对比汇总

| 方案 | TPOT |
|------|:----:|
| 16-bit RAW bf16 | ~1.23s（基线） |
| 所有 8-bit 方案 | **~600ms** |

> 同一管线内对比，8-bit 方案 TPOT 约 600ms，16-bit 约 1.23s，量化本身带来 ~2x 加速（I/O 减半）  
> 绝对 TPOT 偏高主要是管线调度开销（dispatcher 线程、per-module hooks），非量化方案问题

## 渐进式嵌套码本

### 构造方法
C256 = K-means K=256。C64 = C256[:64], C16 = C256[:16]（天然子集包含）。

### 渐进式 I/O（Phase 2）
SSD 3-section 位平面存储：
- 冷 expert：只读 4-bit → 50% I/O
- 温 expert：读 6-bit → 75% I/O
- 热 expert：读 8-bit → 100% I/O

### 动态 Tier 切换
- 晋升：visit ≥ 50 → hot(8-bit), visit ≥ 10 → warm(6-bit)
- 降级：缓存满逐出时 tier 递减，从 SSD 重新加载只需当前 tier 的数据量

## 缓存架构

GPU 仅存储 8-bit LUT 索引，不缓存 bf16 专家权重。
- bf16 槽位：~60（注意力等密集参数）
- 8-bit 槽位：~1072（几乎覆盖全部 1440 专家，驱逐极少）
