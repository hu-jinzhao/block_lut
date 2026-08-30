# LUT-MoE：基于分块LUT与冷热异构量化的边缘MoE模型部署研究


## 一、项目背景

### 1.1 问题场景

大模型推理面临显存墙。Mixture-of-Experts (MoE) 模型虽然总参数量大，但每个 token 只激活少数专家，
这为"将非活跃专家卸载到 SSD、按需加载到 GPU"提供了可能。本工作基于此思路，
采用 bf16 拆分编码 + 两级 GPU 缓存（全精度槽位 + 压缩槽位）+ SSD offload 的底层架构，
目标是在 8GB 显存的消费级 GPU（如 RTX 5060）上运行 Qwen2-MoE-2.7B（24 层 × 60 专家，
共 4320 个专家权重矩阵）。

然而，SSD 带宽（~3 GB/s）仍然是推理延迟的主要瓶颈。对专家权重进行**有损压缩**可以
减少 I/O 量和 GPU 显存占用，是提升推理性能的关键路径。

### 1.2 核心挑战："小而精"MoE 专家矩阵难以量化

与 Dense 大模型（如 LLaMA-65B）不同，MoE 专家权重矩阵具有三个阻碍常规量化的特性：

| 特性 | 证据 | 含义 |
|------|------|------|
| **无低秩结构** | SVD 衰减曲线与同维度随机高斯矩阵几乎完全重合，无肘部拐点 | 低秩分解（SVD/LoRA）不可行 |
| **行列分布随机** | 128×128 子矩阵呈"雪花噪声"，行 RMS std=0.0011，列 RMS std=0.0014 | per-channel/per-row 结构化量化无依据 |
| **无跨专家冗余** | 60 个专家两两余弦相似度 mean=0.002，max=0.007 | 共享码本、跨专家 delta 编码不可行 |

这三项恰好是现有量化方法各自依赖的核心假设。它们在 MoE 专家场景下全部不成立。
此外，以下方向也被系统探索并排除：

- **输出感知 LS 优化**：对 240 对 expert-projection 优化后，Δcos=+7×10⁻⁶，K-means 已是最优
- **GPTQ**：4-bit cos +0.0036 over RTN，边际改善，受信息论限制
- **QuIP incoherence processing**：仅 +1.68 dB PSNR（19.16→20.84），距 6-bit 仍有 10.8 dB 差距
- **跨层 delta 编码**：同索引专家跨相邻层 cosine≈0.002，完全正交
- **4-bit 加性双 LUT / 共享 LUT+per-block 缩放**：均失败，6dB/bit 信息论天花板

---

## 二、第一部分：Block-wise LUT 量化（已完成）

### 2.1 方法

核心洞察：虽然整个专家矩阵"野"，但 **128 元素的 block 内分布可控**。

**三步流程**：

1. **Block 划分**：将专家权重（1408×2048）按 128 个元素一组切分为 block
2. **Absmax 归一化**：每个 block 除以其绝对值最大值，将值域统一到 [-1, 1]
3. **K-means 聚类**：在所有 block 的归一化值上训练 K-means (K=256)，得到 256 个质心（LUT 查找表）

**存储格式**：

```
每个元素: uint8 索引 (8-bit) → 指向 256-entry LUT 表
每个 block: bf16 absmax (16-bit) → 还原量级
```

**压缩率**：(8 + 16/128) / 16 = 8.125 / 16 = **0.508**（约为 bf16 的一半）

**GPU 推理 kernel**：

```cuda
// 1. 将 256-entry LUT 加载到 shared memory
// 2. 每个线程: value = LUT[indices[i]] * absmax[block_id]
// 3. 输出 bf16 权重矩阵
```

### 2.2 为什么 Block 归一化是可行的

**问题**：同一个专家矩阵内，不同 block 的 absmax 从 0.029 跨越到 0.193（6.6 倍差异）。
若用单一量化 scale，小 block 精度被压碎，大 block 被截断。

**解决**：每个 block 独立归一化后，所有 block 的值收敛到 [-1, 1] 内的统一光滑单峰分布。
6918 万个归一化值形成的分布被 256 个 K-means 质心完美覆盖——质心在高概率区域密集、
长尾区域稀疏，自适应数据密度。归一化空间 PSNR = 54.0 dB。

**对比**：K-means（45.03 dB）优于 NF8（44.06 dB）和 FP8 E4M3（31.87 dB），
因为 K-means 直接从数据学习质心位置，不依赖解析分布假设。

### 2.3 实验结果

| 指标 | BLOCKLUT (K-means 256) | Uniform int8 | FP8 E4M3 |
|------|------------------------|--------------|----------|
| Weight PSNR | **44.4 dB** | 43.5 dB | 31.9 dB |
| Bits/elem | 8.125 | 8.125 | 8.125 |
| 压缩率 | 0.508 | 0.508 | 0.500 |
| KL divergence vs lossless | **0.0053** | — | — |
| Output cosine similarity | **0.9986** | — | — |

- KL < 0.01 的 token 占 87%，100% token 的 KL < 0.1
- 推理输出语义正确，与 lossless baseline 几乎不可区分
- 相比 uniform int8 在同等存储预算下提高 0.9 dB

### 2.4 工程实现

修改了 11 个文件（约 265 行），涉及：

- **CUDA kernel**（`tensor_recover.cu`）：新增 `cuda_blocklut_recover_to_bf16`
- **C++ 压缩引擎**（`tensor_engine.cpp`）：BLOCKLUT codec（IdentityCompressor + SM 路径）
- **模型拓扑**（`model_topology.cpp`）：per-tensor 调用 blocklut kernel
- **Python offload**（`model_offload.py`）：量化管线（向量化 block 处理 + LUT 搜索）

---

## 三、第二部分：冷热异构量化与动态嵌套 LUT（进行中）

### 3.1 动机

虽然 Qwen MoE 的全局路由接近均匀（所有 60 专家在所有层都被使用），但在**单次对话**中，
不同专家确实有不同的激活频率。如果能根据实时访问模式对不同专家分配不同的量化精度：

- **热专家**（高频访问）→ 8-bit 高精度
- **温专家**（中频访问）→ 6-bit 中等精度
- **冷专家**（低频访问）→ 4-bit 低精度

可以在保持推理质量的同时进一步降低平均 bit 率。

**关键挑战**：专家的"冷热"程度会随着对话内容实时变化——不能静态分配，需要**动态调整**。

### 3.2 嵌套 LUT（Nested LUT）设计

#### 3.2.1 三层 LUT 结构

| Tier | 名称 | 唯一值数 | 等效位宽 | LUT 表大小 |
|------|------|---------|---------|-----------|
| 0 (Hot) | full256 | 256 | 8-bit | 512 bytes |
| 1 (Warm) | mapped64 | 64 | 6-bit | 512 bytes |
| 2 (Cold) | mapped16 | 16 | 4-bit | 512 bytes |

所有 LUT 表都是 256-entry（适应 GPU kernel 的 `__shared__ uint16_t lut_smem[256]`），
但 mapped64/mapped16 表中有大量重复值（分别只有 64/16 个唯一值）。

**GPU kernel 无需修改**——始终查 256-entry LUT，只是表中的值随 tier 改变。

#### 3.2.2 嵌套生成策略（核心创新）

初始方案：K-means K=256 → 贪婪合并到 64 和 16 → **失败**（mapped64 仅 41 dB，
241/256 的映射不是最近质心）。

**修正方案——以 6-bit 为基准**：

1. **K-means K=64** 直接在 block 归一化数据上训练 → 得 64 个质心（PSNR 50.0 dB）
2. **向下映射（64→16）**：从 64 个质心贪婪合并到 16 个 → 生成 mapped16 LUT（**待测试**）
3. **向上扩展（64→256）**：保留 64 个质心，通过插值/微调扩展为 256 个 → 生成 full256 LUT

> 此前从 K=256 直接贪婪合并到 16 仅得 37.7 dB（与 mapped64 从 K=256 合并只得 41 dB 同理）。
> 以 K=64 为基准向下合并到 16，得益于基准质心质量更高（50 dB vs 41 dB），
> 预期 mapped16 质量也会有相应改善。这是当前正在验证的关键假设。

**渐进式加载**：当专家从 4-bit 升级到 6-bit（或 6-bit 升级到 8-bit）时，
不需要从 SSD 重新加载全部权重数据，只需加载剩余的 delta bit（例如 4→8 bit 只需额外加载 4 bit），
配合对应的 LUT 表即可还原。

> **注**：当前实现中，SSD 上仍存储完整的 8-bit 索引。真正的变长位宽存储和渐进式 delta 加载
> 是下一步的工程目标。详见 §5。

#### 3.2.3 动态 Tier 切换机制

基于底层缓存管理框架，在 `cache.cpp` 中实现：

**晋升（UpdateOnHit）**——专家被命中时根据访问频率升级：
```
visit_count >= 50 → tier 0 (hot,  8-bit)
visit_count >= 10 → tier 1 (warm, 6-bit)
默认              → tier 2 (cold, 4-bit)
```

**降级（RegisterCacheSlotForNode）**——缓存槽位不足触发逐出时逐步降级：
```
tier 0 → tier 1 (hot → warm)
tier 1 → tier 2 (warm → cold)
tier 2 → 保持 tier 2
```

**Tier 变更检测**（`model_topology.cpp`）：Node 保存 `last_decompress_tier`，
当 `lut_tier != last_decompress_tier` 时强制从 SSD 重新加载并用新 LUT 表解压。

### 3.3 PPL 评估结果（max_length=8192, WikiText-2）

| 方法 | 有效位宽 | PPL | Δ vs Lossless | 说明 |
|------|:-------:|:---:|:-------------:|------|
| Lossless (bf16) | 16 bit | **7.76** | — | 原始精度基线 |
| K-means K=256 (独立) | 8.125 bit | **7.76** | **-0.00** | ✅ 无损 |
| K=64 扩展→256 | 8.125 bit | **7.76** | **-0.00** | ✅ 包含 C64/C16 |
| K-means K=64 (独立) | 6.125 bit | **7.77** | **+0.01** | ✅ 几乎无损 |
| K=16 扩展→64 | 6.125 bit | **7.77** | **+0.01** | ✅ 包含 C16 |
| K-means K=16 (独立) | 4.125 bit | **7.83** | **+0.07** | ⚠️ 轻微损失 |

> "扩展"表示渐进式嵌套码本：C16 ⊂ C64 ⊂ C256。低 bit 质心是高 bit 质心的子集，支持渐进式 bit 加载。
> 独立 K-means 无包含约束，质量略好但无法用于渐进式加载。

### 3.4 渐进式 bit 加载（Phase 2，已完成）

SSD 上按 **3-section 位平面** 存储：
- Section 1 (bits 0-3)：4-bit 基准，N/2 字节
- Section 2 (bits 4-5)：2-bit delta，N/4 字节
- Section 3 (bits 6-7)：2-bit delta，N/4 字节

冷专家只读 Section 1（50% I/O），升温时补读剩余 delta。

#### 动态 Tier 切换

**晋升（UpdateOnHit）**——专家被命中时根据访问频率升级：
| 访问量 | 目标 Tier | 位宽 |
|:------:|:---------:|:----:|
| ≥ 50 | 0 (hot) | 8-bit |
| ≥ 10 | 1 (warm) | 6-bit |
| 默认 | 2 (cold) | 4-bit |

**降级（Evict）**——缓存满触发逐出时逐步降级：
```
tier 0 → tier 1  (hot → warm, 丢掉 2-bit)
tier 1 → tier 2  (warm → cold, 丢掉 2-bit)
tier 2 → 彻底逐出，下次从 SSD 重新加载
```

### 3.5 与已有工作的对比

| 方法 | 方案 | Bit 预算 | MoE 适用性 |
|------|------|---------|-----------|
| GPTVQ (2024) | Hessian VQ + EM + SVD | 2-4 bit | 复杂，需 GPU 小时级训练 |
| QuIP# (2024) | Hadamard + 格点码本 | 2-4 bit | 推理开销大 |
| DeepSeek FP8 | 训练时在线量化 | 8 bit | 需 QAT，PTQ 场景不适用 |
| **LUT-MoE (BlockLUT)** | Block absmax + K-means LUT | 8 bit | 极简，sklearn 秒级训练，推理友好 |
| **LUT-MoE (NestLUT)** | 嵌套 LUT + 动态 tier | 4-8 bit | 自适应，渐进式加载，MoE 专用 |

---

## 四、技术路线图：从 BLOCKLUT 到 NESTEDLUT 的演进

```
全局 LUT (K-means 256, PSNR 31 dB)
    │  ❌ 全局码本不适合深层权重
    ▼
Block128 uniform int8 (PSNR 43.5 dB)
    │  ✅ 解决量级差异
    ▼
Block128 + K-means LUT 256 (PSNR 44.4 dB)  ← BLOCKLUT
    │  ✅ 数据驱动码本，比 uniform +0.9 dB
    │  ✅ KL=0.0053，输出近乎无损
    ▼
静态 8/6 混合精度（层-wise tier 分配）
    │  ✅ 浅层 8-bit + 深层 K-means 6-bit (50 dB)
    │  ✅ 推理输出正确
    ▼
动态 8/6/4 嵌套 LUT（per-expert 访问频率驱动）  ← NestLUT
    │  ✅ 渐进式码本（C16 ⊂ C64 ⊂ C256，K=256 取前 64/16）
    │  ✅ 3-section 位平面存储（冷 50% I/O、温 75% I/O）
    │  ✅ 动态 tier 晋升/降级（10→温，50→热，Evict 降级）
    ▼
渐进式 bit 加载 + 变长 SSD 存储（已完成）
    │  ✅ 冷加载只读 4-bit（I/O 减半）
    │  ✅ 升温加载 delta bit（2-bit 增量）
    │  ✅ 降级丢弃高 bit（GPU 缓存密度翻倍）
    ▼
访问频率阈值调优 + 自适应优化（规划中）
```

---

## 五、待完成工作

### 5.1 紧急

1. **4-bit 质量提升（K=64 → 16 贪婪合并）**
   - 旧方案（K=256 贪婪→16）仅 37.7 dB，已弃用
   - **当前方向**：从 K-means K=64（50 dB 基准）贪婪合并到 16，测试 mapped16 质量
   - 类比：mapped64 从 K=256 合并（41 dB）→ K-means K=64 直接训练（50 dB，+9 dB），
     预期 mapped16 从 K=64 基准合并也会有显著改善
   - 备选方向：K-means K=16 直接在数据上训练；或探索 NF（power transform）等解析方法

2. **端到端动态切换验证**
   - 在真实多轮对话场景下运行 NESTEDLUT 动态模式，验证提升/降级频率和缓存行为
   - 当前仅验证了静态 8/6 混合，动态路径尚未端到端测试（因 4-bit 质量不足导致输出退化）

### 5.2 中期

3. **渐进式 bit 加载实现**
   - 当前 SSD 存储仍是完整 8-bit 索引，未节省存储和 I/O
   - 需要实现：按当前 tier 打包存储（4/6/8 bit），升级时仅加载 delta bits
   - 配合 LUT 表的内存映射，实现无缝 tier 切换

4. **访问频率阈值调优**
   - 当前阈值（10/50）是手动设定的，需要在实际工作负载上优化
   - 考虑基于 EMA（指数移动平均）的自适应阈值，而非固定计数

5. **GPU 缓存策略适配**
   - 8-bit bf16 热专家 vs 4-bit 压缩温专家在 GPU 缓存中的空间分配策略
   - 8GB 显存下 mixed_864 的可行性分析（粗略估算：~3.5 GB dense + ~1.7 GB hot + ~2.6 GB warm）

### 5.3 长期

6. **更低位宽探索（< 4 bit）**
   - 探索 block-wise NF 量化在 3-4 bit 的表现
   - 在信息论天花板的约束下，通过混合精度（而非纯降 bit）实现平均 bit 率下降

7. **输出空间感知的混合精度**
   - 当前基于访问频率分配 bit，但 4-bit 输出扰动 per-expert 几乎均匀（cos std=0.0013）
   - 频率-based、PSNR-based、输出扰动-based 三条混合精度路径均已探索，无显著差异
   - 需要更细粒度的输出敏感度度量

---

## 六、产出总结

| 类别 | 内容 | 状态 |
|------|------|------|
| **方法** | Block128 absmax + K-means LUT 量化 | ✅ 完成 |
| **方法** | 嵌套 LUT（以 6-bit K-means 为基准，双向扩展） | ✅ 完成 |
| **方法** | 基于访问频率的动态 tier 切换 | ✅ 机制完成 |
| **系统** | BLOCKLUT 全管线集成（CUDA kernel + codec + Python offload） | ✅ 完成 |
| **系统** | NESTEDLUT 全管线集成（三层 LUT + tier 路由） | ✅ 完成 |
| **实验** | Weight PSNR、KL divergence、输出 cosine similarity 全套评估 | ✅ 完成 |
| **实验** | 与 uniform int8、FP8 E4M3、NF8、GPTQ、QuIP 的对比 | ✅ 完成 |
| **实验** | 专家特性分析（SVD、余弦相似度、block absmax 分布） | ✅ 完成 |
| **实验** | 常规量化排除分析（低秩、delta、输出感知、加性 LUT 等） | ✅ 完成 |
| **实验** | PPL 评估（max_len=8192，6 组对比） | ✅ 完成 |
| **方法** | 渐进式嵌套码本（C16 ⊂ C64 ⊂ C256） | ✅ 完成 |
| **系统** | 3-section 位平面 SSD 存储（冷 50% I/O） | ✅ 完成 |
| **系统** | 动态 tier 晋升/降级（热度驱动精度） | ✅ 完成 |
| **实验** | 动态切换端到端验证（单轮对话） | ✅ 完成 |
| **待完成** | 动态切换多轮对话验证 | 🔲 |
| **待完成** | 自适应阈值优化 | 🔲 |
| **系统** | vLLM 量化插件集成（LUTConfig + LUTFusedMoEMethod） | ✅ 完成 |
| **系统** | bf16 → LUT safetensors 离线转换管线 | ✅ 完成 |
| **系统** | CUDA 快速编码核（0.9ms/专家, 正确率 99.999%） | ✅ 完成 |
| **实验** | LUT 4bit Qwen-MoE 在 vLLM 上推理验证 | ✅ 完成 |
| **实验** | LUT 4bit vs GPTQ 4bit PSNR 对比 | ✅ LUT 20.43dB > GPTQ 17.98dB |
| **实验** | LUT 4bit vs GPTQ 4bit PPL 对比 (WikiText-2) | ✅ LUT 7.3885 < GPTQ 7.6910 |
| **实验** | bf16 baseline PPL | ✅ 7.2958 |
| **实验** | vLLM TTFT / TPOT 测量 | 🔲 需优化后重测 |
| **框架** | Llama (HuggingFace) 架构部署 | ✅ 已完成 |
| **框架** | vLLM 框架部署 | ✅ 已完成（待性能优化） |

---

## 七、评估指标

| 指标 | 含义 | 状态 | 对比对象 |
|------|------|------|---------|
| **Weight PSNR** | 权重量化重建误差（var-based） | ✅ LUT 4bit=**20.43dB**, GPTQ=**17.98dB** | GPTQ 4-bit, RTN 4-bit |
| **PPL (WikiText-2)** | 困惑度，2k context | ✅ bf16=**7.2958**, LUT 4bit=**7.3885**, GPTQ=**7.6910** | bf16, GPTQ 4bit |
| **LUT vs GPTQ 对比** | PSNR + PPL 双重验证 | ✅ LUT 优于 GPTQ（+2.45dB PSNR, -0.30 PPL） | GPTQ 4bit (Qwen官方) |
| **TTFT** | 首 token 延迟（冷启动 / 热缓存） | 🔲 待测 | Uniform int8（同 pipeline）, 原始 bf16 |
| **TTOT** | 生成 N token 总耗时 | 🔲 待测 | Uniform int8（同 pipeline）, 原始 bf16 |
| **加速比** | TTFT/TTOT 相对原始 bf16 的比值 | 🔲 基于 TTFT/TTOT 计算 | Uniform int8 |
| **压缩率** | 压缩后 / 原始 bf16 存储比 | ✅ 0.508 | Uniform int8 (0.508), FP8 (0.5), bf16 (1.0) |
| **下游正确率** | MMLU 等 benchmark 准确率 | 🔲 需 lm-eval-harness | Uniform int8, 原始 bf16 |
| **吞吐量** | tokens/s，持续生成场景 | 🔲 待测 | Uniform int8, 原始 bf16 |
| **功耗** | 推理功耗 (W) | ❌ 无硬件 | — |

## 八、多框架部署：Llama (HuggingFace) + vLLM（新增）

### 8.1 动机

原 LUT-MoE 系统基于 HuggingFace Transformers + 自定义 C++ 运行时（`LUT_MoE.so`）实现，通过猴子补丁替换 MoE 层实现渐进式加载。为了将 LUT 量化方法推广到更多推理场景，我们将其部署到 vLLM 推理框架。

### 8.2 Llama (HuggingFace) 架构部署

在 HuggingFace transformers 上的原始部署方案：

- **MoE 层替换**：猴子补丁 `DeepseekV2MoE` → `DeepseekMoEBlock`，`Qwen2MoeSparseMoeBlock` → `Qwen2MoEBlock`
- **权重加载**：`from_pretrained` 时拦截，将 bf16 专家权重拆分为 exponent chunks + sign-mantissa 写入 SSD
- **前向传播**：通过 `ExpertExecutor` → C++ ExpertDispatcher → TensorEngine 从 SSD 按需加载 + 解压
- **缓存管理**：C++ `LUT_MoECacheHandle` 实现两级 GPU+Pinned 缓存，支持 tier 晋升/降级
- **CUDA 核**：`cuda_blocklut_recover_to_bf16` 在 GPU 上将 LUT 索引还原为 bf16

### 8.3 vLLM 框架部署（已完成）

将 LUT 量化方法以 `FusedMoEMethodBase` 插件形式部署到 vLLM。

#### 核心工程实现

**1. 量化参数创建（`create_weights`）**

在模型初始化阶段直接以 uint8 类型创建专家权重参数（`w13_weight`、`w2_weight`），替代原有的 bf16 参数。配合 `w13_absmax`、`w2_absmax` 和 `_codebook` 参数存储 block 归一化因子和 LUT 查找表。uint8 参数量化使专家权重显存占用从 28.6GB 降至 16.4GB（节省 43%）。

**2. 权重加载与量化配置（`process_weights_after_loading`）**

模型加载完成后，从 config.json 读取 LUT 配置信息（`code_type`、`lut_path`），从模型目录加载预训练好的 256-entry codebook（`blocklut_256.npy`），从独立的 npz 文件加载每个专家每个 block 的 absmax 值。absmax 文件与 safetensors 分离存储，避免修改 vLLM 的权重加载流程。

**3. 手写 CUDA LUT-GEMV 融合核（`cuda_lut_gemv.py`）**

这是最核心的优化。针对 decode 阶段 batch=1 的特点，手写了一个将 codebook 查找 + absmax 缩放 + 矩阵向量乘(GEMV)融合为单个 CUDA kernel 的算子：

```cuda
// 每个 block 负责一行输出
// 1. 加载 256-entry codebook 到 shared memory
// 2. 每个线程累加部分列: 
//    accum += input[col] * codebook[indices[row*K+col]] * absmax[(row*K+col)/128]
// 3. warp-level + block-level reduction 得到最终结果
// 
// 无中间显存分配，codebook 一次性读取到 shared memory
```

相比 PyTorch 的 `gather + arange + clamp + multiply + matmul` 五步操作（2.67ms），融合 kernel 只需 0.127ms/GEMV，加速比 **13x**。

**4. 模型离线转换管线（`convert_model.py`）**

用同款 CUDA kernel（反转为编码方向）将原始 bf16 权重编码为 LUT 格式的 safetensors。支持 BlockLUT（8-bit）和 NestedLUT（4/6/8-bit 渐进式）两种编码。总转换时间约 36 秒（Qwen-MoE，4320 个专家矩阵），转换后模型从 28.6GB 降至 16.4GB。

**5. 前向推理策略（`apply`）**

针对 prefill 和 decode 两个阶段的不同特征采用差异化计算路径：

- **Decode (batch=1)**：使用手写 CUDA GEMV 融合核，只对被路由器选中的 top-4 专家进行 LUT 解算。4 个选中的专家权重以 uint8 索引存于显存，解算时直接送入 GPU kernel 内联还原为 bf16 参与矩阵乘，无需中间 bf16 缓冲。
- **Prefill (batch>1)**：由于多 token 场景下几乎所有专家都会被选中，采用一次性批量解压全部专家权重→bf16 + batched F.linear 的方式，避免逐 token 反复解压。

**6. vLLM 量化注册机制**

通过修改 vLLM 的 `quantization/__init__.py`，在 `get_quantization_config` 中添加懒加载入口。config.json 中配置 `quantization_config: {"quant_method": "lut", "code_type": "NESTEDLUT", ...}` 即可启用，无需修改 vLLM 主线代码。

#### 模型转换管线

```
bf16 safetensors (28.6GB)
    │  CUDA 编码核 (fast_encode.py, 0.9ms/专家, 36s 完成)
    ▼
uint8 indices + bf16 absmax (16.4GB)
    │  safetensors 格式，保留原始权重名
    ▼
vLLM 加载 → LUTConfig → LUTFusedMoEMethod
    │  quantization='lut' via config.json
    ▼
CUDA LUT-GEMV kernel (0.127ms) → output
```

### 8.4 当前指标

| 指标 | 值 | 说明 |
|------|:---:|------|
| 模型 | Qwen1.5-MoE-A2.7B | 24层×60专家, top-4 |
| 硬件 | RTX A5000 (24GB) | WSL2 |
| 显存 | 28.6GB→16.4GB | 节省 43% |
| TTFT | ~1.0s | 含模型加载+prefill |
| TPOT | ~200ms | decode 阶段 |
| 吞吐量 | ~4.5 tok/s | 单请求连续生成 |
| LUT GEMV kernel | 0.127ms | 融合 kernel, 13x 加速 |

### 8.5 llama.cpp 框架规划（设计中）

参照 `llama-cpp-port/` 目录下详细设计方案，核心要点：

- **GGML_TYPE 扩展**：新增 `GGML_TYPE_BLOCKLUT8/6/4` 三个枚举类型及其 type_traits
- **CUDA 反量化 kernel**：复用 `cuda_blocklut_recover_to_bf16`，注册到 ggml_cuda 流水线
- **ExpertCacheManager**：GPU 缓存池 + SSD 按需加载 + 动态 tier 升降级（冷专家 50% I/O、热专家 100% I/O）
- **build_moe_ffn 改造**：在 llama.cpp 的 MoE 前向中插入 expert 加载/解压步骤
- **渐进式 bit 加载**：4-bit 冷专家仅读取 50% 磁盘数据，升温时补读剩余 delta bits
