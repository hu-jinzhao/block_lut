# LUT-MoE for vLLM

将 LUT 聚类量化和渐进式加载策略部署在 vLLM 推理框架下的实现。

## 概述

本项目将 [LUT-MoE](https://github.com/NJU-MINT/LUT-MoE) 的 **LUT聚类量化** 和 **渐进式加载** 策略迁移到 vLLM 推理框架，实现：

1. **BlockLUT (8-bit)**：使用 256 个质心对 128-element 块进行 K-means 聚类量化，压缩率 **0.508**
2. **NestedLUT (4/6/8-bit)**：三阶渐进量化，根据专家访问频率动态调整位宽
3. **渐进式加载**：从 SSD 按需加载专家权重，tier 0/1/2 分别对应 8/6/4 bit
4. **访问频率追踪**：通过 `ProgressiveExpertCache` 实现专家热度的统计和 tier 升降级

## 目录结构

```
vllm_lut/
├── __init__.py              # 包导出
├── config.py                # LUT-MoE vLLM 配置
├── quantizer.py             # LUT 聚类量化算法 (BlockLUT/NestedLUT)
├── moe_layer.py             # 自定义 MoE 层 (DeepseekV2MoE_LUT)
├── progressive_cache.py     # 渐进式缓存管理系统
├── patcher.py               # vLLM 模型猴子补丁
├── engine.py                # 主引擎入口 (LUT_MoE_for_vLLM)
├── wsl_compat.py            # WSL 兼容性补丁
├── benchmark_compare.py     # 交叉框架性能对比基准测试
├── run_lut_vllm.py          # vLLM LUT-MoE 基准测试运行器
├── test_import.py           # 导入测试
├── test_vllm_integration.py # vLLM 集成测试
└── setup_env.sh             # 环境变量配置脚本
```

## 环境要求

- **原生 Linux** (推荐)：vLLM V1 引擎需要 UVA 支持
- Python 3.12+
- CUDA 12.8+ (取决于 vLLM 编译版本)
- PyTorch 2.11+
- 至少 16GB GPU 显存 (用于 DeepSeek-V2-Lite)

> **注意**：vLLM 0.25.1 在 WSL2 下存在 UVA 兼容性问题。建议在原生 Linux 环境下运行基准测试。

## 安装

```bash
# 1. 确保 CUDA 运行时库可用
export LD_LIBRARY_PATH=/path/to/nvidia/cu13/lib:$LD_LIBRARY_PATH

# 2. 设置 WSL 兼容性 (仅 WSL 需要)
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# 3. 运行环境配置
source vllm_lut/setup_env.sh

# 4. 验证安装
python3 -m vllm_lut.test_import
```

## 快速开始

### 使用 Python API

```python
from vllm_lut import LUT_MoE_for_vLLM

# 基本用法 (BLOCKLUT 量化)
engine = LUT_MoE_for_vLLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    lut_config={
        "code_type": "BLOCKLUT",
        "lut_path": "/path/to/lut_codebooks",
    }
)

# 推理
output = engine.generate(["The future of AI is"])
print(output[0].outputs[0].text)
```

### 渐进式加载

```python
from vllm_lut import LUT_MoE_for_vLLM

engine = LUT_MoE_for_vLLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    lut_config={
        "code_type": "NESTEDLUT",       # 使用嵌套 LUT
        "lut_tier": 2,                   # 初始 tier (0=8bit, 1=6bit, 2=4bit)
        "enable_progressive_loading": True,  # 启用渐进式加载
        "prefetcher_topk": 3,           # 预取 top-k 专家
    }
)
```

### 更多配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `code_type` | `BLOCKLUT` | 量化类型: `BLOCKLUT` / `NESTEDLUT` / `LUT` |
| `lut_path` | `""` | LUT codebook 路径，为空则自动训练 |
| `lut_tier` | `0` | 初始 tier: 0=8bit, 1=6bit, 2=4bit |
| `enable_progressive_loading` | `False` | 是否启用 SSD 渐进式加载 |
| `gpu_cache_ratio` | `0.6` | GPU 缓存比例 |
| `prefetcher_topk` | `3` | 预取专家数量，0 为禁用 |
| `use_vllm_fused_kernel` | `True` | 是否使用 vLLM 融合 MoE 核 |

## 性能基准测试

```bash
# 1. vLLM 基准测试 (BLOCKLUT 量化)
python3 -m vllm_lut.run_lut_vllm \
    --model /path/to/deepseek-model \
    --code_type BLOCKLUT \
    --num_runs 5 \
    --max_tokens 100

# 2. vLLM 基线 (无量化)
python3 -m vllm_lut.run_lut_vllm \
    --model /path/to/deepseek-model \
    --baseline \
    --num_runs 5

# 3. 交叉框架对比
python3 -m vllm_lut.benchmark_compare \
    --model /path/to/deepseek-model \
    --mode all \
    --num_runs 3
```

## 架构说明

### LUT 量化流程

```
原始 bf16 权重
    ↓ 分块 (128 elements/block)
块归一化 (除以 block absmax)
    ↓ K-means 聚类
256-entry 量化表 (LUT codebook)
    ↓ 最近邻分配
uint8 索引 + bf16 absmax (压缩率 0.508)
```

### 渐进式加载

```
                   ┌─────────────────────┐
                   │   GPU Cache (bf16)   │
                   │  Tier 0: hot experts │
                   │  Tier 1: warm experts│
                   └──────────┬──────────┘
                              │ 未命中时解压
                   ┌──────────▼──────────┐
                   │  LUT 格式 (GPU/CPU)  │
                   │  uint8 indices       │
                   │  + bf16 codebook     │
                   └──────────┬──────────┘
                              │ 未命中时从SSD加载
                   ┌──────────▼──────────┐
                   │  SSD (渐进式位宽)    │
                   │  Tier 2: 只读4bit   │
                   │  Tier 1: 读6bit     │
                   │  Tier 0: 读8bit     │
                   └─────────────────────┘
```

### Tier 升降级规则

- **Tier 0 (HOT)**: visit_count >= 50, 缓存 bf16 权重到 GPU
- **Tier 1 (WARM)**: visit_count >= 10, 首次访问时解压
- **Tier 2 (COLD)**: visit_count < 10, 按需解压
- **降级**: 缓存淘汰时 tier 自动降级 (0→1→2)

## 对比测试

建议在两个框架上运行相同的模型和提示来对比速度：

| 对比项 | HF + LUT-MoE | vLLM + LUT-MoE |
|--------|-------------|----------------|
| 模型加载 | 通过 `from_pretrained` + 自定义 MoE 块 | 通过 vLLM `LLM` API |
| 权重量化 | 使用 C++ 后端 (`LUT-MoE.so`) | 使用 Python 纯实现 (`LUTQuantizer`) |
| SSD 卸载 | ✅ 原生支持 (C++ TensorEngine) | 通过 `ProgressiveExpertCache` 模拟 |
| 融合核 | 逐专家计算 | 可选择使用 vLLM 融合核 |
| KV Cache | 标准 HF KV Cache | PagedAttention |

## 已知问题

1. **WSL 兼容性**: vLLM 0.25.1 V1 引擎在 WSL2 下不支持 UVA。请使用原生 Linux 环境。
2. **CUDA 版本**: 确保系统中的 NVIDIA 驱动支持 vLLM 编译用的 CUDA 版本。
3. **模型支持**: 当前已测试 DeepSeek-V2，Qwen-MoE 和 Switch Transformer 的支持正在开发中。

## License

Academic Non-Commercial License. 参见项目根目录的 LICENSE 文件。
