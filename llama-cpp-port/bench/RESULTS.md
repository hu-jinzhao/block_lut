# BlockLUT CUDA Kernel Benchmark — 完整对比

## 测试环境
RTX A5000 (Ampere, 80 SMs), CUDA 12.4, 单 expert 2048×1408, B=32

## 全面对比

| 版本 | Time | Blocks | 思路 |
|:----|:----:|:------:|------|
| V0 dequant+cuBLAS | **0.87ms** | 14336 | 反量化→cuBLAS matmul（现有管线）|
| V1 fused naive | **1.23ms** | 2048 | 每行一个 block，逐元素解压+乘加 |
| V2 fused tiled smem | **5.11ms** | 256 | smem 缓存 X，多行合并（B 太小无效） |
| V3a K-tile split | **5.12ms** | 22528 | 沿 K 拆分为独立 block + reduction |
| V3b per-element | **5.59ms** | 65536 | 每 block 一个输出元素 |

## 结论

**B=32 小 batch 下 fused 无法超越两阶段。** 原因是：

1. **cuBLAS 极度优化** — tensor core、自动调优、寄存器级 tiling
2. **dequant kernel 已饱和带宽** — 390 GB/s (51% of peak)
3. **融合 kernel 的额外开销不可忽略** — 增加 block 数→warp 调度开销；增加 smem→同步开销

**预测: B≥128 时 fused 会反超。** X 矩阵 704KB 超出 L2 缓存 (6MB 但不同行冲突)，smem tiling 开始生效。

## 工程建议

对于当前系统（Jetson 边缘推理），建议使用 V0 两阶段方案：
- 已跑通全链路并验证 PPL 5.6046（近乎无损）
- 利用 ExpertCacheManager 的专家缓存避免重复反量化
- SSD 卸载 + 渐进 4/6/8bit 加载节省 I/O

Fused kernel 优化在 server 端大 batch 场景（vLLM, TensorRT-LLM）是值得投入的方向。
