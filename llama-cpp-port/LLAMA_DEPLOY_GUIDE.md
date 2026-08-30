# LLaMA 架构上的 LUT-MoE 部署实施指南

## 目标模型

使用 LLaMA MoE 架构（如 `LLM_ARCH_LLAMA`）部署，模型文件中专家权重以 BLOCKLUT 格式存储。

## 环境

- **硬件**: NVIDIA Jetson (ARM CPU + 8GB GPU) 或 任意 CUDA 设备
- **软件**: llama.cpp (GGML + CUDA backend)

---

## 文件改动清单（详细）

### 第一部分：GGML 类型系统

#### 1. `ggml/include/ggml.h`

```c
// 在 enum ggml_type 末尾（GGML_TYPE_I1024 之后）添加：
#define GGML_TYPE_BLOCKLUT_START 64
GGML_TYPE_BLOCKLUT8  = 64,  // BlockLUT 8-bit (K=256, 8.125 bit/elem)
GGML_TYPE_BLOCKLUT6  = 65,  // BlockLUT 6-bit (K=64, 6.125 bit/elem)
GGML_TYPE_BLOCKLUT4  = 66,  // BlockLUT 4-bit (K=16, 4.125 bit/elem)
#define GGML_TYPE_BLOCKLUT_END 66

// type_traits 宏更新
// BLOCKLUT block_size = 256 (elements), type_size = 1 (byte/elem)
// blck_size = 128 + sizeof(uint16_t) for absmax
```

#### 2. `ggml/src/ggml.c`

在 `ggml_internal_get_type_traits` 或类型特性表中添加：

```c
case GGML_TYPE_BLOCKLUT8:
    {
        traits = (ggml_type_traits) {
            .type_name = "blocklut8",
            .blck_size     = 130,  // 128 elements + 1 absmax(bf16=2bytes), 对齐到256
            .type_size     = 1,    // 每元素 1 字节索引
            .is_quantized  = true,
            .to_float      = blocklut8_to_float32,   // CPU 解压
            .from_float    = NULL,  // 只读类型
            .vec_dot       = NULL,
            .vec_dot_type  = GGML_TYPE_F16,
        };
    } break;
// 同理 BLOCKLUT6, BLOCKLUT4
```

#### 3. `ggml/src/ggml-cuda/blocklut.cu` (新文件)

从 LUT-MoE 的 `tensor_recover.cu` 移植：

```cuda
// kernel 1: cuda_blocklut_recover_to_bf16 (完整 8-bit 解压)
// kernel 2: cuda_nestedlut_recover_to_bf16 (渐进式 4/6-bit 解压) 

#include "common.cuh"
#include <cuda_bf16.h>

// ---- BlockLUT 8-bit 解压 ----
// 输入: indices (uint8*), absmax (bf16*), lut (uint16*)
// 输出: bf16 weight matrix
__global__ void blocklut8_dequantize_kernel(
    uint16_t* __restrict__ output,
    const uint8_t* __restrict__ indices,
    const uint16_t* __restrict__ absmax,
    const uint16_t* __restrict__ lut,
    const int n
) {
    __shared__ uint16_t lut_smem[256];
    // ... 从 LUT-MoE cuda_blocklut_recover_to_bf16 直接复制 ...
}

// ---- NestedLUT 渐进式解压 ----
// tier: 0=8bit(全量), 1=6bit(跳过high), 2=4bit(仅low)
// 输入: 3-section bitplane packed data
__global__ void nestedlut_dequantize_kernel(
    uint16_t* __restrict__ output,
    const uint8_t* __restrict__ packed_data,
    const uint16_t* __restrict__ absmax,
    const uint16_t* __restrict__ lut,
    const int n,
    const int tier
) {
    // tier 0: 正常解压 8-bit
    // tier 1: unpack 6-bit (low+mid), 用 mapped64 LUT
    // tier 2: unpack 4-bit (low only), 用 mapped16 LUT
}
```

#### 4. `ggml/src/ggml-cuda/ggml-cuda.cu`

在 `ggml_cuda_can_mul_mat` 中添加：
```c
case GGML_TYPE_BLOCKLUT8:
case GGML_TYPE_BLOCKLUT6:
case GGML_TYPE_BLOCKLUT4:
    return true;  // 可以先解压再 matmul
```

在 matmul 流水线中添加：
```c
// ggml_cuda_mul_mat_vec_real 或 ggml_cuda_mul_mat 中：
if (src0->type == GGML_TYPE_BLOCKLUT8) {
    // 1. 创建临时 bf16 buffer
    // 2. 调用 blocklut8_dequantize_kernel
    // 3. 用解压后的 bf16 tensor 继续 matmul
}
```

**关键优化**：将反量化 kernel 的输出直接作为 matmul 的输入，避免额外的 global memory 读写。

---

### 第二部分：模型加载器

#### 5. `src/llama-model-loader.cpp`

在 `llama_model_loader::load_tensor` 中：

```cpp
// 当 tensor 类型是 BLOCKLUT 时：
// 不展开为 fp16，保持压缩态
if (tensor->type == GGML_TYPE_BLOCKLUT8 || 
    tensor->type == GGML_TYPE_BLOCKLUT6 ||
    tensor->type == GGML_TYPE_BLOCKLUT4) {
    
    // 计算压缩后的存储大小
    size_t compressed_size = ggml_row_size(tensor->type, ggml_nelements(tensor));
    
    // 分配 uint8_t 压缩存储
    tensor->data = malloc(compressed_size);
    
    // 记录文件偏移（用于运行时 SSD pread）
    file_offsets[tensor_id] = {file_offset, compressed_size, tensor->type};
    
    // 从 GGUF 读取压缩数据
    file->read_raw(tensor->data, compressed_size);
} else {
    // 原始逻辑
    tensor->data = malloc(ggml_row_size(tensor->type, ggml_nelements(tensor)));
    file->read_raw(tensor->data, ggml_row_size(tensor->type, ggml_nelements(tensor)));
}
```

加载 LUT 表：
```cpp
// 在模型元数据中读取 LUT 表
std::vector<uint16_t> lut_table;
if (ml.get_arr_data(LLM_KV_LUT_MOE_LUT_TABLE, lut_table)) {
    // 复制到 GPU 常量内存或全局内存
    cudaMemcpyToSymbol(lut_device, lut_table.data(), 
                       lut_table.size() * sizeof(uint16_t));
}
```

#### 6. `src/llama-io.h` (新文件)

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cuda_runtime.h>

struct ExpertFileOffset {
    size_t offset_4bit;    // section 1 (bits 0-3)
    size_t offset_6bit;    // section 2 (bits 4-5)
    size_t offset_8bit;    // section 3 (bits 6-7)
    size_t total_size;     // 压缩后总大小
};

class ExpertCacheManager {
public:
    static ExpertCacheManager& instance();
    
    bool init(
        int num_layers,
        int num_experts,
        size_t gpu_cache_bytes,
        const std::string& gguf_path,
        std::vector<std::vector<ExpertFileOffset>>&& offsets,
        const uint16_t* host_lut,
        const uint16_t* host_lut_mapped64,
        const uint16_t* host_lut_mapped16
    );
    
    uint16_t* get_expert_weights(
        int layer_id,
        int expert_id,
        cudaStream_t stream
    );
    
    void prefetch(int layer_id, int expert_id);
    void print_stats();
    
private:
    // ... 详见 README.md 中的 ExpertCacheManager 类定义 ...
};

// 全局单例
extern ExpertCacheManager g_expert_cache;
```

#### 7. `src/llama-io.cpp` (新文件)

实现 `ExpertCacheManager` 的全部方法，包含：
- SSD `pread()` 读取
- CPU 端 BlockLUT 解压（`blocklut8_to_float32`）
- GPU 异步传输（`cudaMemcpyAsync`）
- 缓存逐出算法

---

### 第三部分：计算图集成（最关键）

#### 核心决策：在 `ggml_cuda_mul_mat_id` 中拦截

**不要改 `build_moe_ffn` 或 `llama-graph.cpp`，而是直接修改 CUDA 后端的 MoE 专家调度函数。**

原因：MoE 的 Expert 选择（Router Top-K）发生在计算图**执行阶段**，
`build_moe_ffn` 只是把 `ggml_mul_mat_id` op 插入图中，真正的专家 ID 要等到 GPU 执行 softmax + argsort 后才确定。

#### 8. `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_mul_mat_id` 修改

在 `ggml_cuda_mul_mat_id` 函数的**通用路径**（非 mmq/mmf 快速路径）中，
在逐 expert 循环里（第 1877-1922 行）插入 cache 加载逻辑：

```cpp
// 在 ggml_cuda_mul_mat_id 的逐 expert 循环中：
for (int64_t i02 = 0; i02 < ne02; ++i02) {
    if (tokens_per_expert[i02] == 0) {
        continue;
    }

    // === LUT-MoE 插入点 ===
    // 如果源权重是 BLOCKLUT 类型，从 expert 缓存加载/解压
    void* actual_weight_data = (char*)src0->data + i02 * nb02;
    if (src0->type == GGML_TYPE_BLOCKLUT8 ||
        src0->type == GGML_TYPE_BLOCKLUT6 ||
        src0->type == GGML_TYPE_BLOCKLUT4) {
        
        // 1. 计算 expert 在模型中的 ID
        int expert_id = i02;  // 或在加载时映射
        int layer_id = current_layer_id;  // 需要在上下文传递
        
        // 2. 从 ExpertCacheManager 获取展开后的 bf16 权重
        //    首次访问触发 SSD pread + CPU 解压 + cudaMemcpyAsync
        uint16_t* decompressed = g_expert_cache.get_expert_weights(
            layer_id, expert_id, stream
        );
        
        // 3. 替换权重数据指针为展开后的 bf16 数据
        actual_weight_data = (char*)decompressed;
        
        // 4. 临时修改 src0_slice 的类型为 bf16
        src0_slice.type = GGML_TYPE_BF16;  // 或 F16
    }
    // === 结束 LUT-MoE 插入 ===

    ggml_tensor src0_slice = *src0;
    src0_slice.ne[2] = 1;
    src0_slice.nb[3] = src0_slice.nb[2];
    src0_slice.data  = actual_weight_data;  // 使用 cache 数据
    // ...
    ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
}
```

**多 token 并行优化**：
- `tokens_per_expert[i02]` 表示有多少 token 选择了专家 i02
- 在调用 `get_expert_weights` 前，可以提前并行预取所有需要的专家
- 利用 CUDA streams 重叠 pread + 解压 + 计算

#### 9. 多 token 批量加载优化

在 `ggml_cuda_mul_mat_id` 循环前，收集所有需要的专家：

```cpp
// 在逐 expert 循环前，批量预取
std::vector<int> needed_experts;
for (int64_t i02 = 0; i02 < ne02; ++i02) {
    if (tokens_per_expert[i02] > 0) {
        needed_experts.push_back(i02);
    }
}
// 批量预取——内部用线程池并行发起 SSD pread
if (src0->type >= GGML_TYPE_BLOCKLUT_START && 
    src0->type <= GGML_TYPE_BLOCKLUT_END) {
    g_expert_cache.batch_prefetch(layer_id, needed_experts, stream);
}
```

---

### 第四部分：Python 转换脚本

#### 9. `convert_hf_to_gguf.py` 扩展

新增 `--blocklut` 和 `--blocklut-k` 参数。

核心量化函数（从 `model_offload.py` 移植）：

```python
def quantize_to_blocklut(weight_fp32: np.ndarray, K: int = 256):
    """将 fp32 权重转换为 BlockLUT 格式"""
    x = weight_fp32.ravel()
    n = x.size
    bs = 128
    nb = (n + bs - 1) // bs
    
    # pad
    if nb * bs > n:
        x = np.pad(x, (0, nb * bs - n))
    
    blocks = x.reshape(nb, bs)
    absmax = np.max(np.abs(blocks), axis=1)
    absmax = np.maximum(absmax, 1e-12)
    normalized = blocks / absmax[:, np.newaxis]
    
    # K-means 量化 (使用 sklearn 或预加载的 LUT)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, n_init=3, random_state=0)
    kmeans.fit(normalized.reshape(-1, 1))
    
    # LUT 表
    lut = kmeans.cluster_centers_.flatten()  # K 个质心
    
    # 量化
    midpoints = (lut[:-1] + lut[1:]) / 2
    indices = np.searchsorted(midpoints, normalized.ravel()).astype(np.uint8)
    
    indices = indices[:n]  # 去除 padding
    
    # absmax 转 bf16 uint16
    absmax_bf16 = torch.from_numpy(absmax).to(torch.bfloat16)
    absmax_u16 = absmax_bf16.view(torch.int16).numpy().astype(np.uint16)
    
    return indices, absmax_u16, lut.astype(np.float32)
```

写入 GGUF：

```python
writer.add_tensor(
    name="model.layers.0.mlp.experts.down.weight",
    tensor_data=packed_blocklut,  # [indices_bytes + absmax_bytes]
    raw_dtype=GGML_TYPE_BLOCKLUT8
)
# 同时写入 LUT 表作为模型元数据
writer.add_key_value("lut-moe.lut_table", lut_bytes)
writer.add_key_value("lut-moe.block_size", 128)
writer.add_key_value("lut-moe.k", 256)
```

对于 NestedLUT 渐进式存储：

```python
def quantize_nestedlut(weight_fp32):
    """生成嵌套 LUT + 3-section 位平面"""
    indices, absmax_u16, lut256 = quantize_to_blocklut(weight_fp32, K=256)
    
    # 渐进式位平面
    low = (indices & 0x0F).astype(np.uint8)           # bits 0-3
    mid = ((indices >> 4) & 0x03).astype(np.uint8)    # bits 4-5
    high = ((indices >> 6) & 0x03).astype(np.uint8)   # bits 6-7
    
    # 打包
    packed_low = pack_4bit_to_8bit(low)
    packed_mid = pack_2bit_to_8bit(mid)
    packed_high = pack_2bit_to_8bit(high)
    
    packed = np.concatenate([packed_low, packed_mid, packed_high])
    return packed, absmax_u16, (lut256, lut64, lut16)
```

---

## 部署脚本

新增 `llama-cpp-port/deploy.sh`：

```bash
#!/bin/bash
# 1. 用 convert_hf_to_gguf.py 量化模型
python convert_hf_to_gguf.py \
    --model /path/to/llama-moe \
    --blocklut \
    --blocklut-k 256 \
    --outfile models/llama-moe-blocklut.gguf

# 2. 用构建好的 llama.cpp 运行
./build/bin/main \
    -m models/llama-moe-blocklut.gguf \
    --gpu-memory 4096 \          # GPU 缓存大小 (MB)
    --expert-cache-size 2048 \   # 专家缓存池大小 (MB)
    --nested-lut \               # 启用渐进式加载
    --prompt "Hello, world"
```

---

## 验证

### 正确性验证
```bash
# 比较 BlockLUT 量化模型 vs 原始模型输出
./build/bin/quantize-stats \
    -m models/llama-moe-blocklut.gguf
# 输出: PSNR, KL divergence vs baseline
```

### 性能验证
```bash
# 测试不同缓存大小下的 TPOT
for cache in 512 1024 2048 4096; do
    ./build/bin/main \
        -m models/llama-moe-blocklut.gguf \
        --expert-cache-size $cache \
        --prompt "Once upon a time" \
        -n 128 \
        --perf-output
done
```

### Jetson 特定优化
```bash
# 启用 CPU-GPU 流水线重叠
./build/bin/main \
    -m models/llama-moe-blocklut.gguf \
    --gpu-memory 4096 \
    --expert-cache-size 2048 \
    --cpu-decompress-threads 4 \  # Jetson ARM 多核解压
    --prefetch-depth 2           # 预取深度
```
