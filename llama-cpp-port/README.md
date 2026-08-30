# LUT-MoE → llama.cpp 无损/渐进量化迁移计划

## 概述

将 LUT-MoE 的 BlockLUT / NestedLUT 量化算法移植到 llama.cpp 的 GGUF + GGML 生态中，
使得任意 MoE 架构（LLaMA MoE、Qwen2MoE、DeepSeek 等）在推理时支持：
- ✅ **BlockLUT 8-bit 量化**（PSNR 44.4 dB，PPL 无损失）
- ✅ **NestedLUT 动态 8/6/4-bit 渐进精度**（按热度自动升降级）
- ✅ **SSD 卸载 + 专家缓存池**（仅在 GPU 保留热门专家）
- ✅ **渐进式 I/O**（冷专家 50% I/O、热专家 100% I/O）

## 设计原则

1. **不改 GGML 计算图结构**：所有修改在 tensor 加载/解压层完成，不破坏 ggml 的计算 DAG
2. **复用现有 GGUF 量化框架**：新增 GGML_TYPE 枚举，无缝融入已有 dequant 流水线
3. **MoE 专家独立处理**：不影响 dense 层的现有加载逻辑

---

## 改造总览（需要改什么）

```
llama.cpp/
├── ggml/include/ggml.h              # [修改] 新增 BlockLUT 类型枚举
├── ggml/src/ggml.c                  # [修改] 注册新类型的 type_traits
├── ggml/src/ggml-cuda/
│   ├── mmid.cu                      # [修改] MoE 专家选择后触发 SSD 读取
│   ├── mmid.cuh                     
│   ├── blocklut.cu                  # [新增] BlockLUT 反量化 CUDA kernel
│   ├── blocklut.cuh                 
│   ├── ggml-cuda.cu                 # [修改] 注册 blocklut 的 dequant 路径
│   └── convert.cu                   # [修改] blocklut → fp16 转换
├── src/
│   ├── llama-model-loader.cpp       # [修改] 加载 GGUF 时处理 blocklut 类型
│   ├── llama-model-loader.h
│   ├── llama-impl.cpp               # [修改] model loading 时分配缓存池
│   ├── llama-impl.h
│   ├── llama-graph.cpp              # [修改] build_moe_ffn 中插入 expert loading
│   ├── llama-graph.h
│   ├── llama-io.h / .cpp            # [新增] 专家缓存池 + SSD I/O 管理器
│   └── models/llama.cpp             # [修改] 注册 blocklut 加载路径
└── convert_hf_to_gguf.py            # [修改] Python 侧 BlockLUT 量化管线
```

---

## 改造一：GGUF 类型系统扩展

### 1.1 新增 GGML_TYPE 枚举

在 `ggml/include/ggml.h` 中新增三个类型：

```c
// BlockLUT: uint8 索引 + bf16 absmax（block_size=128）
GGML_TYPE_BLOCKLUT8 = 64,  // 8.125 bit, K-means K=256
GGML_TYPE_BLOCKLUT6 = 65,  // 6.125 bit, K-means K=64 (nested)
GGML_TYPE_BLOCKLUT4 = 66,  // 4.125 bit, K-means K=16 (nested)
```

### 1.2 注册 type_traits

在 `ggml.c` 中，为 BLOCKLUT 类型注册：

| 属性 | 值 | 说明 |
|------|-----|------|
| `block_size` | 256 (elements) | 解压时一次处理 256 个元素 |
| `type_size` | 1 (bytes/elem) | 每元素 1 字节索引 |
| `is_quantized` | true | 是量化类型 |
| `blck_size` | 128 + 2 (absmax/bytes) | 128 元素 + 2 字节 absmax |

存储布局：
```
[uint8 indices: N bytes] [bf16 absmax: N/128 * 2 bytes]
总大小 = N + 2*ceil(N/128) bytes，压缩比 = (8 + 16/128) / 16 = 0.508
```

对于 NestedLUT 6-bit / 4-bit，采用 3-section 位平面存储：
```
Section 1 (bits 0-3): N/2 bytes  ← 冷专家只读此段
Section 2 (bits 4-5): N/4 bytes
Section 3 (bits 6-7): N/4 bytes
```

### 1.3 GGUF Key 扩展

在模型加载时，新增元信息 key：

```
LUT-MoE.blocklut     = 1              # 开启 blocklut 量化
LUT-MoE.block_size   = 128            # block 大小
LUT-MoE.lut_tier     = 0              # 初始 tier (0=8bit, 1=6bit, 2=4bit)
LUT-MoE.nested_lut   = 0              # 是否使用嵌套 LUT
LUT-MoE.lut_path     = "lut.npy"      # LUT 表文件路径（GPU 侧加载）
```

---

## 改造二：GPU 反量化 Kernel（最核心）

### 2.1 BlockLUT → bf16 反量化 CUDA Kernel

在 `ggml-cuda/blocklut.cu` 中实现：

```cuda
// 输入: indices(uint8) + absmax(bf16) + LUT(bf16×256)
// 输出: bf16 weight matrix
// 流程:
//   1. 256-entry LUT → shared memory
//   2. 每个线程处理 8 个元素
//   3. value = LUT[indices[i]] * absmax[block_id]
//   4. 写回 bf16 向量
```

复用现有 `tensor_recover.cu` 中的 `cuda_blocklut_recover_to_bf16` kernel，适配 ggml 的 buffer 管理接口。

### 2.2 NestedLUT 渐进式解压

对于三种 tier，使用统一的 kernel 但不同 LUT 表：

| Tier | LUT 表 | 位宽 | 磁盘 I/O |
|------|--------|------|----------|
| 0 (hot) | full256 (256 entries) | 8-bit | 100% (全量读取) |
| 1 (warm) | mapped64 (64 unique) | 6-bit | 75% (跳过 high section) |
| 2 (cold) | mapped16 (16 unique) | 4-bit | 50% (仅 low section) |

**GPU kernel 无需修改**——始终查 256-entry LUT，只是表中值随 tier 改变。

### 2.3 注册到 ggml_cuda 流水线

在 `ggml_cuda.cu` 的 `ggml_cuda_can_mul_mat` 和 `ggml_cuda_mul_mat` 中，
添加 BLOCKLUT 类型的分支：在执行 matmul 前先调用反量化 kernel 将 blocklut 权重展开为 bf16。

```
ggml_cuda_mul_mat()
  └─ if src0 type is BLOCKLUT*
       └─ ggml_cuda_op_blocklut_dequantize()  // 展开为 bf16
       └─ 然后执行标准的 bf16 matmul
```

---

## 改造三：专家缓存池 + SSD I/O 管理器（最复杂）

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────┐
│                   llama.cpp 推理引擎                    │
├───────────────┬──────────────────┬──────────────────┤
│  Dense 层     │  Router (Top-K)  │  MoE Expert FFN  │
│  (常驻GPU)    │  (常驻GPU)       │  (按需加载)      │
└───────────────┴──────────────────┴──────────────────┘
                        │ 选择专家
                        ▼
┌─────────────────────────────────────────────────────┐
│              Expert Cache Manager (新增)              │
├──────────────────────┬──────────────────────────────┤
│  GPU Expert Pool     │  SSD Prefetch Engine         │
│  (固定大小环形缓冲)    │  (pread + CPU 解压线程)      │
│  热门: 8-bit bf16    │  冷门: 只读 4-bit section    │
│  温: 6-bit 压缩态     │  延迟解压 + 异步传输         │
│  冷: 被逐出 → SSD    │                              │
└──────────────────────┴──────────────────────────────┘
```

### 3.2 ExpertCacheManager 类

新增 `src/llama-io.h`，作为全局单例：

```cpp
class ExpertCacheManager {
public:
    struct ExpertKey {
        int layer_id;
        int expert_id;
    };

    // 初始化：分配 GPU 缓存池 + 打开 SSD offload 文件
    bool init(
        size_t gpu_cache_size,    // GPU 缓存池大小（字节）
        int num_layers,
        int num_experts,
        const std::string& offload_path,  // SSD 上 .gguf 偏移映射
        const uint16_t* lut_host,         // LUT 查找表（主机端）
        int n_lut_entries                 // LUT 条目数 (256)
    );

    // 在 Router 选出专家后调用——确保专家权重在 GPU 上
    // 返回指向 GPU 上 bf16 权重的指针
    uint16_t* ensure_expert_on_device(
        int layer_id,
        int expert_id,
        cudaStream_t stream
    );

    // 预取：在下一层计算时主动加载预测的专家
    void prefetch_expert(
        int layer_id,
        int expert_id
    );

    // 缓存逐出回调
    void evict_if_needed();

    // 更新热度统计（在 ensure_expert_on_device 中自动调用）
    void record_hit(int layer_id, int expert_id);

private:
    struct CacheSlot {
        int layer_id;
        int expert_id;
        uint16_t* gpu_ptr;      // GPU 上 bf16 展开后的权重
        uint8_t* compressed_gpu; // GPU 上压缩态 (4/6/8-bit indices)
        int tier;               // 0=hot(8bit), 1=warm(6bit), 2=cold(4bit)
        uint64_t visit_count;
        uint64_t last_access_time;
        bool occupied;
    };
    
    std::vector<CacheSlot> slots_;
    std::mutex mutex_;
    
    // SSD 文件句柄（用于 pread）
    int offload_fd_;
    
    // LUT 表 (GPU)
    uint16_t* lut_device_;
    uint16_t* lut_device_mapped64_;
    uint16_t* lut_device_mapped16_;
    
    // 文件偏移表：[layer_id][expert_id] → {offset_4bit, offset_6bit, offset_8bit, size}
    std::vector<std::vector<FileOffsets>> file_offsets_;
    
    // 逐出策略：LFU + tier 降级
    size_t select_victim_slot();
    void demote_slot(size_t slot_id);  // 8→6→4→evict
};
```

### 3.3 Expert Loading 流程

```
ensure_expert_on_device(layer, expert):
  1. 查缓存表 → 如果已在 GPU 上:
     - record_hit()
     - 如果 tier 变更（如 4→8bit），补读 delta bit
     - 返回 gpu_ptr
  2. 缓存未命中 → 需要加载:
     a. select_victim_slot() 逐出最不常用的专家
        - 如果 victim 是 tier≥0，先降级到 tier+1
        - 如果 victim 是 tier=2 且 slot 满，直接丢弃
     b. CPU 线程发起 pread() 从 SSD 读取压缩数据
        - 当前 tier=2 (cold)：只读 4-bit section (50% I/O)
        - 当前 tier=1 (warm)：读 6-bit (75% I/O)  
        - 当前 tier=0 (hot)：读 8-bit (100% I/O)
     c. CPU 上反量化（LUT lookup + absmax 乘加）
     d. cudaMemcpyAsync 到 GPU 缓存槽位
     e. 更新缓存表
     f. 返回 gpu_ptr
```

### 3.4 SSD 文件结构

GGUF 文件中，MoE 专家的权重以 BLOCKLUT 格式存储，同时保留一个**外部偏移索引文件**：

```
model.gguf 内部:
  tensor "model.layers.0.mlp.experts.down.weight" 
    → type=BLOCKLUT8 (或 BLOCKLUT6/4)
    → data: [uint8 indices...] [bf16 absmax...]
    
external .offset 文件 (mmap 友好):
  [layer][expert][section_offset...]
  快速定位每个专家在 GGUF 文件中的位置
```

**关键设计**：不在 GGUF 中内嵌偏移表，而是生成一个轻量的 `.meta` 侧文件，
避免修改 GGUF 解析器的核心逻辑。

---

## 改造四：`build_moe_ffn` 接入点

### 4.1 核心修改位置

在 `llama-graph.cpp` 的 `build_moe_ffn` 函数中，当使用 BLOCKLUT 量化时，
**不能用 `ggml_mul_mat_id` 直接计算**（因为权重是压缩态的）。

需要插入一个**预处理步骤**：

```cpp
// 在 build_moe_ffn 中，当检测到 expert tensor 是 BLOCKLUT 类型时：
if (up_exps->type == GGML_TYPE_BLOCKLUT8) {
    // 1. 获取 Router 选出的 expert ids (selected_experts)
    // 2. 调用 ExpertCacheManager::ensure_expert_on_device() 
    //    为每个选中的 expert 加载/展开权重
    // 3. 将展开后的 bf16 权重替换到计算图中
    // 4. 继续标准 bf16 matmul 路径
    
    // 具体实现：拦截 selected_experts tensor，
    // 对每个 unique expert，调用 ensure_expert_on_device()
    // 然后用展开后的 tensor 替换 up_exps/down_exps
    // 最后 fallthrough 到标准的 ggml_mul_mat
}
```

### 4.2 两种策略

**策略 A（纯 ggml 层修改，推荐）**：
- 在 `build_moe_ffn` 中不直接改图，而是通过 `ExpertCacheManager` 在 GPU 缓存中展开权重
- 展开后的 bf16 tensor 通过 `ggml_set_name` 替换 `up_exps` 的引用
- 优点：改动最小，不需要修改 ggml_mul_mat_id 的 CUDA kernel
- 缺点：多一次 GPU memcpy（压缩态→展开态）

**策略 B（深度集成）**：
- 直接在 `ggml_mul_mat_id` 的 CUDA 实现中处理 BLOCKLUT
- 在线 decompress + matmul fused
- 优点：无额外 memcpy，带宽最优
- 缺点：需要修改 CUDA template，复杂度高

**推荐：先实现策略 A，再做策略 B 优化。**

---

## 改造五：模型加载管线

### 5.1 Python 侧：`convert_hf_to_gguf.py` 扩展

新增 `--blocklut` 参数，在转换 HuggingFace 模型到 GGUF 时执行 BlockLUT 量化：

```python
# 伪代码
if args.blocklut:
    for name, tensor in state_dict.items():
        if is_moe_expert(tensor):
            # 1. 128-block 划分
            # 2. absmax 归一化
            # 3. K-means 量化 (K=256)
            # 4. 存储为 BLOCKLUT8 类型
            gguf_writer.add_tensor(name, quantized_data, type=BLOCKLUT8)
        else:
            # dense 层保持 fp16
            gguf_writer.add_tensor(name, tensor, type=FP16)
```

对于 NestedLUT，额外生成 3-section 位平面存储：
```python
# 渐进式存储
low_4bit = indices & 0x0F          # 4-bit 基准
mid_2bit = (indices >> 4) & 0x03   # 2-bit delta
high_2bit = (indices >> 6) & 0x03  # 2-bit delta
# 打包存储（压缩率为 50%/75%/100%）
```

### 5.2 C++ 侧：加载 BLOCKLUT 权重

在 `llama_model_loader.cpp` 中，当遇到 BLOCKLUT 类型的 tensor 时：

1. 分配 `uint8_t` 类型的存储（而不是展开为 fp16）
2. 记录该 tensor 在 GGUF 文件中的偏移量
3. 记录 LUT 表（全局共享，model 级别）
4. 注册到 `ExpertCacheManager` 的 file_offsets 表
5. **不立即加载到 GPU**——等 Router 激活时才按需加载

---

## 改造六：缓存逐出策略（Cache Eviction）

### 6.1 Tier 降级机制

```
Expert 访问 → 热度计数+1

GPU 缓存满 → 触发 Evict:
  1. 找到 visit_count 最小的 slot (LFU)
  2. 根据当前 tier 降级:
     tier 0 (8-bit) → 丢弃 high 2-bit → 变为 tier 1 (6-bit)
     tier 1 (6-bit) → 丢弃 mid 2-bit → 变为 tier 2 (4-bit)
     tier 2 (4-bit) → 直接逐出，标记为 free
  
  3. 如果降级后仍需要空间，继续降级下一个 victim
```

### 6.2 提升机制

```cpp
void record_hit(int layer_id, int expert_id) {
    slot.visit_count++;
    if (slot.visit_count >= 50 && slot.tier > 0) {
        upgrade_tier(slot, 0);  // → 8-bit (需要从 SSD 补读 delta)
    } else if (slot.visit_count >= 10 && slot.tier > 1) {
        upgrade_tier(slot, 1);  // → 6-bit
    }
}
```

---

## 实现优先级路线图

### Phase 1: 最小可行产品（2-3 天）
```
[ ] GGUF 类型枚举 + type_traits
[ ] BlockLUT CUDA dequant kernel（from tensor_recover.cu）
[ ] convert_hf_to_gguf.py 支持 --blocklut 8-bit
[ ] 单个 expert 的 GPU 加载验证
```

### Phase 2: 缓存管理器（3-4 天）
```
[ ] ExpertCacheManager 基础实现
[ ] SSD pread + CPU 解压流水线
[ ] build_moe_ffn 集成
[ ] LRU/LFU 逐出
```

### Phase 3: 渐进式加载（2-3 天）
```
[ ] 3-section 位平面存储
[ ] NestedLUT 6-bit/4-bit 支持
[ ] 动态 tier 升降级
[ ] 端到端多轮对话测试
```

### Phase 4: 优化（持续）
```
[ ] Fused decompress + matmul kernel
[ ] 多 CPU 线程并行解压
[ ] 预取策略（next-layer expert prediction）
[ ] Jetson 特定优化（CPU-GPU 流水线重叠）
```

---

## 与原始 ZipMoE / LUT-MoE 代码复用关系

| 组件 | 来源 | 改造方式 |
|------|------|----------|
| BlockLUT CUDA kernel | `csrc/kernels/tensor_recover.cu` | 直接移植为 ggml-cuda kernel |
| LUT 表管理 | `csrc/tensor_engine/tensor_engine.cpp` | 简化，复用 LUT 加载逻辑 |
| 缓存池 slot 管理 | `csrc/memory/memory_manager.hpp` | 适配 ggml 的 buffer 管理 |
| 逐出策略 | `csrc/memory/cache.hpp` | 复用 CachePolicy 接口 |
| SSD I/O | `csrc/tensor_engine/tensor_engine.cpp` 的 pread | 复用 pread + decompress 模式 |
| Python 量化管线 | `runtime/model_offload.py` | 集成到 convert_hf_to_gguf.py |
| 热点计数 | `runtime/model_offload.py` | 简化 C++ 端实现 |
