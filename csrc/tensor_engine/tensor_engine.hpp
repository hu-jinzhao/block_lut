// Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <filesystem>
#include <functional>
#include <mutex>
#include <atomic>
#include <variant>
#include <memory>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <list>
#include <deque>
#include <vector>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


#include "tasks.hpp"
#include "thread_pool.hpp"
#include "compressors.hpp"

#include "../base/noncopyable.hpp"
#include "../utils/logger.hpp"
#include "base/lut_moe_tensor_index.hpp"
#include "memory/cache.hpp"
#include "topology/model_topology.hpp"
#include "kernels/memory_docks.hpp"


#define OS_PAGE_SIZE 4096

namespace py = pybind11;

using LUT_MoECompressor = std::variant<
    std::unique_ptr<LZ4compressor>,
    std::unique_ptr<LZ4HCcompressor>,
    std::unique_ptr<ZSTDcompressor>,
    std::unique_ptr<IdentityCompressor>
>;

struct PythonNdArrayInfo {
    uint8_t* ptr;
    size_t size;
};


struct AsyncOffloadBuffer {
    uint32_t tensor_id;
    std::vector<std::vector<uint8_t>> exp_chunks_data;
    std::vector<uint8_t> sm_data;
    size_t sm_size;
};




struct EngineConfig{
    std::string CODE_TYPE = "LZ4HC";
    int num_file_chunks = 5;
    std::string offload_dir = "/";
    std::string offload_file = "/";
    int num_compute_workers = 8;
    double decompression_delay = 1800;
    double sm_io_delay = 2000;
    bool bind_core = false;
    int LZ4_accelerationLevel = 5;
    int LZ4HC_compressionLevel = 9;
    int ZSTD_compressionLevel = 1;
    size_t shared_mem_size = 0;
    bool is_lut = false;
    bool is_blocklut = false;
};
typedef std::shared_ptr<EngineConfig> EngineConfigPtr;



struct LUT_MoETask {

    double priority;
    NodePtr node;
    int layer_idx;
    int expert_idx;
    torch::Device src_device = DISK_DEVICE;
    torch::Device dst_device = DEFAULT_CUDA_DEVICE;
    int num_token_reqs;
    TaskState state;
    bool prediction_hit = false;
    std::shared_ptr<PreemptToken> preempt_token;
    std::atomic<int> num_pending_operations;
    uint16_t* tensor_ptr = nullptr;
    uint8_t* exp_ptr = nullptr;
    uint8_t* sm_ptr = nullptr;
    bool on_operation_done(){
        return num_pending_operations.fetch_sub(1, std::memory_order_acq_rel)==1;
    }
    void receive_ptrs(
        uint16_t* gpu_tensor_ptr,
        uint8_t* gpu_exp_ptr,
        uint8_t* gpu_sm_ptr
    ){
        this->tensor_ptr = gpu_tensor_ptr;
        this->exp_ptr = gpu_exp_ptr;
        this->sm_ptr = gpu_sm_ptr;
    }

};
typedef std::shared_ptr<LUT_MoETask> LUT_MoETaskPtr;


struct LUT_MoETaskPtrCompare {
    bool operator()(const LUT_MoETaskPtr& a,
                    const LUT_MoETaskPtr& b) const {
        return a->num_token_reqs < b->num_token_reqs; 
    }
};

struct LUT_MoESchedulerMeta {
public:
    size_t num_operations = 0;
    size_t num_sm_ios = 0;

    LUT_MoESchedulerMeta(
        size_t num_compute_workers,
        size_t num_operations_per_task,
        double io_compute_delay,
        double sm_io_delay
    ): num_compute_workers_(num_compute_workers), num_operations_per_task_(num_operations_per_task),
        io_compute_delay_(io_compute_delay), sm_io_delay_(sm_io_delay)
        {}
    
    bool add_and_update(
        bool is_typeI
    ){
        if(is_typeI){num_sm_ios += 1;}
        num_operations += num_operations_per_task_;
        double io_bottleneck = num_sm_ios * sm_io_delay_;
        double compute_bottleneck = ( num_operations / num_compute_workers_ ) * io_compute_delay_;
        return compute_bottleneck > io_bottleneck;
    }

    void reset(){
        this->num_operations = 0;
        this->num_sm_ios = 0;
    }

private:
    double io_compute_delay_;
    double sm_io_delay_;
    size_t num_operations_per_task_;
    size_t num_compute_workers_;
};





class TensorEngine: public base::noncopyable{

public:
    EngineConfigPtr config_ptr;
    std::unordered_set<LUT_MoETaskPtr> new_task_queue;
    std::deque<LUT_MoETaskPtr> scheduled_task_queue;
    std::deque<LUT_MoETaskPtr> ready_queue;
    std::mutex new_queue_mutex;
    std::mutex scheduled_queue_mutex;
    std::mutex ready_queue_mutex;
    std::condition_variable new_queue_cv;
    std::condition_variable scheduled_queue_cv;
    std::condition_variable ready_queue_cv;
    cudaStream_t tensor_stream;
private:
    LUT_MoESchedulerMeta scheduler_core;
    PreemptiveThreadPool compute_pool;
    PreemptiveThreadPool IO_thread;
    std::vector<uint8_t*> compressed_buffer;
    uint8_t* compressed_buffer_base_;
    LUT_MoECompressor compressor;
    std::vector<uint8_t*> decompression_pageable_ptrs;
    std::vector<std::pair<uint8_t*, uint8_t*>> exp_pinned_ptrs;
    std::pair<uint8_t*, uint8_t*> io_pinned_ptr;
    int fd_offload_file;
    std::vector<uint16_t> lut_table_;
    uint16_t* lut_device_ = nullptr;
    uint16_t* lut_device_mapped64_ = nullptr;
    uint16_t* lut_device_mapped16_ = nullptr;
    bool nested_lut_active_ = false;
    cudaStream_t initialize_stream();
    LUT_MoECompressor initialize_compressor(const std::string& CODE_TYPE);
public:
    TensorEngine(
        EngineConfigPtr config_ptr
    );
    ~TensorEngine();
    std::unique_ptr<uint8_t[]> compress_chunkfile_core(
        uint8_t* uncompressed_ptr,
        size_t input_size,
        size_t& compressed_size
    );
    std::future<uint8_t*> compress_submit(
        uint32_t tensor_id,
        int chunk_idx,
        uint8_t* uncompressed_ptr,
        size_t file_size
    );
    size_t write_compressed_append(
        void* data_ptr,
        size_t data_size
    );
    void PipelineCompress(
        const std::vector<py::array_t<uint8_t>>& exponents_chunks,
        const py::array_t<uint8_t>& sign_mantissa,
        uint32_t tensor_id
    );
    void BatchPipelineCompress(
        const std::vector<uint32_t>& tensor_ids,
        const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
        const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
    );
    void OffloadDense(
        torch::Tensor& tensor,
        uint32_t tensor_id
    );
    inline size_t pread_exact(
        int fd, 
        void* buf, 
        size_t count, 
        size_t offset
    );
    void ReadDense(
        uint8_t* target_gpu_ptr,
        size_t file_offset,
        size_t nbytes
    );
    void OpenOffloadFile();
    std::unique_ptr<uint8_t[]> decompress_chunkfile_core(
        uint8_t* compressed_ptr,
        size_t compressed_size,
        size_t original_size
    );
    void ToWorkers(LUT_MoETaskPtr& task);
    std::future<void> io_sign_mantissa_submit(
        LUT_MoETaskPtr task,
        size_t sm_file_offset,
        uint8_t* host_sm_pinned_ptr,
        size_t sm_size
    );
    std::future<void> io_decompress_mov_submit(
        LUT_MoETaskPtr task,
        size_t exp_file_offset,
        size_t compressed_size,
        size_t original_size,
        uint8_t* host_exp_pinned_ptr_with_offset
    );
    void task_ready_notify(const LUT_MoETaskPtr& task);
    void unregister_memory(LUT_MoETaskPtr& task);
    void LUT_MoESchedulerFunc();
    void SetLUTTable(py::array_t<uint16_t>& lut_array);
    void SetNestedLUTTables(
        py::array_t<uint16_t>& lut_full,
        py::array_t<uint16_t>& lut_mapped64,
        py::array_t<uint16_t>& lut_mapped16
    );
    const uint16_t* get_lut_device() const { return lut_device_; }
    const uint16_t* get_lut_device_for_tier(int tier) const {
        if (!nested_lut_active_ || tier <= 0) return lut_device_;
        if (tier == 1) return lut_device_mapped64_;
        return lut_device_mapped16_;
    }
};


inline size_t TensorEngine::pread_exact(
    int fd, 
    void* buf, 
    size_t count, 
    size_t offset
){
    size_t bytes_read = 0;
    char* ptr = (char*)buf;
    while (bytes_read < count) {
        ssize_t n = pread(fd, ptr + bytes_read, count - bytes_read, offset + bytes_read);
        if (n == -1) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;
        bytes_read += n;
    }
    return bytes_read;
}

extern std::unique_ptr<TensorEngine> kLUT_MoETensorEngine;