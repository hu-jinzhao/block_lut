// Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
// All rights reserved.
//
// Modifications and additions to this file are licensed under the
// Academic Non-Commercial License. See the LICENSE file in the
// project root for details.
//
// -------------------------------------------------------------------
// DERIVED FROM:
// EfficientMoE (Apache License 2.0)
// Copyright (c) EfficientMoE.
//
// The original code is licensed under the Apache License, Version 2.0.
// This file contains substantial modifications.
// -------------------------------------------------------------------



#include "lut_moe_prefetch_handle.hpp"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "base/lut_moe_tensor_handle.hpp"
#include "base/lut_moe_tensor_index.hpp"
#include "common/pytorch.hpp"
#include "common/time.hpp"
#include "memory/cache.hpp"
#include "task_scheduler.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/logger.hpp"
#include "kernels/memory_docks.hpp"


LUT_MoEPrefetchHandle::LUT_MoEPrefetchHandle(
    const std::string& prefix,
    const std::string& offload_file_name,
    const std::string& CODE_TYPE,
    const std::string& caching_algorithm,
    const double& device_memory_ratio,
    const double& gpu_pool_ratio,
    double decompression_delay,
    double sm_io_delay,
    int num_compute_threads,
    int num_file_chunks,
    int prefetcher_topk,
    int expert_topk,
    size_t num_elements_per_expert,
    size_t num_tensors_per_expert,
    int num_expert_layers,
    int num_experts,
    int LZ4_accelerationLevel,
    int LZ4HC_compressionLevel,
    int ZSTD_compressionLevel,
    double hyperparam_state_margin,
    bool bind_core
):
    prefix_(prefix),
    last_layer_id_(0),
    has_cleaned_up_resources_(false)
{
    if (prefix_.back() != '/') {
        prefix_ += '/';
    }

    kTensorIndex = std::make_unique<LUT_MoETensorIndex>();
    kLUT_MoETensorHandle = std::make_unique<LUT_MoETensorHandle>(prefix_);
    kMemoryDock = std::make_unique<MemoryDock>(
        num_experts,
        num_tensors_per_expert,
        num_file_chunks,
        num_elements_per_expert
    );

    kTopologyHandle = std::make_unique<LUT_MoETopologyHandle>();
    kTaskPool = std::make_unique<LUT_MoETaskPool>();

    cudaDeviceProp prop;
    int device_id = 0;
    cudaGetDeviceProperties(&prop, device_id);
    size_t GPU_Hardware_Total_Global_Memory = prop.totalGlobalMem;
    size_t total_memory_pool_size 
        = (size_t)( (double)GPU_Hardware_Total_Global_Memory * device_memory_ratio );

    kLUT_MoECacheHandle = std::make_unique<LUT_MoECacheHandle>(
        total_memory_pool_size,
        num_tensors_per_expert * num_elements_per_expert * sizeof(uint16_t),
        num_tensors_per_expert * num_elements_per_expert * sizeof(uint8_t),
        gpu_pool_ratio,
        1,
        num_expert_layers,
        num_experts,
        hyperparam_state_margin,
        caching_algorithm
    );
    EngineConfigPtr tensor_engine_config =  std::make_shared<EngineConfig>();
    tensor_engine_config->CODE_TYPE = CODE_TYPE;
    tensor_engine_config->decompression_delay = decompression_delay;
    tensor_engine_config->sm_io_delay = sm_io_delay;
    tensor_engine_config->num_file_chunks = num_file_chunks;
    tensor_engine_config->offload_dir = prefix_;
    tensor_engine_config->offload_file = prefix_ + offload_file_name;
    tensor_engine_config->num_compute_workers = num_compute_threads;
    tensor_engine_config->bind_core = bind_core;
    tensor_engine_config->LZ4_accelerationLevel = LZ4_accelerationLevel;
    tensor_engine_config->LZ4HC_compressionLevel = LZ4HC_compressionLevel;
    tensor_engine_config->ZSTD_compressionLevel = ZSTD_compressionLevel;
    tensor_engine_config->shared_mem_size = num_elements_per_expert * sizeof(uint8_t);
    kLUT_MoETensorEngine = std::make_unique<TensorEngine>(tensor_engine_config);
}


LUT_MoEPrefetchHandle::~LUT_MoEPrefetchHandle(){
    if (!has_cleaned_up_resources_) {
        CleanUpResources();
    }
}


void LUT_MoEPrefetchHandle::CleanUpResources() {
  kTaskPool.reset();
  kLUT_MoETensorHandle.reset();
  kTensorIndex.reset();
  kTopologyHandle.reset();
  kLUT_MoECacheHandle.reset();
  kLUT_MoETensorEngine.reset();
  has_cleaned_up_resources_ = true;
}


void LUT_MoEPrefetchHandle::SetUpDensePool(){
    size_t total_dense_size = kTopologyHandle->GetDenseNodesTotalSize();
    kLUT_MoECacheHandle->SetUpDensePool(total_dense_size);
}


void LUT_MoEPrefetchHandle::ResetAccessCounts(){
    kLUT_MoECacheHandle->reset_global_access_counts();
}


void LUT_MoEPrefetchHandle::SetLUTTable(py::array_t<uint16_t>& lut_array){
    kLUT_MoETensorEngine->SetLUTTable(lut_array);
}

void LUT_MoEPrefetchHandle::SetNestedLUTTables(
    py::array_t<uint16_t>& lut_full,
    py::array_t<uint16_t>& lut_mapped64,
    py::array_t<uint16_t>& lut_mapped16
){
    kLUT_MoETensorEngine->SetNestedLUTTables(lut_full, lut_mapped64, lut_mapped16);
}

void LUT_MoEPrefetchHandle::SetNodeLutTier(const NodePtr& node, int tier){
    if (node){
        node->lut_tier = tier;
    }
}

void LUT_MoEPrefetchHandle::SetAllSparseNodesLutTier(int tier){
    const auto& sparse_nodes = kTopologyHandle->GetSparseNodes();
    int count = 0;
    for (const auto& node : sparse_nodes){
        node->lut_tier = tier;
        count++;
    }
    std::cout << "[LUT_MoE] Set " << count << " sparse nodes to LUT tier " << tier << std::endl;
}

void LUT_MoEPrefetchHandle::OpenOffloadFile(){
    kLUT_MoETensorEngine->OpenOffloadFile();
}


uint32_t LUT_MoEPrefetchHandle::AcquireTensor(
    uint64_t& request_id,
    torch::Tensor& buffer
){
    auto tensor_id = kLUT_MoETensorHandle->GetTensorId((void*)buffer.data_ptr());

    void* old_ptr = (void*)buffer.data_ptr();
    DLOG_TRACE("Acquire tensor ", tensor_id, old_ptr);

    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    node->state = 1;

    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end() ||
        node_id_to_tensor_ids_[node->id].size() == 0) {
        node_id_to_tensor_ids_[node->id] = std::unordered_set<std::uint32_t>();
        for (auto& tensor_id : node->tensor_ids) {
            node_id_to_tensor_ids_[node->id].insert(tensor_id);
        }

        node->mutex.lock();
        std::unique_lock<std::mutex> lock(node->mutex, std::adopt_lock);
        if(node->is_sparse){
            DLOG_INFO("Acquire Tensor for Sparse, id = ", node->tensor_ids[0]);
        }
        kTaskPool->StartExec(request_id, node);
        node->cv.wait(lock, [node] { return node->state == 0; });
        lock.release();

    }
    kLUT_MoETensorHandle->SetTensor(tensor_id, buffer);
    kLUT_MoETensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
    return tensor_id;
}

void LUT_MoEPrefetchHandle::ReleaseTensor(
    std::uint64_t& request_id,
    torch::Tensor& buffer
){
    auto tensor_id = kLUT_MoETensorHandle->GetTensorId((void*)buffer.data_ptr());
    void* old_ptr = (void*)buffer.data_ptr();
    DLOG_TRACE("Release tensor ", tensor_id, old_ptr);
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end()) {
        DLOG_ERROR("Node not found in node_id_to_tensor_ids_", node->str());
        return;
    }
    auto current_layer_id = node->corr_id & 0xFFFFFFFF;
    if (
        current_layer_id != last_layer_id_ && 
        node_id_to_tensor_ids_[last_node_->id].size() != 0
    ){
        node_id_to_tensor_ids_[last_node_->id].clear();
        kTaskPool->StopExec(request_id, last_node_);
        last_node_->mutex.unlock();
    }
    last_layer_id_ = current_layer_id;
    last_node_ = node;
    node_id_to_tensor_ids_[node->id].erase(tensor_id);

    if (node_id_to_tensor_ids_[node->id].size() == 0) { 
        kTaskPool->StopExec(request_id, node); 
        node->mutex.unlock();
    }

    if (kTopologyHandle->IsLastNode(node)) {
        DLOG_TRACE("Node is last, clean up", node->str());
        request_id_to_nodes_.erase(request_id);
    }
    at::TensorOptions options;
    options = options.device(torch::kCPU);
    options = options.dtype(buffer.dtype());
    auto zero_tensor = torch::zeros({1}, options);
    buffer.set_data(zero_tensor);
    kLUT_MoETensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}


void LUT_MoEPrefetchHandle::FetchTensors(
    std::uint64_t& request_id, 
    const std::vector<std::uint32_t>& buffer
){
    for (std::uint32_t tensor_id : buffer) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        assert(!node->is_sparse);
        kTaskPool->FetchExec(request_id, node);
    }
}


void LUT_MoEPrefetchHandle::OffloadTensor(
    const std::uint32_t tensor_id,
    torch::Tensor& tensor,
    const std::vector<py::array_t<uint8_t>>& exponents_chunks,
    const py::array_t<uint8_t>& sign_mantissa,
    bool is_sparse
){
    kLUT_MoETensorHandle->StoreTensor(
        tensor_id,
        tensor,
        exponents_chunks,
        sign_mantissa,
        is_sparse
    );

    auto ckpt_index_path = prefix_ + std::string(LUT_MOE_INDEX_NAME);

    std::unique_lock<std::mutex> lock(mutex_);
    kTensorIndex->Serialize(ckpt_index_path.c_str());
}

void LUT_MoEPrefetchHandle::BatchOffloadTensor(
    const std::vector<std::uint32_t> tensor_ids,
    std::vector<torch::Tensor>& tensors,
    const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
    const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
){
    kLUT_MoETensorHandle->BatchStoreTensor(
        tensor_ids,
        tensors,
        batch_exponents_chunks,
        batch_sign_mantissa
    );

    auto ckpt_index_path = prefix_ + std::string(LUT_MOE_INDEX_NAME);

    std::unique_lock<std::mutex> lock(mutex_);
    kTensorIndex->Serialize(ckpt_index_path.c_str());
}


void LUT_MoEPrefetchHandle::RegisterTensor(
    torch::Tensor& tensor,
    const std::uint32_t tensor_id
){
    kLUT_MoETensorHandle->RegisterTensor(tensor_id, tensor);
}


void LUT_MoEPrefetchHandle::UpdateTensorMap(
    std::uint64_t old_data_ptr,
    std::uint64_t new_data_ptr
){
    kLUT_MoETensorHandle->UpdateTensorMap(
        (void*)old_data_ptr,
        (void*)new_data_ptr
    );
}


void LUT_MoEPrefetchHandle::SetTopology(
    const std::vector<
        std::tuple<
            std::string, 
            std::vector<std::vector<TensorID>>
        >
    >& topology
){
  kTopologyHandle->InitializeTopology(topology);
}


bool LUT_MoEPrefetchHandle::IsTensorOffloaded(
    const std::uint32_t tensor_id
){
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = kTensorIndex->find(tensor_id);
  bool is_offloaded = it != kTensorIndex->end();
  if (is_offloaded) {
    it->second.id = tensor_id;
  }
  return is_offloaded;
}



bool LUT_MoEPrefetchHandle::IsTensorIndexInitialized() const {
  return kLUT_MoETensorHandle->IsTensorIndexInitialized();
}


int LUT_MoEPrefetchHandle::GetNodeDefaultDevice(
    std::vector<std::uint32_t> tensor_ids
) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
  return node->default_device.index();
}


