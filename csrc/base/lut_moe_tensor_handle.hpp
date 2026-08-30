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


#pragma once
#include <torch/extension.h>
#include "../tensor_engine/tensor_engine.hpp"
#include "lut_moe_tensor_index.hpp"
#include "noncopyable.hpp"
extern const char* LUT_MOE_PARAM_NAME;
extern const char* LUT_MOE_INDEX_NAME;
class LUT_MoETensorHandle: public base::noncopyable{

public:

    explicit LUT_MoETensorHandle(const std::string& prefix);
    ~LUT_MoETensorHandle() = default;

    void StoreTensor(
        const std::uint32_t tensor_id,
        torch::Tensor& buffer,
        const std::vector<py::array_t<uint8_t>>& exponents_chunks,
        const py::array_t<uint8_t>& sign_mantissa,
        bool is_sparse
    );
    void BatchStoreTensor(
        const std::vector<std::uint32_t> tensor_ids,
        std::vector<torch::Tensor>& buffers,
        const std::vector<std::vector<py::array_t<uint8_t>>>& batch_exponents_chunks,
        const std::vector<py::array_t<uint8_t>>& batch_sign_mantissa
    );
    void RegisterTensor(uint32_t tensor_id, torch::Tensor& buffer);
    void SetTensor(uint32_t tensor_id, torch::Tensor& buffer);
    void SetTensor(
        uint32_t tensor_id,
        torch::Tensor& buffer,
        const torch::Device& device
    );
    uint32_t GetTensorId(void* memory_ptr) const;
    void UpdateTensorMap(void* old_data_ptr, void* new_data_ptr);

    bool IsTensorIndexInitialized() const { return is_serialized_; }

    int64_t GetTensorSizeAligned(const uint32_t tensor_id) const;
    torch::TensorOptions GetTensorOptions(const uint32_t tensor_id) const;

private:
    std::string GetIndexFileName(const uint32_t file_id) const;
    std::string prefix_;
    uint32_t file_id_;
    int64_t file_offset_;
    std::unordered_map<void*, std::uint32_t> tensor_to_id_;
    std::mutex mutex_;
    bool is_serialized_ = false;
};

extern std::unique_ptr<LUT_MoETensorHandle> kLUT_MoETensorHandle;