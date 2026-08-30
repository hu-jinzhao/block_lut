// Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>


void lut_moe_launch_tensor_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_exp_ptr,
    const uint8_t* gpu_sm_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
);

void lut_moe_launch_lut_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_indices_ptr,
    const uint16_t* gpu_lut_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
);

void lut_moe_launch_blocklut_recover(
    uint16_t* gpu_output_tensor,
    const uint8_t* gpu_indices_ptr,
    const uint16_t* gpu_absmax_ptr,
    const uint16_t* gpu_lut_ptr,
    size_t num_elements,
    cudaStream_t tensor_stream
);

// GPU unpack for progressive bit-plane format (4-bit packed -> 8-bit)
void lut_moe_launch_unpack_4bit(
    uint8_t* gpu_data,
    size_t num_elements,
    cudaStream_t tensor_stream
);
