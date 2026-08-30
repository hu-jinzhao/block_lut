// Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.



#include "memory_manager.hpp"
#include <iostream>
#include <cassert>

LUT_MoEGPUMemoryPool::LUT_MoEGPUMemoryPool(
    size_t pool_size,
    size_t slot_size,
    int pool_id,
    uint16_t* handover_ptr
):
    base_ptr_(handover_ptr), pool_size_(pool_size), slot_size_(slot_size), pool_id_(pool_id)
{
    num_slots_ = (pool_size + slot_size_ - 1) / slot_size_;
    pool_size_ = slot_size_ * num_slots_;
    if (num_slots_ == 0){ DLOG_FATAL("[LUT_MoEGPUMemoryPool] Pool size too small for even one slot!"); }

    if (base_ptr_ == nullptr){
        CUDA_CHECK( cudaMalloc(&base_ptr_, pool_size_) );
    }
    init_slots();
}

LUT_MoEGPUMemoryPool::~LUT_MoEGPUMemoryPool(){
    destroy_pool();
}

void LUT_MoEGPUMemoryPool::init_slots(){
    slots_.resize(num_slots_);
    free_slots_.reserve(num_slots_);

    uint16_t* slot_ptr = base_ptr_;
    for (size_t i = 0; i < num_slots_; i++){
        slots_[i].id = i;
        slots_[i].size = slot_size_;
        slots_[i].ptr = slot_ptr;
        free_slots_.push_back(i);

        slot_ptr = reinterpret_cast<uint16_t*>(
            reinterpret_cast<uint8_t*>(slot_ptr) + slot_size_
        );
    }
}

void LUT_MoEGPUMemoryPool::destroy_pool(){
    if (base_ptr_){
        cudaFree(base_ptr_);
        base_ptr_ = nullptr;
    }
}


void LUT_MoEGPUMemoryPool::place(
    NodePtr& node,
    size_t slot_id
){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if(slot_id >= num_slots_){ DLOG_FATAL("[LUT_MoEGPUMemoryPool] Invalid slot ID."); }
    CacheSlot& slot = slots_[slot_id];
    if(!slot.is_free()){ DLOG_FATAL("[LUT_MoEGPUMemoryPool] Slot {} is not free!.", std::to_string(slot_id) ); }
    slot.node = node;
    node_to_slot_[node->id] = slot_id;
    auto it = std::find(
        free_slots_.begin(),
        free_slots_.end(),
        slot_id
    );
    if (it != free_slots_.end()){
        free_slots_.erase(it);
    }
}


void LUT_MoEGPUMemoryPool::evict( size_t slot_id ){

    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (slot_id >= num_slots_){ DLOG_FATAL("[LUT_MoEGPUMemoryPool] Invalid slot ID."); }

    CacheSlot& slot = slots_[slot_id];
    if(slot.is_free()){
        return;
    }
    node_to_slot_.erase(slot.node->id);
    slot.reset();
    free_slots_.push_back(slot_id);
}

bool LUT_MoEGPUMemoryPool::contains( const NodePtr& node ) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return node_to_slot_.find(node->id) != node_to_slot_.end();
}

bool LUT_MoEGPUMemoryPool::has_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return !free_slots_.empty();
}

size_t LUT_MoEGPUMemoryPool::get_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[LUT_MoEGPUMemoryPool] No free slot exists.");
    }
    return free_slots_.back();
}

size_t LUT_MoEGPUMemoryPool::pop_free_slot() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[LUT_MoEGPUMemoryPool] No free slot exists.");
    }
    size_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    return slot_id;
}

CacheSlot* LUT_MoEGPUMemoryPool::get_slot( size_t slot_id ){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (slot_id>=num_slots_) { return nullptr; }
    return &slots_[slot_id];
}

const std::vector<CacheSlot>& LUT_MoEGPUMemoryPool::get_all_slots() const {
    return slots_;
}

size_t LUT_MoEGPUMemoryPool::get_slot_id_for_node(size_t node_id) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto it = node_to_slot_.find(node_id);
    if (it != node_to_slot_.end()) {
        return it->second;
    }
    return SIZE_MAX;
}

size_t LUT_MoEGPUMemoryPool::num_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return num_slots_;
}

size_t LUT_MoEGPUMemoryPool::num_free_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_slots_.size();
}


void LUT_MoEGPUMemoryPool::print_status() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    std::cout << "[LUT_MoEGPUMemoryPool] Status:\n"
                << "  Total slots: " << num_slots_ << "\n"
                << "  Free slots: " << free_slots_.size() << "\n"
                << "  Used slots: " << (num_slots_ - free_slots_.size()) << "\n"
                << "  Slot size: " << slot_size_ << " bytes\n"
                << "  Total size: " << pool_size_ << " bytes\n";
}


LUT_MoEPinnedMemoryPool::LUT_MoEPinnedMemoryPool(
    size_t pool_size,
    size_t slot_size,
    int pool_id,
    uint8_t* host_handover_ptr,
    uint8_t* device_handover_ptr
):
    host_base_ptr_(host_handover_ptr), device_base_ptr_(device_handover_ptr),
    pool_size_(pool_size), slot_size_(slot_size), pool_id_(pool_id)
{

    num_slots_ = (pool_size + slot_size_ - 1) / slot_size_;
    pool_size_ = slot_size_ * num_slots_;
    if (num_slots_ == 0){ DLOG_FATAL("[LUT_MoEGPUMemoryPool] Pool size too small for even one slot!"); }
    if (host_base_ptr_==nullptr){
        CUDA_CHECK( 
            cudaHostAlloc(&host_base_ptr_, pool_size_, cudaHostAllocMapped)
        );
        CUDA_CHECK(
            cudaHostGetDevicePointer(&device_base_ptr_, host_base_ptr_, 0)
        );
    }
    init_slots();

}


LUT_MoEPinnedMemoryPool::~LUT_MoEPinnedMemoryPool(){
    destroy_pool();
}


void LUT_MoEPinnedMemoryPool::init_slots(){
    slots_.resize(num_slots_);
    free_slots_.reserve(num_slots_);

    uint8_t* host_slot_ptr = host_base_ptr_;
    uint8_t* device_slot_ptr = device_base_ptr_;
    for (size_t i = 0; i < num_slots_; i++){
        slots_[i].id = i;
        slots_[i].size = slot_size_;
        slots_[i].ptr = std::make_pair(host_slot_ptr, device_slot_ptr);
        free_slots_.push_back(i);
        host_slot_ptr = host_slot_ptr + slot_size_;
        device_slot_ptr = device_slot_ptr + slot_size_;
    }
}


void LUT_MoEPinnedMemoryPool::destroy_pool(){
    if (host_base_ptr_){
        cudaFreeHost(host_base_ptr_);
        host_base_ptr_ = nullptr;
    }
}

void LUT_MoEPinnedMemoryPool::place(
    NodePtr& node,
    size_t slot_id
){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if(slot_id >= num_slots_){ DLOG_FATAL("[LUT_MoEPinnedMemoryPool] Invalid slot ID."); }
    SMSlot& slot = slots_[slot_id];
    if(!slot.is_free()){ DLOG_FATAL("[LUT_MoEPinnedMemoryPool] Slot {} is not free!.", std::to_string(slot_id) ); }
    slot.node = node;
    node_to_slot_[node->id] = slot_id;
    auto it = std::find(
        free_slots_.begin(),
        free_slots_.end(),
        slot_id
    );
    if (it != free_slots_.end()){
        free_slots_.erase(it);
    }

}


void LUT_MoEPinnedMemoryPool::evict( size_t slot_id ){

    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (slot_id >= num_slots_){ DLOG_FATAL("[LUT_MoEPinnedMemoryPool] Invalid slot ID."); }

    SMSlot& slot = slots_[slot_id];
    if(slot.is_free()){
        return;
    }
    node_to_slot_.erase(slot.node->id);
    slot.reset();
    free_slots_.push_back(slot_id);
}


bool LUT_MoEPinnedMemoryPool::contains( const NodePtr& node ) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return node_to_slot_.find(node->id) != node_to_slot_.end();
}


bool LUT_MoEPinnedMemoryPool::has_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return !free_slots_.empty();
}


size_t LUT_MoEPinnedMemoryPool::get_free_slot() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[LUT_MoEGPUMemoryPool] No free slot exists.");
    }
    return free_slots_.back();
}


size_t LUT_MoEPinnedMemoryPool::pop_free_slot() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (free_slots_.empty()){
        DLOG_ERROR("[LUT_MoEGPUMemoryPool] No free slot exists.");
    }
    size_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    return slot_id;
}


SMSlot* LUT_MoEPinnedMemoryPool::get_slot( size_t slot_id ){
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (slot_id>=num_slots_) { return nullptr; }
    return &slots_[slot_id];
}


const std::vector<SMSlot>& LUT_MoEPinnedMemoryPool::get_all_slots() const {
    return slots_;
}



size_t LUT_MoEPinnedMemoryPool::get_slot_id_for_node(size_t node_id) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    auto it = node_to_slot_.find(node_id);
    if (it != node_to_slot_.end()) {
        return it->second;
    }
    return SIZE_MAX;
}


size_t LUT_MoEPinnedMemoryPool::num_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return num_slots_;
}


size_t LUT_MoEPinnedMemoryPool::num_free_slots() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    return free_slots_.size();
}


void LUT_MoEPinnedMemoryPool::print_status() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    std::cout << "[LUT_MoEGPUMemoryPool] Status:\n"
                << "  Total slots: " << num_slots_ << "\n"
                << "  Free slots: " << free_slots_.size() << "\n"
                << "  Used slots: " << (num_slots_ - free_slots_.size()) << "\n"
                << "  Slot size: " << slot_size_ << " bytes\n"
                << "  Total size: " << pool_size_ << " bytes\n";
}

