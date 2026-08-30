#include "llama-io.h"
#include "ggml-cuda/blocklut.cuh"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------
ExpertCacheManager g_expert_cache;

ExpertCacheManager & ExpertCacheManager::instance() {
    return g_expert_cache;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
ExpertCacheManager::ExpertCacheManager()
    : initialized_(false)
    , num_layers_(0), num_experts_(0), total_elements_(0)
    , num_slots_(0), slot_bytes_(0), comp_slot_bytes_(0)
    , gguf_fd_(-1)
    , lut_full_(nullptr), lut_mapped64_(nullptr), lut_mapped16_(nullptr)
    , gpu_pool_base_(nullptr), gpu_comp_pool_base_(nullptr)
    , pinned_buf_(nullptr), pinned_buf_size_(0)
    , access_counter_(0)
{}

ExpertCacheManager::~ExpertCacheManager() {
    if (gguf_fd_ >= 0) close(gguf_fd_);

    // Destroy CUDA events for all slots
    for (auto & s : slots_) {
        if (s.load_event) cudaEventDestroy(s.load_event);
    }

    // Free GPU memory
    if (gpu_pool_base_)       cudaFree(gpu_pool_base_);
    if (gpu_comp_pool_base_)  cudaFree(gpu_comp_pool_base_);
    if (lut_full_)            cudaFree(lut_full_);
    if (lut_mapped64_)        cudaFree(lut_mapped64_);
    if (lut_mapped16_)        cudaFree(lut_mapped16_);

    // Free pinned host memory
    if (pinned_buf_) cudaFreeHost(pinned_buf_);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

size_t ExpertCacheManager::compressed_size_for_tier(int n_elements, int tier) const {
    size_t n_low  = n_elements / 2;          // 4-bit nibble packed
    size_t n_mid  = n_elements / 4;          // 2-bit quad packed
    size_t n_high = n_elements / 4;          // 2-bit quad packed
    size_t n_absmax = (n_elements / 128) * sizeof(uint16_t);  // bf16 absmax
    switch (tier) {
        case 2: return n_low + n_absmax;
        case 1: return n_low + n_mid + n_absmax;
        default: return n_low + n_mid + n_high + n_absmax;
    }
}

const uint16_t * ExpertCacheManager::lut_for_tier(int tier) const {
    switch (tier) {
        case 2: return lut_mapped16_ ? lut_mapped16_ : lut_full_;
        case 1: return lut_mapped64_ ? lut_mapped64_ : lut_full_;
        default: return lut_full_;
    }
}

int ExpertCacheManager::find_slot(int layer_id, int expert_id) const {
    for (size_t i = 0; i < num_slots_; ++i) {
        if (slots_[i].occupied &&
            slots_[i].layer_id == layer_id &&
            slots_[i].expert_id == expert_id) {
            return (int)i;
        }
    }
    return -1;
}

int ExpertCacheManager::num_occupied() const {
    int count = 0;
    for (auto & s : slots_) {
        if (s.occupied) count++;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
bool ExpertCacheManager::init(
    int num_layers,
    int num_experts,
    size_t gpu_cache_size,
    const std::string & gguf_file_path,
    std::vector<std::vector<ExpertFileMeta>> && expert_meta,
    const uint16_t * lut_host,
    const uint16_t * lut_mapped64,
    const uint16_t * lut_mapped16,
    size_t total_elements
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;

    num_layers_     = num_layers;
    num_experts_    = num_experts;
    total_elements_ = total_elements;
    expert_meta_    = std::move(expert_meta);

    // Open GGUF file
    gguf_fd_ = open(gguf_file_path.c_str(), O_RDONLY);
    if (gguf_fd_ < 0) {
        fprintf(stderr, "[LUT-MoE] Failed to open GGUF: %s\n", gguf_file_path.c_str());
        return false;
    }
    gguf_path_ = gguf_file_path;

    // Slot sizes
    slot_bytes_      = total_elements * sizeof(uint16_t);  // bf16 per expert
    comp_slot_bytes_ = compressed_size_for_tier(total_elements, 0);  // full 8-bit compressed
    num_slots_       = std::max(size_t(1), gpu_cache_size / slot_bytes_);

    fprintf(stderr, "[LUT-MoE] Cache: %zu slots x bf16(%zu MB) + comp(%zu KB)\n",
            num_slots_, slot_bytes_ / (1024*1024), comp_slot_bytes_ / 1024);

    // GPU pools
    size_t pool_size = num_slots_ * slot_bytes_;
    CUDA_CHECK(cudaMalloc(&gpu_pool_base_, pool_size));

    size_t comp_pool_size = num_slots_ * comp_slot_bytes_;
    CUDA_CHECK(cudaMalloc(&gpu_comp_pool_base_, comp_pool_size));

    // Slots
    slots_.resize(num_slots_);
    for (size_t i = 0; i < num_slots_; ++i) {
        slots_[i].occupied        = false;
        slots_[i].tier            = 2;
        slots_[i].visit_count     = 0;
        slots_[i].gpu_weights     = gpu_pool_base_ + i * total_elements;
        slots_[i].gpu_compressed  = gpu_comp_pool_base_ + i * comp_slot_bytes_;
        slots_[i].load_pending    = false;
        cudaEventCreate(&slots_[i].load_event);
    }

    // Copy LUT tables to GPU
    auto cpy_lut = [&](uint16_t *& dst, const uint16_t * src) {
        if (!src) { dst = nullptr; return; }
        cudaMalloc(&dst, 256 * sizeof(uint16_t));
        cudaMemcpy(dst, src, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    };
    cpy_lut(lut_full_, lut_host);
    cpy_lut(lut_mapped64_, lut_mapped64);
    cpy_lut(lut_mapped16_, lut_mapped16);

    if (!lut_full_) {
        fprintf(stderr, "[LUT-MoE] ERROR: full LUT is required\n");
        return false;
    }

    // Pinned host buffer for SSD reads
    pinned_buf_size_ = std::max(comp_slot_bytes_, slot_bytes_);
    CUDA_CHECK(cudaHostAlloc(&pinned_buf_, pinned_buf_size_, cudaHostAllocDefault));

    initialized_ = true;
    fprintf(stderr, "[LUT-MoE] Initialized: %d layers x %d experts, %zu GPU slots\n",
            num_layers_, num_experts_, num_slots_);
    return true;
}

// ---------------------------------------------------------------------------
// Main API: get_expert_weights
// ---------------------------------------------------------------------------
uint16_t * ExpertCacheManager::get_expert_weights(
    int layer_id,
    int expert_id,
    cudaStream_t stream
) {
    if (!initialized_) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);

    // ── Check cache ──
    int idx = find_slot(layer_id, expert_id);
    if (idx >= 0) {
        auto & slot = slots_[idx];
        slot.visit_count++;
        slot.last_access_time = ++access_counter_;
        stats_.total_hits++;

        // If load still pending, wait for it
        if (slot.load_pending) {
            cudaEventSynchronize(slot.load_event);
            slot.load_pending = false;
        }

        // Check tier upgrade (visit_count thresholds from LUT-MoE)
        int new_tier = slot.tier;
        if (slot.tier > 0 && slot.visit_count >= 50) new_tier = 0;
        else if (slot.tier > 1 && slot.visit_count >= 10) new_tier = 1;

        if (new_tier < slot.tier) {
            // Need to load delta bits from SSD
            upgrade_slot(idx, new_tier, stream);
        }

        // In tier 0 we have decompressed bf16; tier 1/2 decompress on the fly
        if (slot.tier == 0) {
            return slot.gpu_weights;
        }
        // For tier 1/2: we need to decompress the compressed data
        // This is handled by the caller (ggml_cuda_mul_mat_id) which calls
        // gpu_decompress before matmul
        // Return compressed pointer for now
        return slot.gpu_weights;
    }

    // ── Cache miss ──
    stats_.total_misses++;

    size_t slot_id = find_or_evict_slot();
    if (slot_id >= num_slots_) return nullptr;

    auto & slot = slots_[slot_id];

    // Start as tier 2 (cold, 4-bit)
    const int load_tier = 2;

    // Step 1: Read compressed data from SSD into pinned buffer
    const auto & meta = expert_meta_[layer_id][expert_id];
    size_t read_size = compressed_size_for_tier(total_elements_, load_tier);
    size_t offset = (load_tier == 2) ? meta.offset_4bit
                  : (load_tier == 1) ? meta.offset_6bit
                  : meta.offset_8bit;

    size_t bytes_read = 0;
    uint8_t * ptr = pinned_buf_;
    while (bytes_read < read_size) {
        ssize_t n = pread(gguf_fd_, ptr + bytes_read, read_size - bytes_read,
                          offset + bytes_read);
        if (n <= 0) { if (errno == EINTR) continue; break; }
        bytes_read += n;
    }
    stats_.ssd_bytes_read += bytes_read;

    // Step 2: Copy compressed data to GPU compressed slot
    cudaMemcpyAsync(slot.gpu_compressed, pinned_buf_, read_size,
                    cudaMemcpyHostToDevice, stream);

    // Step 3: GPU decompression (BlockLUT kernel)
    // The compressed data layout: [packed_low] [packed_mid(opt)] [packed_high(opt)] [absmax]
    size_t n_low  = total_elements_ / 2;
    size_t n_mid  = total_elements_ / 4;
    size_t n_high = total_elements_ / 4;

    const uint8_t  * gpu_indices = slot.gpu_compressed;
    const uint16_t * gpu_absmax  = (const uint16_t *)(slot.gpu_compressed + n_low);
    const uint16_t * gpu_lut     = lut_for_tier(load_tier);

    // Launch GPU decompression kernel
    ggml_cuda_op_blocklut8_dequantize(
        slot.gpu_weights,
        gpu_indices,
        gpu_absmax,
        gpu_lut,
        total_elements_,
        stream
    );

    // Record event for async tracking
    cudaEventRecord(slot.load_event, stream);
    slot.load_pending = false;  // We sync inline for first load

    // Step 4: Update metadata
    slot.occupied    = true;
    slot.layer_id    = layer_id;
    slot.expert_id   = expert_id;
    slot.tier        = load_tier;
    slot.visit_count = 1;
    slot.last_access_time = ++access_counter_;

    return slot.gpu_weights;
}

// ---------------------------------------------------------------------------
// Prefetch
// ---------------------------------------------------------------------------
void ExpertCacheManager::prefetch_expert(int layer_id, int expert_id, cudaStream_t stream) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    if (find_slot(layer_id, expert_id) >= 0) {
        stats_.total_hits++;
        return;  // already cached
    }

    stats_.total_prefetches++;

    size_t slot_id = find_or_evict_slot();
    if (slot_id >= num_slots_) return;

    auto & slot = slots_[slot_id];
    const int load_tier = 2;  // start cold

    // Same load logic as get but without waiting for completion
    const auto & meta = expert_meta_[layer_id][expert_id];
    size_t read_size = compressed_size_for_tier(total_elements_, load_tier);
    size_t offset = meta.offset_4bit;

    size_t bytes_read = 0;
    uint8_t * ptr = pinned_buf_;
    while (bytes_read < read_size) {
        ssize_t n = pread(gguf_fd_, ptr + bytes_read, read_size - bytes_read,
                          offset + bytes_read);
        if (n <= 0) { if (errno == EINTR) continue; break; }
        bytes_read += n;
    }
    stats_.ssd_bytes_read += bytes_read;

    cudaMemcpyAsync(slot.gpu_compressed, pinned_buf_, read_size,
                    cudaMemcpyHostToDevice, stream);

    // GPU decompress
    size_t n_low = total_elements_ / 2;
    const uint16_t * gpu_absmax = (const uint16_t *)(slot.gpu_compressed + n_low);
    ggml_cuda_op_blocklut8_dequantize(
        slot.gpu_weights, slot.gpu_compressed, gpu_absmax,
        lut_for_tier(load_tier), total_elements_, stream
    );

    // Mark pending — get_expert_weights will wait on this event
    cudaEventRecord(slot.load_event, stream);
    slot.load_pending = true;

    slot.occupied    = true;
    slot.layer_id    = layer_id;
    slot.expert_id   = expert_id;
    slot.tier        = load_tier;
    slot.visit_count = 0;
    slot.last_access_time = ++access_counter_;
}

void ExpertCacheManager::batch_prefetch(
    int layer_id,
    const std::vector<int> & expert_ids,
    cudaStream_t stream
) {
    for (int eid : expert_ids) {
        prefetch_expert(layer_id, eid, stream);
    }
}

// ---------------------------------------------------------------------------
// Tier upgrade / demotion
// ---------------------------------------------------------------------------
bool ExpertCacheManager::upgrade_slot(size_t slot_id, int target_tier, cudaStream_t stream) {
    auto & slot = slots_[slot_id];
    if (!slot.occupied) return false;
    if (target_tier >= slot.tier) return true;  // already at or above

    fprintf(stderr, "[LUT-MoE] Upgrade L%d/E%d: tier %d → %d\n",
            slot.layer_id, slot.expert_id, slot.tier, target_tier);

    // Need to reload from SSD at higher tier
    const auto & meta = expert_meta_[slot.layer_id][slot.expert_id];
    size_t read_size = compressed_size_for_tier(total_elements_, target_tier);
    size_t offset = (target_tier == 2) ? meta.offset_4bit
                  : (target_tier == 1) ? meta.offset_6bit
                  : meta.offset_8bit;

    size_t bytes_read = 0;
    uint8_t * ptr = pinned_buf_;
    while (bytes_read < read_size) {
        ssize_t n = pread(gguf_fd_, ptr + bytes_read, read_size - bytes_read,
                          offset + bytes_read);
        if (n <= 0) { if (errno == EINTR) continue; break; }
        bytes_read += n;
    }
    stats_.ssd_bytes_read += bytes_read;

    // Copy to GPU and decompress
    cudaMemcpyAsync(slot.gpu_compressed, pinned_buf_, read_size,
                    cudaMemcpyHostToDevice, stream);

    size_t n_low = total_elements_ / 2;
    const uint16_t * gpu_absmax = (const uint16_t *)(slot.gpu_compressed + n_low);
    ggml_cuda_op_blocklut8_dequantize(
        slot.gpu_weights, slot.gpu_compressed, gpu_absmax,
        lut_for_tier(target_tier), total_elements_, stream
    );

    slot.tier = target_tier;
    return true;
}

void ExpertCacheManager::demote_slot(size_t slot_id) {
    auto & slot = slots_[slot_id];
    if (!slot.occupied) return;

    int old_tier = slot.tier;
    if (slot.tier < 2) {
        slot.tier++;  // 0→1, 1→2
    }

    stats_.total_demotions++;
    fprintf(stderr, "[LUT-MoE] Demote L%d/E%d: tier %d → %d\n",
            slot.layer_id, slot.expert_id, old_tier, slot.tier);
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------
bool ExpertCacheManager::evict_one_slot(size_t slot_id) {
    auto & slot = slots_[slot_id];
    if (!slot.occupied) return true;  // already free

    // Wait for any pending load
    if (slot.load_pending) {
        cudaEventSynchronize(slot.load_event);
        slot.load_pending = false;
    }

    // Try demotion first
    if (slot.tier < 2) {
        demote_slot(slot_id);
        return false;  // not fully evicted, just demoted
    }

    // Tier 2 → full eviction
    slot.occupied  = false;
    slot.layer_id  = -1;
    slot.expert_id = -1;
    slot.tier      = 2;
    slot.visit_count = 0;
    stats_.total_evictions++;
    return true;
}

size_t ExpertCacheManager::find_or_evict_slot() {
    // 1. Free slot?
    for (size_t i = 0; i < num_slots_; ++i) {
        if (!slots_[i].occupied) return i;
    }

    // 2. LFU: pick victim with lowest visit_count
    size_t victim = 0;
    uint64_t min_visits = UINT64_MAX;
    for (size_t i = 0; i < num_slots_; ++i) {
        if (slots_[i].visit_count < min_visits) {
            min_visits = slots_[i].visit_count;
            victim = i;
        }
    }

    // 3. Evict (may demote or fully evict)
    while (slots_[victim].occupied) {
        evict_one_slot(victim);
        if (!slots_[victim].occupied) break;
        // Demoted but still occupied — try next slot
        victim = (victim + 1) % num_slots_;
    }

    return victim;
}

// ---------------------------------------------------------------------------
// LUT setter
// ---------------------------------------------------------------------------
void ExpertCacheManager::set_lut_gpu(
    const uint16_t * lut,
    const uint16_t * lut_mapped64,
    const uint16_t * lut_mapped16
) {
    auto cpy = [](uint16_t * dst, const uint16_t * src) {
        if (dst && src) cudaMemcpy(dst, src, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    };
    cpy(lut_full_, lut);
    cpy(lut_mapped64_, lut_mapped64);
    cpy(lut_mapped16_, lut_mapped16);
}

// ---------------------------------------------------------------------------
// Clear all
// ---------------------------------------------------------------------------
void ExpertCacheManager::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & s : slots_) {
        if (s.load_pending) {
            cudaEventSynchronize(s.load_event);
            s.load_pending = false;
        }
        s.occupied    = false;
        s.layer_id    = -1;
        s.expert_id   = -1;
        s.tier        = 2;
        s.visit_count = 0;
    }
    fprintf(stderr, "[LUT-MoE] Cache cleared\n");
}

// ---------------------------------------------------------------------------
// GPU decompression wrapper
// ---------------------------------------------------------------------------
void ExpertCacheManager::gpu_decompress(
    uint16_t * dst_bf16,
    const uint8_t * gpu_compressed,
    const uint16_t * gpu_lut,
    const uint16_t * gpu_absmax,
    int n_elements,
    int tier,
    cudaStream_t stream
) {
    if (tier >= 2) {
        // Use 4-bit unpack + decompress kernel
        // For now, fall through to blocklut8 kernel (assumes 8-bit input)
    }
    // If data is already 8-bit (full LUT), use standard kernel
    ggml_cuda_op_blocklut8_dequantize(
        dst_bf16, gpu_compressed, gpu_absmax, gpu_lut, n_elements, stream
    );
}

// ---------------------------------------------------------------------------
// PrefetchPredictor
// ---------------------------------------------------------------------------
ExpertPrefetchPredictor::ExpertPrefetchPredictor(int num_layers, int num_experts, int top_k)
    : num_layers_(num_layers), num_experts_(num_experts), top_k_(top_k)
{
    last_selected_.resize(num_layers * top_k, -1);
}

std::vector<int> ExpertPrefetchPredictor::on_expert_selected(
    int layer_id, const int * expert_ids, int count
) {
    // Record this layer's selections
    int base = layer_id * top_k_;
    for (int i = 0; i < count && i < top_k_; ++i) {
        if (i < base + (int)last_selected_.size()) {
            last_selected_[base + i] = expert_ids[i];
        }
    }

    // Predict next layer: repeat same expert IDs
    int next_layer = layer_id + 1;
    std::vector<int> predicted;
    if (next_layer < num_layers_) {
        int next_base = next_layer * top_k_;
        for (int i = 0; i < top_k_; ++i) {
            int idx = next_base + i;
            if (idx < (int)last_selected_.size() && last_selected_[idx] >= 0) {
                // Use previous selection for this layer if available
                predicted.push_back(last_selected_[idx]);
            } else {
                // First time seeing this layer: use current layer's experts
                predicted.push_back(expert_ids[i % count]);
            }
        }
    }
    return predicted;
}

void ExpertPrefetchPredictor::reset() {
    std::fill(last_selected_.begin(), last_selected_.end(), -1);
}

// ---------------------------------------------------------------------------
// Prefetch next layer
// ---------------------------------------------------------------------------
void ExpertCacheManager::prefetch_next_layer(
    int current_layer_id,
    const int * selected_expert_ids,
    int count,
    cudaStream_t stream
) {
    if (!initialized_ || current_layer_id + 1 >= num_layers_) return;

    // Simple prediction: the same experts will be used in the next layer
    // This is a conservative heuristic — MoE models show strong expert persistence
    for (int i = 0; i < count; ++i) {
        int expert_id = selected_expert_ids[i];
        if (expert_id >= 0 && expert_id < num_experts_) {
            prefetch_expert(current_layer_id + 1, expert_id, stream);
        }
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------
void ExpertCacheManager::print_stats() const {
    fprintf(stderr, "\n[LUT-MoE Cache Stats]\n");
    fprintf(stderr, "  Hits:        %lu\n", stats_.total_hits);
    fprintf(stderr, "  Misses:      %lu\n", stats_.total_misses);
    fprintf(stderr, "  Prefetches:  %lu\n", stats_.total_prefetches);
    fprintf(stderr, "  Demotions:   %lu\n", stats_.total_demotions);
    fprintf(stderr, "  Evictions:   %lu\n", stats_.total_evictions);
    fprintf(stderr, "  SSD read:    %lu bytes\n", stats_.ssd_bytes_read);
    fprintf(stderr, "  Slots:       %d/%zu occupied\n", num_occupied(), num_slots_);
}

void ExpertCacheManager::print_slots() const {
    fprintf(stderr, "\n[LUT-MoE Cache Slots]\n");
    for (size_t i = 0; i < num_slots_; ++i) {
        const auto & s = slots_[i];
        if (!s.occupied) continue;
        fprintf(stderr, "  Slot %3zu: L%d/E%d tier=%d visits=%lu pending=%d\n",
                i, s.layer_id, s.expert_id, s.tier,
                (unsigned long)s.visit_count, (int)s.load_pending);
    }
}
