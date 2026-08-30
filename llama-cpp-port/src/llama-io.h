#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <string>
#include <functional>
#include <thread>
#include <condition_variable>
#include <queue>
#include <cuda_runtime.h>

// Per-expert file offset metadata (for SSD pread)
struct ExpertFileMeta {
    size_t offset_4bit;   // file offset to 4-bit section (low nibbles)
    size_t offset_6bit;   // file offset to 6-bit (low+mid)
    size_t offset_8bit;   // file offset to full 8-bit (all 3 sections)
    size_t compressed_size; // total compressed bytes in file
    int    n_elements;      // number of bf16 elements
};

// Cache slot — tracks one expert in GPU memory
struct ExpertCacheSlot {
    int      layer_id;
    int      expert_id;
    int      tier;              // 0=hot(bf16 decompressed), 1=warm(6bit comp), 2=cold(4bit comp)
    uint64_t visit_count;
    uint64_t last_access_time;

    // GPU buffers
    uint16_t * gpu_weights;     // decompressed bf16 (valid in tier 0)
    uint8_t  * gpu_compressed;  // compressed bitplane data (valid in tier 0/1/2)

    // Async tracking
    cudaEvent_t load_event;     // fires when GPU decompress finishes
    bool        load_pending;   // true while async load is in flight

    bool occupied;

    ExpertCacheSlot()
        : layer_id(-1), expert_id(-1), tier(2), visit_count(0),
          last_access_time(0), gpu_weights(nullptr), gpu_compressed(nullptr),
          load_event(nullptr), load_pending(false), occupied(false) {}
};

// Cache statistics
struct CacheStats {
    uint64_t total_hits;
    uint64_t total_misses;
    uint64_t total_prefetches;
    uint64_t total_evictions;
    uint64_t total_demotions;
    uint64_t ssd_bytes_read;
    uint64_t gpu_decompress_us;

    CacheStats() : total_hits(0), total_misses(0), total_prefetches(0),
                   total_evictions(0), total_demotions(0),
                   ssd_bytes_read(0), gpu_decompress_us(0) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// ExpertCacheManager
//
// Manages a GPU memory pool for MoE expert weights. Experts are stored on SSD
// in BlockLUT compressed format; loaded on-demand with GPU-side decompression.
//
// Tier system (matches LUT-MoE NestedLUT):
//   Tier 0 (hot)   : bf16 decompressed on GPU   ← best quality, 100% I/O
//   Tier 1 (warm)  : 6-bit compressed on GPU     ← 75% I/O, re-decompress on use
//   Tier 2 (cold)  : 4-bit compressed on GPU     ← 50% I/O
//   Evicted         : SSD only
//
// Eviction: LFU + tier demotion (0→1→2→free) before full eviction.
// ─────────────────────────────────────────────────────────────────────────────
class ExpertCacheManager {
public:
    static ExpertCacheManager & instance();

    bool init(
        int num_layers,
        int num_experts,
        size_t gpu_cache_size,              // total GPU bytes for decompressed pool
        const std::string & gguf_file_path,
        std::vector<std::vector<ExpertFileMeta>> && expert_meta,
        const uint16_t * lut_host,           // 256-entry full LUT (bf16 uint16)
        const uint16_t * lut_mapped64,       // 256-entry mapped64 LUT
        const uint16_t * lut_mapped16,       // 256-entry mapped16 LUT
        size_t total_elements                // bf16 elements per expert weight
    );

    // ── Main API ──

    // Get a pointer to decompressed bf16 weights for this expert.
    // If not cached, loads from SSD → GPU decompress → returns pointer.
    // Blocks until decompression is done (syncs on internal CUDA event).
    uint16_t * get_expert_weights(
        int layer_id,
        int expert_id,
        cudaStream_t stream
    );

    // Async prefetch: start loading without blocking.
    // The expert will be available on a later get_expert_weights call.
    void prefetch_expert(
        int layer_id,
        int expert_id,
        cudaStream_t stream
    );

    // Batch prefetch: load multiple experts at once.
    void batch_prefetch(
        int layer_id,
        const std::vector<int> & expert_ids,
        cudaStream_t stream
    );

    // Prefetch next layer: after router selects experts at layer N,
    // predict and start loading experts for layer N+1.
    void prefetch_next_layer(
        int current_layer_id,
        const int * selected_expert_ids,
        int count,
        cudaStream_t stream
    );

    // ── Cache control ──

    // Set LUT tables on GPU (update for tier changes)
    void set_lut_gpu(
        const uint16_t * lut,
        const uint16_t * lut_mapped64,
        const uint16_t * lut_mapped16
    );

    // Clear all slots (e.g., at sequence boundary)
    void clear_all();

    // ── Accessors ──

    bool is_initialized() const { return initialized_; }
    const CacheStats & stats() const { return stats_; }
    void reset_stats() { stats_ = CacheStats(); }
    int  num_slots() const { return (int)num_slots_; }
    int  num_occupied() const;

    // ── Debug ──

    void print_stats() const;
    void print_slots() const;

private:
    ExpertCacheManager();
    ~ExpertCacheManager();

    ExpertCacheManager(const ExpertCacheManager &) = delete;
    ExpertCacheManager & operator=(const ExpertCacheManager &) = delete;

    // ── Internal helpers ──

    // Find a free slot, or evict+demote to make room
    size_t find_or_evict_slot();

    // Evict: demote tier or fully clear. Returns true if fully evicted.
    bool evict_one_slot(size_t slot_id);

    // Demote a slot: tier 0→1 (drop bf16, keep 6-bit comp), 1→2 (drop to 4-bit), 2→free
    void demote_slot(size_t slot_id);

    // Upgrade a slot to a higher tier (load delta bits from SSD)
    bool upgrade_slot(size_t slot_id, int target_tier, cudaStream_t stream);

    // Read compressed data from SSD
    bool read_from_ssd(
        int layer_id, int expert_id,
        size_t offset, size_t size,
        uint8_t * dst
    );

    // GPU-side BlockLUT decompression (launches CUDA kernel)
    // compressed input → bf16 output, all on-GPU
    void gpu_decompress(
        uint16_t * dst_bf16,
        const uint8_t * gpu_compressed,
        const uint16_t * gpu_lut,
        const uint16_t * gpu_absmax,
        int n_elements,
        int tier,
        cudaStream_t stream
    );

    // Compute compressed size for a given tier
    size_t compressed_size_for_tier(int n_elements, int tier) const;

    // Select LUT device pointer for a given tier
    const uint16_t * lut_for_tier(int tier) const;

    // Find slot by expert key
    int find_slot(int layer_id, int expert_id) const;

    // ── State ──

    bool initialized_;
    int  num_layers_;
    int  num_experts_;
    size_t total_elements_;

    // Cache
    std::vector<ExpertCacheSlot> slots_;
    size_t num_slots_;
    size_t slot_bytes_;          // bytes for decompressed bf16 weights
    size_t comp_slot_bytes_;     // bytes for compressed data per slot

    // SSD
    int  gguf_fd_;
    std::string gguf_path_;
    std::vector<std::vector<ExpertFileMeta>> expert_meta_;

    // GPU LUT tables (device pointers)
    uint16_t * lut_full_;        // 256 entries, full precision bf16
    uint16_t * lut_mapped64_;
    uint16_t * lut_mapped16_;

    // GPU memory pools
    uint16_t * gpu_pool_base_;       // decompressed bf16 pool
    uint8_t  * gpu_comp_pool_base_;  // compressed data pool

    // Host pinned memory for async SSD→GPU transfer
    uint8_t * pinned_buf_;
    size_t    pinned_buf_size_;

    // Stats
    CacheStats stats_;
    uint64_t   access_counter_;

    // Thread safety
    mutable std::mutex mutex_;
};

// ─────────────────────────────────────────────────────────────────────────────
// ExpertPrefetchPredictor
//
// Predicts next-layer experts based on current-layer router selections.
// Strategy: "repeat" — experts used in layer N are prefetched for layer N+1.
// ─────────────────────────────────────────────────────────────────────────────
class ExpertPrefetchPredictor {
public:
    ExpertPrefetchPredictor(int num_layers, int num_experts, int top_k);

    // Called after router selects experts at a given layer.
    // Returns list of expert IDs to prefetch for next layer.
    std::vector<int> on_expert_selected(int layer_id, const int * expert_ids, int count);

    // Reset at sequence boundary
    void reset();

private:
    int num_layers_;
    int num_experts_;
    int top_k_;
    std::vector<int> last_selected_;  // [layer * top_k + slot]
};

extern ExpertCacheManager g_expert_cache;
