# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
Progressive Expert Cache for vLLM.

Implements the dynamic tier promotion/demotion system for expert weights:
  - Tier 0 (HOT, 8-bit): frequently accessed, decompressed to bf16 and cached in GPU
  - Tier 1 (WARM, 6-bit): moderately accessed, decompressed to bf16 on demand
  - Tier 2 (COLD, 4-bit): rarely accessed, decompressed on demand

The cache manages GPU memory for decompressed bf16 expert weights,
with automatic eviction and tier transitions based on access frequency.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch


class ExpertCacheEntry:
    """A single entry in the expert cache."""

    __slots__ = ("layer_id", "expert_id", "weights", "tier",
                 "visit_count", "last_access_step")

    def __init__(
        self,
        layer_id: int,
        expert_id: int,
        tier: int = 2,
    ):
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.weights: Optional[Dict[str, torch.Tensor]] = None  # bf16 weights
        self.tier = tier  # 0=hot, 1=warm, 2=cold
        self.visit_count = 0
        self.last_access_step = 0

    def key(self) -> Tuple[int, int]:
        return (self.layer_id, self.expert_id)

    def __repr__(self) -> str:
        return (f"ExpertCacheEntry(layer={self.layer_id}, expert={self.expert_id}, "
                f"tier={self.tier}, visits={self.visit_count})")


class ProgressiveExpertCache:
    """
    GPU-side cache for decompressed MoE expert weights with progressive tier management.

    Manages which experts are kept in decompressed bf16 form in GPU memory,
    and which need on-the-fly decompression from LUT format.

    Tier thresholds:
      - Tier 0 (HOT): visit_count >= 50, decompressed bf16 kept in GPU
      - Tier 1 (WARM): visit_count >= 10, decompressed on first access, cached
      - Tier 2 (COLD): visit_count < 10, decompressed on each access
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        max_gpu_entries: int = 256,
        hot_threshold: int = 50,
        warm_threshold: int = 10,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.max_gpu_entries = max_gpu_entries
        self.device = device
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold

        # All expert entries (tracked regardless of cache state)
        self.entries: Dict[Tuple[int, int], ExpertCacheEntry] = {}

        # Decompressed cache: key -> dict of weight tensors
        self._cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        # Access statistics
        self._global_step = 0
        self._layer_frequency: List[Counter] = [
            Counter() for _ in range(num_layers)
        ]

    def _get_or_create_entry(
        self, layer_id: int, expert_id: int
    ) -> ExpertCacheEntry:
        key = (layer_id, expert_id)
        if key not in self.entries:
            self.entries[key] = ExpertCacheEntry(layer_id, expert_id)
        return self.entries[key]

    def record_access(self, layer_id: int, expert_id: int) -> int:
        """
        Record an expert access and return the current LUT tier.

        Args:
            layer_id: layer index
            expert_id: expert index

        Returns:
            tier: 0=hot(8-bit), 1=warm(6-bit), 2=cold(4-bit)
        """
        entry = self._get_or_create_entry(layer_id, expert_id)
        entry.visit_count += 1
        entry.last_access_step = self._global_step
        self._layer_frequency[layer_id][expert_id] += 1
        self._global_step += 1

        # Promote if threshold reached
        if entry.visit_count >= self.hot_threshold:
            entry.tier = 0
        elif entry.visit_count >= self.warm_threshold:
            entry.tier = 1
        else:
            entry.tier = 2

        return entry.tier

    def get_tier(self, layer_id: int, expert_id: int) -> int:
        """Get the current tier for an expert."""
        entry = self._get_or_create_entry(layer_id, expert_id)
        return entry.tier

    def cache_weights(
        self,
        layer_id: int,
        expert_id: int,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        """
        Store decompressed bf16 weights in the GPU cache.

        May evict older entries if cache is full.
        """
        key = (layer_id, expert_id)
        entry = self._get_or_create_entry(layer_id, expert_id)

        # Check if we need to evict
        if key not in self._cache and len(self._cache) >= self.max_gpu_entries:
            self._evict_one()

        self._cache[key] = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }
        entry.weights = self._cache[key]

    def get_cached_weights(
        self, layer_id: int, expert_id: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached bf16 weights, or None if not cached."""
        key = (layer_id, expert_id)
        return self._cache.get(key)

    def is_cached(self, layer_id: int, expert_id: int) -> bool:
        """Check if expert's bf16 weights are in GPU cache."""
        return (layer_id, expert_id) in self._cache

    def evict(self, layer_id: int, expert_id: int) -> bool:
        """Evict an expert from cache. Returns True if was cached."""
        key = (layer_id, expert_id)
        if key in self._cache:
            entry = self._get_or_create_entry(layer_id, expert_id)
            entry.weights = None
            # Demote on eviction
            if entry.tier == 0:
                entry.tier = 1
            elif entry.tier == 1:
                entry.tier = 2
            del self._cache[key]
            return True
        return False

    def _evict_one(self) -> None:
        """Evict the least recently used entry from cache."""
        if not self._cache:
            return

        # Find entry with lowest last_access_step (LRU)
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self.entries[k].last_access_step
        )
        self.evict(*lru_key)

    def clear(self) -> None:
        """Clear all cached weights."""
        for entry in self.entries.values():
            entry.weights = None
        self._cache.clear()

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        tier_counts = Counter()
        for entry in self.entries.values():
            tier_counts[entry.tier] += 1
        return {
            "total_entries": len(self.entries),
            "cached_entries": len(self._cache),
            "max_cache_size": self.max_gpu_entries,
            "tier_counts": {
                "hot": tier_counts[0],
                "warm": tier_counts[1],
                "cold": tier_counts[2],
            },
            "global_steps": self._global_step,
        }

    def predict_next_layer(
        self,
        current_layer: int,
        topk: int = 5,
    ) -> Tuple[int, List[int]]:
        """
        Predict which experts will be needed in the next layer.
        Uses frequency + recent history heuristics.

        Returns:
            (next_layer_id, list_of_predicted_expert_ids)
        """
        next_layer = (current_layer + 1) % self.num_layers
        freq = self._layer_frequency[next_layer]

        if not freq:
            return next_layer, []

        # Return most frequent experts
        top_experts = [
            eid for eid, _ in freq.most_common(topk)
        ]
        return next_layer, top_experts[:topk]
