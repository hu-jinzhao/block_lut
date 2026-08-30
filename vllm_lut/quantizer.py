# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.

"""
LUT Clustering Quantizer for MoE Expert Weights.

Implements Block-wise Lookup-Table (BlockLUT) quantization and
NestedLUT progressive quantization for Mixture-of-Experts weights.

Key algorithms:
  - BlockLUT (8-bit): K-means on 128-element blocks, 256-entry codebook
  - NestedLUT (4/6/8-bit): Three-tier nested codebooks for progressive quality
    - Tier 0 (8-bit): full256 codebook
    - Tier 1 (6-bit): mapped64 codebook (64 unique values, padded to 256)
    - Tier 2 (4-bit): mapped16 codebook (16 unique values, padded to 256)
"""

import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class LUTQuantizer:
    """
    Lookup-Table Quantizer for MoE expert weights.

    Supports BlockLUT (8-bit block-wise) and NestedLUT (4/6/8-bit progressive).

    The quantized format stores:
      - indices: uint8 array of LUT indices (one per element)
      - absmax: bf16 array of block-level normalization factors (one per 128 elements)
      - lut_codebook: bf16 array of 256 centroid values
    """

    def __init__(
        self,
        code_type: str = "BLOCKLUT",
        block_size: int = 128,
        lut_size: int = 256,
        device: str = "cuda",
    ):
        assert code_type in ("LUT", "BLOCKLUT", "NESTEDLUT"), \
            f"Unsupported code_type: {code_type}"
        self.code_type = code_type
        self.block_size = block_size
        self.lut_size = lut_size
        self.device = device

        # Trained codebooks (populated by train() or load())
        self.lut_codebook: Optional[torch.Tensor] = None  # [256] bf16
        self.nested_mapped64: Optional[torch.Tensor] = None  # [256] bf16 (64 unique + repeats)
        self.nested_mapped16: Optional[torch.Tensor] = None  # [256] bf16 (16 unique + repeats)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @torch.no_grad()
    def train(
        self,
        expert_weights: torch.Tensor,
        n_iter: int = 50,
    ) -> None:
        """
        Train LUT codebook(s) on expert weight data.

        Args:
            expert_weights: Tensor of shape [num_experts, rows, cols] or flattened [N,]
            n_iter: Number of K-means iterations.
        """
        # Flatten all expert weights
        if expert_weights.dim() == 3:
            n, r, c = expert_weights.shape
            flat_weights = expert_weights.view(n * r * c)
        elif expert_weights.dim() == 2:
            flat_weights = expert_weights.view(-1)
        else:
            flat_weights = expert_weights

        if self.code_type == "LUT":
            # Basic LUT: global normalization + k-means on raw values
            self._train_basic_lut(flat_weights, n_iter)
        elif self.code_type in ("BLOCKLUT", "NESTEDLUT"):
            # BlockLUT: block-wise normalization + k-means on normalized values
            # Reshape into blocks of block_size
            total_elements = flat_weights.shape[0]
            num_blocks = math.ceil(total_elements / self.block_size)
            padded_size = num_blocks * self.block_size
            if padded_size != total_elements:
                padded = torch.zeros(padded_size, dtype=flat_weights.dtype,
                                     device=flat_weights.device)
                padded[:total_elements] = flat_weights
                flat_weights = padded

            # Block normalization
            blocks = flat_weights.view(-1, self.block_size)  # [num_blocks, 128]
            block_absmax, _ = blocks.abs().max(dim=1, keepdim=True)  # [num_blocks, 1]
            block_absmax = block_absmax.clamp(min=1e-10)
            normalized = blocks / block_absmax  # [num_blocks, 128]

            # Train on normalized values
            train_data = normalized.reshape(-1)
            self._train_kmeans(train_data, self.lut_size, n_iter)

            # For NESTEDLUT, train nested codebooks
            if self.code_type == "NESTEDLUT":
                self._train_nested_codebooks(train_data, n_iter)

    def _train_basic_lut(self, data: torch.Tensor, n_iter: int) -> None:
        """Train a simple LUT (no block normalization)."""
        self._train_kmeans(data, self.lut_size, n_iter)

    def _train_kmeans(
        self, data: torch.Tensor, k: int, n_iter: int,
        max_samples: int = 200000,
    ) -> torch.Tensor:
        """
        K-means clustering using mini-batch approach for memory efficiency.

        For large datasets (>max_samples), uses a random subset for training
        to avoid OOM from the [N, k] distance matrix.

        Args:
            data: 1D tensor of training data
            k: number of clusters
            n_iter: number of iterations
            max_samples: maximum samples for distance computation

        Returns:
            sorted centroids as bf16 tensor
        """
        data = data.to(self.device).float()

        # Use subset for large datasets to avoid OOM in distance matrix
        n_total = data.shape[0]
        if n_total > max_samples:
            subset_idx = torch.randperm(n_total, device=self.device)[:max_samples]
            train_data = data[subset_idx]
        else:
            train_data = data

        # Sample initial centroids randomly
        n_init = min(k, train_data.shape[0])
        idx = torch.randperm(train_data.shape[0], device=self.device)[:n_init]
        centroids = train_data[idx].clone()

        if train_data.shape[0] < k:
            pad = torch.zeros(k - n_init, device=self.device)
            centroids = torch.cat([centroids, pad])

        for _ in range(n_iter):
            # Distance: chunk train_data to avoid OOM with large k
            chunk_size = min(65536, train_data.shape[0])
            all_assignments = []
            for start in range(0, train_data.shape[0], chunk_size):
                end = min(start + chunk_size, train_data.shape[0])
                chunk = train_data[start:end]
                dists = torch.abs(chunk.unsqueeze(1) - centroids.unsqueeze(0))
                all_assignments.append(dists.argmin(dim=1))
            assignments = torch.cat(all_assignments, dim=0)

            # Update centroids (vectorized mean)
            for i in range(k):
                mask = assignments == i
                if mask.any():
                    centroids[i] = train_data[mask].mean()

        # Sort by value for determinism
        centroids, _ = torch.sort(centroids)
        self.lut_codebook = centroids.to(torch.bfloat16).cpu()
        return self.lut_codebook

    def _train_nested_codebooks(
        self, data: torch.Tensor, n_iter: int
    ) -> None:
        """
        Train nested codebooks for progressive LUT:
          - mapped16: 16 unique centroids (4-bit)
          - mapped64: 64 unique centroids (6-bit)
          - full256: 256 unique centroids (8-bit) — already trained
        """
        if self.lut_codebook is None:
            raise RuntimeError("Must train full256 codebook first.")

        # Train K=64 on block-normalized data
        centroids_64 = self._train_kmeans(data, 64, n_iter)

        # Derive K=16 by greedy merging from K=64 centroids
        centroids_16 = self._greedy_merge(centroids_64, 16)

        # Build mapped tables (256-entry with repetitions)
        self.nested_mapped64 = self._build_mapped_table(centroids_64, 256)
        self.nested_mapped16 = self._build_mapped_table(centroids_16, 256)

    def _greedy_merge(
        self, centroids: torch.Tensor, target_k: int
    ) -> torch.Tensor:
        """
        Greedily merge closest centroid pairs until target_k remain.
        Used to derive K=16 codebook from K=64 codebook.
        """
        c = centroids.clone()
        while c.shape[0] > target_k:
            # Find closest pair
            diffs = torch.abs(c.unsqueeze(0) - c.unsqueeze(1))
            diffs = diffs + torch.eye(c.shape[0], device=c.device) * float('inf')
            i, j = divmod(diffs.argmin().item(), c.shape[0])
            # Merge: average
            merged = (c[i] + c[j]) / 2
            # Remove j, replace i with merged
            c[i] = merged
            c = torch.cat([c[:j], c[j+1:]])
        c, _ = torch.sort(c)
        return c

    def _build_mapped_table(
        self, centroids: torch.Tensor, table_size: int
    ) -> torch.Tensor:
        """
        Build a table_size-entry table from centroids by repeating values.
        Used to create 256-entry tables from 64 or 16 unique centroids.
        """
        assert table_size >= centroids.shape[0]
        repeats = table_size // centroids.shape[0]
        remainder = table_size % centroids.shape[0]
        table = centroids.repeat(repeats)
        if remainder > 0:
            table = torch.cat([table, centroids[:remainder]])
        return table.to(torch.bfloat16).cpu()

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize(
        self,
        weight: torch.Tensor,
        codebook: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize a weight tensor to BlockLUT format.

        Args:
            weight: bf16 tensor of shape [rows, cols] or [N,]
            codebook: bf16 LUT codebook [256]. Uses self.lut_codebook if None.

        Returns:
            dict with keys:
              - 'indices': uint8 tensor of LUT indices
              - 'absmax': bf16 tensor of block scaling factors
              - 'codebook': bf16 LUT codebook
        """
        if codebook is None:
            codebook = self.lut_codebook
        if codebook is None:
            raise RuntimeError("LUT codebook not trained or loaded. "
                               "Call train() or load() first.")

        codebook = codebook.to(weight.device)
        orig_shape = weight.shape
        flat = weight.reshape(-1)
        total_elements = flat.shape[0]

        if self.code_type in ("BLOCKLUT", "NESTEDLUT"):
            # Block-wise normalization
            num_blocks = math.ceil(total_elements / self.block_size)
            padded_size = num_blocks * self.block_size
            if padded_size != total_elements:
                padded = torch.zeros(padded_size, dtype=flat.dtype, device=flat.device)
                padded[:total_elements] = flat
                flat = padded

            blocks = flat.view(-1, self.block_size)
            absmax, _ = blocks.abs().max(dim=1, keepdim=True)
            absmax = absmax.clamp(min=1e-10)
            normalized = blocks / absmax
            train_data = normalized.reshape(-1)
        else:
            # Basic LUT: normalize globally
            absmax_val = flat.abs().max().clamp(min=1e-10)
            train_data = flat / absmax_val
            absmax = torch.tensor([absmax_val], device=flat.device)

        # Assign to nearest centroid
        train_data_f = train_data.float()
        codebook_f = codebook.float()
        dists = torch.abs(train_data_f.unsqueeze(1) - codebook_f.unsqueeze(0))
        indices = dists.argmin(dim=1).to(torch.uint8)

        # Truncate to original size if padded
        if total_elements < indices.shape[0]:
            indices = indices[:total_elements]

        return {
            "indices": indices.cpu(),
            "absmax": absmax.reshape(-1).cpu().to(torch.bfloat16),
            "codebook": codebook.cpu(),
        }

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decompress(
        self,
        indices: torch.Tensor,
        absmax: torch.Tensor,
        codebook: torch.Tensor,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Decompress LUT-quantized weights back to bf16.

        Args:
            indices: uint8 LUT indices [N]
            absmax: block scaling factors [num_blocks]
            codebook: bf16 LUT [256]
            out_shape: optional reshape target

        Returns:
            bf16 tensor of decompressed weights
        """
        device = indices.device
        codebook = codebook.to(device)

        # LUT lookup: index -> centroid value
        normalized = codebook[indices.long()]  # [N] bf16

        if self.code_type in ("BLOCKLUT", "NESTEDLUT") and absmax.shape[0] > 1:
            # Expand block absmax to element level
            block_id = torch.arange(
                indices.shape[0], device=device
            ) // self.block_size
            if block_id.max() >= absmax.shape[0]:
                block_id = block_id.clamp(max=absmax.shape[0] - 1)
            scale = absmax[block_id]
        else:
            scale = absmax[0]

        weights = normalized * scale

        if out_shape is not None:
            weights = weights.reshape(out_shape)
        return weights

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save LUT codebooks to .npy files."""
        os.makedirs(path, exist_ok=True)
        if self.lut_codebook is not None:
            np.save(os.path.join(path, "blocklut_256.npy"),
                    self.lut_codebook.numpy())
        if self.nested_mapped64 is not None:
            np.save(os.path.join(path, "nested_lut_mapped64.npy"),
                    self.nested_mapped64.numpy())
        if self.nested_mapped16 is not None:
            np.save(os.path.join(path, "nested_lut_mapped16.npy"),
                    self.nested_mapped16.numpy())

    def load(self, path: str) -> bool:
        """Load LUT codebooks from .npy files. Returns True if successful."""
        lut_file = os.path.join(path, "blocklut_256.npy")
        if os.path.exists(lut_file):
            self.lut_codebook = torch.from_numpy(
                np.load(lut_file)
            ).to(torch.bfloat16)
        elif os.path.exists(os.path.join(path, "full256.npy")):
            self.lut_codebook = torch.from_numpy(
                np.load(os.path.join(path, "full256.npy"))
            ).to(torch.bfloat16)
        else:
            return False

        # Optional: nested codebooks
        for attr, fname in [
            ("nested_mapped64", "nested_lut_mapped64.npy"),
            ("nested_mapped16", "nested_lut_mapped16.npy"),
        ]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                setattr(self, attr,
                        torch.from_numpy(np.load(fp)).to(torch.bfloat16))

        return True


# ------------------------------------------------------------------
# Standalone functions (convenience wrappers)
# ------------------------------------------------------------------

def train_lut_codebook(
    weights: torch.Tensor,
    code_type: str = "BLOCKLUT",
    block_size: int = 128,
    lut_size: int = 256,
    n_iter: int = 50,
) -> LUTQuantizer:
    """Train a LUT codebook on expert weights. Returns trained quantizer."""
    quantizer = LUTQuantizer(code_type=code_type, block_size=block_size)
    quantizer.train(weights, n_iter=n_iter)
    return quantizer


def quantize_expert_weights(
    weights: torch.Tensor,
    quantizer: LUTQuantizer,
    code_type: str = "BLOCKLUT",
) -> Dict[str, torch.Tensor]:
    """
    Quantize expert weights using a pre-trained LUT quantizer.

    Args:
        weights: bf16 tensor [num_experts, rows, cols]
        quantizer: trained LUTQuantizer
        code_type: quantization type

    Returns:
        dict with 'indices' [num_experts, N], 'absmax' [num_experts, num_blocks],
        and 'codebook' [256]
    """
    n_experts = weights.shape[0]
    all_indices = []
    all_absmax = []

    for i in range(n_experts):
        result = quantizer.quantize(weights[i])
        all_indices.append(result["indices"])
        all_absmax.append(result["absmax"])

    return {
        "indices": torch.stack(all_indices, dim=0),
        "absmax": torch.stack(all_absmax, dim=0),
        "codebook": quantizer.lut_codebook,
    }


def decompress_blocklut(
    indices: torch.Tensor,
    absmax: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fast BlockLUT decompression (pure PyTorch).

    Args:
        indices: [N] uint8 LUT indices
        absmax: [num_blocks] bf16 scaling factors
        codebook: [256] bf16 codebook
        block_size: elements per block

    Returns:
        [N] bf16 decompressed weights
    """
    device = indices.device
    codebook = codebook.to(device)
    absmax = absmax.to(device)

    normalized = codebook[indices.long()]
    block_id = torch.arange(indices.shape[0], device=device) // block_size
    if block_id.max() >= absmax.shape[0]:
        block_id = block_id.clamp(max=absmax.shape[0] - 1)
    scale = absmax[block_id]

    return normalized * scale
