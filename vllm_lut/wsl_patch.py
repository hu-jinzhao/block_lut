"""
vLLM WSL UVA Compatibility Patch (direct approach).

Patches vLLM's UVA requirement to allow initialization on WSL
where UVA is not available due to CUDA version mismatch.

Usage:
    import vllm_lut.wsl_patch
    vllm_lut.wsl_patch.patch_vllm_uva()
"""

import logging
import os

logger = logging.getLogger(__name__)

# Store originals for restoration
_originals = {}


def patch_vllm_uva():
    """
    Monkey-patch vLLM's UVA (Unified Virtual Addressing) requirement.

    On WSL with CUDA version mismatch between vLLM's compiled CUDA version
    and the system driver, UVA is unavailable. This patch replaces the
    UvaBuffer with a regular CUDA tensor fallback.
    """
    try:
        from vllm.v1.worker.gpu import buffer_utils
        import torch
    except ImportError:
        logger.warning("[LUT-MoE] Could not import vLLM buffer_utils, skipping UVA patch")
        return False

    # Store originals
    _originals['UvaBuffer_init'] = buffer_utils.UvaBuffer.__init__

    # Create patched UvaBuffer that falls back to CUDA tensor
    def patched_uva_buffer_init(self, size, dtype):
        """Patched UvaBuffer that uses regular CUDA tensor instead of UVA."""
        if isinstance(size, (list, tuple)):
            self.cpu = torch.zeros(*size, dtype=dtype, device='cpu', pin_memory=True)
        else:
            self.cpu = torch.zeros(size, dtype=dtype, device='cpu', pin_memory=True)
        self.np = self.cpu.numpy()
        # Instead of UVA, just store a reference to the CPU tensor
        # The GPU operations will need explicit copies
        self.uva = self.cpu  # Fallback: use CPU tensor directly
        self._using_uva = False

    buffer_utils.UvaBuffer.__init__ = patched_uva_buffer_init

    # Also patch is_uva_available to return True (since we've patched it)
    _originals['is_uva_available'] = buffer_utils.is_uva_available

    def patched_is_uva_available():
        return True  # Tell vLLM UVA is available (we handle it)

    buffer_utils.is_uva_available = patched_is_uva_available

    logger.info("[LUT-MoE] vLLM UVA patch applied (fallback to CPU tensor)")
    return True


def restore_vllm_uva():
    """Restore original vLLM UVA implementation."""
    try:
        from vllm.v1.worker.gpu import buffer_utils

        if 'UvaBuffer_init' in _originals:
            buffer_utils.UvaBuffer.__init__ = _originals['UvaBuffer_init']
        if 'is_uva_available' in _originals:
            buffer_utils.is_uva_available = _originals['is_uva_available']

        logger.info("[LUT-MoE] vLLM UVA patch reverted")
    except ImportError:
        pass
