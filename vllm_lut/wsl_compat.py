"""
WSL Compatibility Patch for vLLM.

vLLM V1 engine requires UVA (Unified Virtual Addressing) for pinned memory,
which is not available under WSL with CUDA 13 runtime on 12.8 drivers.

This patch provides workarounds to run vLLM under WSL.

Usage:
    import vllm_lut.wsl_compat
    vllm_lut.wsl_compat.apply_patch()
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def apply_patch():
    """
    Apply WSL compatibility patches to vLLM.

    Checks if running under WSL and applies necessary patches.
    On native Linux, no patches are needed.
    """
    import platform

    # Check if running on WSL
    is_wsl = "microsoft" in platform.uname().release.lower()

    if not is_wsl:
        logger.info("[LUT-MoE] Native Linux detected, no WSL patches needed")
        return True

    logger.warning("[LUT-MoE] WSL detected, applying compatibility patches...")

    # Set required environment variables
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

    # Try to provide libcudart.so.13 if missing
    _ensure_cuda_runtime()

    return True


def _ensure_cuda_runtime():
    """Ensure CUDA 13 runtime can be found by vLLM."""
    cuda13_lib = "/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
    if os.path.isdir(cuda13_lib):
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda13_lib not in current:
            os.environ["LD_LIBRARY_PATH"] = cuda13_lib + ":" + current
            logger.info(f"[LUT-MoE] Added {cuda13_lib} to LD_LIBRARY_PATH")


def check_vllm_compatible():
    """
    Check if vLLM installation is compatible with the current environment.

    Returns:
        (bool, str): (compatible, message)
    """
    import torch

    # Check CUDA
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    # Check vLLM version
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        return False, "vLLM is not installed"

    # Check if we can initialize vLLM
    try:
        from vllm.v1.worker.gpu.buffer_utils import is_uva_available
        uva = is_uva_available()
        if not uva:
            return False, (
                f"vLLM {vllm_version} requires UVA which is not available. "
                "Run on native Linux or update GPU driver."
            )
    except Exception:
        pass

    return True, f"vLLM {vllm_version} is compatible"
