#!/bin/bash
# Environment setup for LUT-MoE vLLM integration
# Run: source setup_env.sh

# Add CUDA 13 runtime to library path (needed by vLLM .so files)
CUDA13_LIB="/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$CUDA13_LIB" ]; then
    if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA13_LIB:"* ]]; then
        export LD_LIBRARY_PATH="$CUDA13_LIB:$LD_LIBRARY_PATH"
    fi
fi

# Add project root to Python path
PROJECT_ROOT="/home/hh/LUT-MoE"
if [[ ":$PYTHONPATH:" != *":$PROJECT_ROOT:"* ]]; then
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

echo "[LUT-MoE vLLM] Environment configured"
echo "  LD_LIBRARY_PATH includes CUDA 13 runtime: $LD_LIBRARY_PATH"
echo "  PYTHONPATH includes project root: $PYTHONPATH"
