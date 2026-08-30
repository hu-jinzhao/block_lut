#!/bin/bash
"""Fix CUDA version requirements in vLLM .so files"""
VLLM="/home/hh/.local/lib/python3.12/site-packages/vllm"
PT="$HOME/.local/bin/patchelf"

# Fix CUDA 12.8 runtime SONAME
$PT --set-soname libcudart.so.13 \
    /home/hh/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

# For each vLLM .so: remove all version info for cudart
for f in "$VLLM"/*.abi3.so; do
    echo "=== Fixing $(basename $f) ==="

    # Get all symbols needing libcudart.so.13
    SYMS=$(objdump -T "$f" | grep "libcudart\\.so\\.13" | awk '{print $NF}')
    for sym in $SYMS; do
        $PT --clear-symbol-version "$sym" "$f" 2>/dev/null
    done

    # Remove and re-add libcudart to clear .gnu.version_r
    $PT --remove-needed libcudart.so.13 "$f" 2>/dev/null
    $PT --remove-needed libcudart.so.12 "$f" 2>/dev/null
    $PT --add-needed libcudart.so.12 "$f" 2>/dev/null

    echo "  Done: $(basename $f)"
done

echo ""
echo "Testing import..."
LD_LIBRARY_PATH=/home/hh/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    python3 -c "import vllm; print('OK:', vllm.__version__)"
