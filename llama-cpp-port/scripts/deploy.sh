#!/bin/bash
# LUT-MoE → llama.cpp 一键部署脚本
# 使用方法:
#   ./deploy.sh --model /path/to/hf-model --out model.gguf [--nested-lut] [--gpu-memory 4096]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="/home/hh/llama.cpp"

echo "============================================"
echo "  LUT-MoE → llama.cpp 部署"
echo "============================================"

# ── 解析参数 ──
MODEL_PATH=""
OUTPUT_PATH=""
NESTED_LUT=""
GPU_MEMORY=4096
EXPERT_CACHE=2048
BUILD=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --out) OUTPUT_PATH="$2"; shift 2 ;;
        --nested-lut) NESTED_LUT="--nested-lut"; shift ;;
        --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
        --expert-cache) EXPERT_CACHE="$2"; shift 2 ;;
        --no-build) BUILD=false; shift ;;
        --help) echo "Usage: $0 --model PATH [--out PATH] [--nested-lut] [--gpu-memory MB] [--expert-cache MB]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model is required"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/\//_/g')
    OUTPUT_PATH="${MODEL_NAME}_blocklut.gguf"
fi

# ── Step 1: Python 转换 ──
echo ""
echo "[1/3] Converting HF model to BlockLUT GGUF ..."
python "$SCRIPT_DIR/convert_lut.py" \
    --model "$MODEL_PATH" \
    --out "$OUTPUT_PATH" \
    $NESTED_LUT

# ── Step 2: 构建 llama.cpp ──
if [ "$BUILD" = true ]; then
    echo ""
    echo "[2/3] Building llama.cpp with LUT-MoE support ..."
    cd "$LLAMA_CPP_DIR"
    mkdir -p build
    cd build
    cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
fi

# ── Step 3: 生成运行时偏移元数据 ──
echo ""
echo "[3/3] Generating runtime metadata ..."
python "$SCRIPT_DIR/generate_meta.py" \
    --gguf "$OUTPUT_PATH" \
    --output "${OUTPUT_PATH}.meta.json"

echo ""
echo "============================================"
echo "  部署完成!"
echo "  模型: $OUTPUT_PATH"
echo "  运行: $LLAMA_CPP_DIR/build/bin/main \\"
echo "          -m $OUTPUT_PATH \\"
echo "          --gpu-memory $GPU_MEMORY \\"
echo "          --expert-cache-size $EXPERT_CACHE \\"
echo "          -p \"Hello, world\""
echo "============================================"
