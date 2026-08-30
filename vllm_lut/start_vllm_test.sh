#!/bin/bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib
python3 /home/hh/LUT-MoE/vllm_lut/run_vllm_test.py > /tmp/vllm_final.log 2>&1
echo "EXIT_CODE=$?" >> /tmp/vllm_final.log
