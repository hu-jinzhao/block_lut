#!/bin/bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME=/home/hh/miniconda3/envs/lut_moe_cu124
export CUDA_PATH=/home/hh/miniconda3/envs/lut_moe_cu124
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib
export PYTHONPATH=/home/hh/LUT-MoE:$PYTHONPATH

exec python3 -c "
import sys
sys.path.insert(0, '/home/hh/LUT-MoE')
import os
os.environ['VLLM_WSL2_ENABLE_PIN_MEMORY'] = '1'
os.environ['CUDA_HOME'] = '/home/hh/miniconda3/envs/lut_moe_cu124'
os.environ['LD_LIBRARY_PATH'] = '/home/hh/.local/lib/python3.12/site-packages/nvidia/cu13/lib'

if __name__ == '__main__':
    import time
    from vllm import LLM, SamplingParams
    print('Loading model...')
    t0 = time.time()
    llm = LLM(
        model='/home/hh/LUT-MoE/models/deepseek',
        trust_remote_code=True,
        dtype='bfloat16',
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    print(f'Loaded in {time.time()-t0:.1f}s')
    sp = SamplingParams(temperature=0.7, max_tokens=20)
    out = llm.generate(['The future of AI is'], sp)
    print(f'Output: {out[0].outputs[0].text}')
    print('SUCCESS!')
" > /tmp/vllm_run4.log 2>&1
echo "EXIT=$?" >> /tmp/vllm_run4.log
