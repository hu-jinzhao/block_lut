#!/usr/bin/env python3
"""
GPTQ 4bit quantization with manual save.
Uses gptqmodel but intercepts weights before the buggy pack step.
Saves weights layer by layer to avoid losing results on crash.
"""
import logging, os, sys, time, json, signal, threading
logging.disable(logging.CRITICAL)
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from safetensors.torch import save_file
import torch

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
OUTPUT_DIR = '/tmp/qwen_gptq_final'
SAVED_FLAG = '/tmp/gptq_done.flag'

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open('/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json') as f:
    texts = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

quantize_config = QuantizeConfig(
    bits=4, group_size=128, desc_act=True, sym=True,
    damp_percent=0.01,
)

print('Loading...', flush=True)
t0 = time.time()
model = GPTQModel.load(MODEL_DIR, quantize_config=quantize_config,
                        device='cpu', trust_remote_code=True)
print(f'Loaded in {time.time()-t0:.1f}s', flush=True)

calib_texts = texts[:32]

# Monkey-patch: intercept pack_model to save weights before it runs
original_pack = model.pack_model if hasattr(model, 'pack_model') else None

def save_current_weights(signal_num=None, frame=None):
    """Save all quantized weights found in the model."""
    print(f'\n[SAVE] Saving quantized weights...', flush=True)
    state = {}
    quant_count = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            state[name] = param.data.cpu()
            quant_count += 1
    for name, param in model.named_parameters():
        if not any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            if name not in state:
                state[name] = param.data.cpu()

    # Also copy config
    import shutil
    for f in os.listdir(MODEL_DIR):
        if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
            shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

    save_file(state, os.path.join(OUTPUT_DIR, 'model.safetensors'))
    disk_size = os.path.getsize(os.path.join(OUTPUT_DIR, 'model.safetensors'))
    print(f'[SAVE] Saved {quant_count} quantized tensors ({disk_size/1e9:.1f}GB)', flush=True)
    open(SAVED_FLAG, 'w').write(f'Saved at {time.time()}\n')
    if signal_num is not None:
        sys.exit(0)

# Register signal handler for SIGUSR1
signal.signal(signal.SIGUSR1, save_current_weights)

print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
print(f'To save weights during quantization, run: kill -USR1 {os.getpid()}', flush=True)
t1 = time.time()

# Start a timer to auto-save after 4 hours (in case of hang)
timer = threading.Timer(14400, save_current_weights)  # 4 hours
timer.daemon = True
timer.start()

try:
    model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
    print(f'Quantize completed in {(time.time()-t1)/60:.1f}min', flush=True)
    # If we get here, quantize succeeded. Save weights.
    save_current_weights()
except Exception as e:
    print(f'Quantize error (maybe pack step): {e}', flush=True)
    # Try to save whatever we have
    save_current_weights()

timer.cancel()
print(f'Done in {(time.time()-t0)/60:.1f}min', flush=True)
