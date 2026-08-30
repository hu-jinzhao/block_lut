#!/usr/bin/env python3
"""GPTQ 4bit - saves quantized weights from QuantLinear submodules."""
import logging, os, sys, time, json, shutil, signal
logging.disable(logging.CRITICAL)
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from safetensors.torch import save_file
import torch.nn as nn

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
OUTPUT_DIR = '/tmp/qwen_gptq_final'
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

# Save function using QuantLinear modules directly
SAVED = [False]

def save_quantized():
    """Extract qweight/qzeros/scales from QuantLinear submodules."""
    if SAVED[0]:
        return
    state = {}
    count = 0
    hf_model = model.model if hasattr(model, 'model') else model
    for name, mod in hf_model.named_modules():
        mod_type = type(mod).__name__
        if 'QuantLinear' in mod_type:
            for attr in ['qweight', 'qzeros', 'scales', 'g_idx']:
                if hasattr(mod, attr):
                    t = getattr(mod, attr)
                    if t is not None and t.device.type != 'meta':
                        state[f'{name}.{attr}'] = t.data.cpu()
                        count += 1
    if count > 0:
        save_file(state, os.path.join(OUTPUT_DIR, 'model_gptq.safetensors'))
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, 'model_gptq.safetensors'))
        print(f'\n[SAVE] {count} tensors, {sz/1e9:.1f}GB', flush=True)
        SAVED[0] = True

# Use ALARM signal to force-save after 6 hours (360 min)
signal.signal(signal.SIGALRM, lambda s, f: save_quantized())
signal.alarm(21600)  # 6 hours

calib_texts = texts[:32]
print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
t1 = time.time()

try:
    model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
    print(f'Quantize OK in {(time.time()-t1)/60:.1f}min', flush=True)
except Exception as e:
    print(f'Quantize: {type(e).__name__}: {e}', flush=True)

# Save if not already saved
save_quantized()

# Copy config
for f in os.listdir(MODEL_DIR):
    if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
        shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

print(f'Done in {(time.time()-t0)/60:.1f}min', flush=True)
print(f'Output: {OUTPUT_DIR}', flush=True)
print(f'Saved: {SAVED[0]}', flush=True)
