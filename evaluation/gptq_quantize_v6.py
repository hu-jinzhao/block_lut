#!/usr/bin/env python3
"""GPTQ 4bit quantization v6 - saves from QuantLinear modules directly."""
import logging, os, sys, time, json, shutil
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

# Skip pack step entirely
model.pack_model = lambda *a, **kw: None
# Also disable the underlying pack function
if hasattr(model, '_pack_model'):
    model._pack_model = lambda *a, **kw: None

SAVED = [False]

def save_quantized_weights(prefix=''):
    """Save qweight/qzeros/scales from ALL QuantLinear modules."""
    if SAVED[0]:
        return 0
    state = {}
    count = 0
    hf_model = model.model if hasattr(model, 'model') else model

    # Find all QuantLinear modules
    for name, mod in hf_model.named_modules():
        cls_name = type(mod).__name__
        if 'QuantLinear' in cls_name:
            for attr in ['qweight', 'qzeros', 'scales', 'g_idx']:
                if hasattr(mod, attr):
                    t = getattr(mod, attr)
                    if t is not None and not t.is_meta:
                        state[f'{name}.{attr}'] = t.data.cpu()
                        count += 1

    if count == 0:
        # Try the gptqmodel wrapper's own modules
        for name, mod in model.named_modules():
            if 'QuantLinear' in type(mod).__name__:
                for attr in ['qweight', 'qzeros', 'scales', 'g_idx']:
                    if hasattr(mod, attr):
                        t = getattr(mod, attr)
                        if t is not None and not t.is_meta:
                            state[f'{name}.{attr}'] = t.data.cpu()
                            count += 1

    if count > 0:
        out = os.path.join(OUTPUT_DIR, f'model_{prefix}.safetensors')
        save_file(state, out)
        sz = os.path.getsize(out)
        print(f'[SAVE] {count} tensors, {sz/1e9:.1f}GB -> {out}', flush=True)
        SAVED[0] = True
    else:
        print(f'[SAVE] No QuantLinear modules found at {prefix}', flush=True)

    return count

calib_texts = texts[:32]
print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
t1 = time.time()

# Set an alarm to save after 5 hours
import signal
signal.signal(signal.SIGALRM, lambda s, f: save_quantized_weights('alarm'))
signal.alarm(18000)  # 5 hours

try:
    model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
    print(f'Quantize OK in {(time.time()-t1)/60:.1f}min', flush=True)
except Exception as e:
    error_msg = f'{type(e).__name__}: {e}'
    print(f'Quantize ended: {error_msg}', flush=True)

# Save quantized weights
n = save_quantized_weights('final')
print(f'Final: {n} quantized tensors', flush=True)

if n > 0:
    # Copy config
    for f in os.listdir(MODEL_DIR):
        if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
            shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))
    print(f'Output: {OUTPUT_DIR}', flush=True)

print(f'Done in {(time.time()-t0)/60:.1f}min', flush=True)
