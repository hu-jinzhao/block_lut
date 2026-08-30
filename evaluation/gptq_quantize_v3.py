#!/usr/bin/env python3
"""
GPTQ 4bit quantization. Monky-patches pack_model to save after each layer.
"""
import logging, os, sys, time, json, shutil, gc
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

# Monkey-patch pack_model to save weights instead of compiling extensions
original_pack = getattr(model, 'pack_model', None)
if original_pack:
    def safe_pack(*args, **kwargs):
        print(f'\n[SAVE] pack_model called - saving weights before pack...', flush=True)
        save_weights(model, f'pre_pack')
        print(f'[SAVE] Proceeding with pack (may hang)...', flush=True)
        return original_pack(*args, **kwargs)
    model.pack_model = safe_pack

# Counters for tracking
layer_counter = [0]

# Monkey-patch the quantize_one_layer or quantize_module methods
# to save weights after each layer
original_quantize_module = getattr(model, 'quantize_module', None)

ALL_SAVED = [False]

def save_weights(m, suffix=''):
    """Save all qweight/qzero/scales tensors from model."""
    state = {}
    quant_count = 0
    for name, param in m.named_parameters():
        if any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            state[name] = param.data.cpu()
            quant_count += 1
    for name, param in m.named_parameters():
        if not any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            if name not in state:
                state[name] = param.data.cpu()
    out_path = os.path.join(OUTPUT_DIR, f'model_{suffix}.safetensors')
    save_file(state, out_path)
    size = os.path.getsize(out_path)
    print(f'[SAVE] Saved {quant_count} quant tensors -> {out_path} ({size/1e9:.1f}GB)', flush=True)
    ALL_SAVED[0] = True
    return quant_count

# Also hook into post_quantize if it exists
if hasattr(model, 'post_quantize') and callable(model.post_quantize):
    original_post_quantize = model.post_quantize
    def hooked_post_quantize(*args, **kwargs):
        print(f'\n[HOOK] post_quantize called, saving first...', flush=True)
        save_weights(model, f'layer{layer_counter[0]}')
        layer_counter[0] += 1
        return original_post_quantize(*args, **kwargs)
    model.post_quantize = hooked_post_quantize

calib_texts = texts[:32]

print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
print(f'Output: {OUTPUT_DIR}', flush=True)
t1 = time.time()

try:
    model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
    print(f'Quantize completed in {(time.time()-t1)/60:.1f}min', flush=True)
except Exception as e:
    print(f'Quantize ended: {type(e).__name__}: {e}', flush=True)

# Final save attempt
n = save_weights(model, 'final')
print(f'\nFinal: saved {n} quantized tensors', flush=True)

# Copy config files
for f in os.listdir(MODEL_DIR):
    if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
        shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

print(f'Done in {(time.time()-t0)/60:.1f}min', flush=True)
print(f'Output: {OUTPUT_DIR}', flush=True)
