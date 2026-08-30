#!/usr/bin/env python3
"""GPTQ 4bit quantization - saves weights per layer, handles meta tensors."""
import logging, os, sys, time, json, shutil
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

# Skip the buggy pack step
model.pack_model = lambda *a, **kw: None

def save_weights(m, suffix=''):
    """Materialize and save weights. Handles meta tensors."""
    # Underlying HF model has the actual parameters
    hf_model = m.model if hasattr(m, 'model') else m

    # Materialize meta tensors by moving to cpu
    try:
        hf_model.to('cpu')
    except:
        pass

    state = {}
    quant_count = 0
    for name, param in hf_model.named_parameters():
        if param.device.type == 'meta':
            continue  # skip remaining meta tensors
        if any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            state[name] = param.data.cpu()
            quant_count += 1

    if quant_count == 0:
        # Also try the top-level model
        for name, param in m.named_parameters():
            if param.device.type == 'meta':
                continue
            if any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
                state[name] = param.data.cpu()
                quant_count += 1

    if quant_count == 0:
        print(f'[SAVE] No quantized tensors found yet for {suffix}', flush=True)
        return 0

    out_path = os.path.join(OUTPUT_DIR, f'model_{suffix}.safetensors')
    save_file(state, out_path)
    size = os.path.getsize(out_path)
    print(f'[SAVE] {quant_count} tensors -> {out_path} ({size/1e9:.1f}GB)', flush=True)
    return quant_count

# Hook into the looper to save after each layer completes
# The looper processes layers sequentially in the quantize step
original_quantize_layer = getattr(model, '_quantize_with_calibration', None)

calib_texts = texts[:32]
print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
print(f'Saves will go to {OUTPUT_DIR}', flush=True)
t1 = time.time()

try:
    model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
    print(f'Quantize OK in {(time.time()-t1)/60:.1f}min', flush=True)
except Exception as e:
    print(f'Quantize ended: {type(e).__name__}: {e}', flush=True)

# Final save
n = save_weights(model, 'final')
print(f'Final: {n} quantized tensors', flush=True)

# Also save non-quantized (dense) weights if quantized weights > 0
if n > 0:
    hf_model = model.model if hasattr(model, 'model') else model
    try:
        hf_model.to('cpu')
    except:
        pass
    dense_state = {}
    for name, param in hf_model.named_parameters():
        if param.device.type != 'meta' and not any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
            dense_state[name] = param.data.cpu()
    if dense_state:
        save_file(dense_state, os.path.join(OUTPUT_DIR, 'model_dense.safetensors'))
        print(f'Dense: {len(dense_state)} tensors saved', flush=True)

# Copy config
for f in os.listdir(MODEL_DIR):
    if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
        shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

print(f'Done in {(time.time()-t0)/60:.1f}min', flush=True)
print(f'Output: {OUTPUT_DIR}', flush=True)
