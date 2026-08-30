#!/usr/bin/env python3
"""GPTQ 4bit quantization for Qwen-MoE, saves weights manually."""
import logging, os, sys, time, json, shutil
logging.disable(logging.CRITICAL)
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'
os.environ['HF_HUB_OFFLINE'] = '1'

from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
from safetensors.torch import save_file

MODEL_DIR = '/home/hh/LUT-MoE/models/qwen'
OUTPUT_DIR = '/tmp/qwen_gptq_final'

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
print(f'Quantizing ({len(calib_texts)} samples)...', flush=True)
t1 = time.time()
model.quantize(calibration=calib_texts, batch_size=1, tokenizer=tokenizer)
print(f'Quantized in {(time.time()-t1)/60:.1f}min', flush=True)

# Save quantized weights manually
print('Saving weights...', flush=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {}
for name, param in model.named_parameters():
    if any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
        state[name] = param.data.cpu()
        print(f'  quant: {name} {param.shape}', flush=True)

for name, param in model.named_parameters():
    if not any(k in name for k in ['qweight', 'qzeros', 'scales', 'g_idx']):
        state[name] = param.data.cpu()

print(f'Saving {len(state)} tensors...', flush=True)
save_file(state, os.path.join(OUTPUT_DIR, 'model.safetensors'))

# Copy config files
for f in os.listdir(MODEL_DIR):
    if f.endswith(('.json', '.py', '.txt')) and not f.startswith('model-'):
        shutil.copy2(os.path.join(MODEL_DIR, f), os.path.join(OUTPUT_DIR, f))

print(f'Saved to {OUTPUT_DIR}', flush=True)
print(f'Total: {(time.time()-t0)/60:.1f}min', flush=True)
