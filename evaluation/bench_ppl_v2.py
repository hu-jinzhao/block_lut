#!/usr/bin/env python3
"""
PPL comparison: LUT 4bit vs RTN Group 4bit on WikiText-2.
Processes one layer at a time to fit in 24GB VRAM.
"""

import os, sys, json, time, math, glob
sys.path.insert(0, '/home/hh/LUT-MoE')

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
DTYPE = torch.bfloat16
DEVICE = "cuda"
GROUP_SIZE = 128
LUT_CODEBOOK = os.path.join(MODEL_DIR, "nested_lut_mapped16.npy")


# =========================================================================
# 1. Quantize weight matrices (CPU)
# =========================================================================

def quantize_rtn_group(weight, group_size=128, bits=4):
    """RTN group quantization on CPU. Returns dequantized weight."""
    w = weight.float()
    orig_shape = w.shape
    w_flat = w.reshape(-1, group_size)
    w_min = w_flat.amin(dim=-1, keepdim=True)
    w_max = w_flat.amax(dim=-1, keepdim=True)
    qmax = 2**bits - 1
    scale = ((w_max - w_min) / qmax).clamp(min=1e-10)
    zero = (-w_min / scale).round()
    q = (w_flat / scale + zero).round().clamp(0, qmax)
    dq = ((q - zero) * scale).reshape(orig_shape)
    return dq.to(weight.dtype)


def quantize_lut(weight, codebook, block_size=128):
    """BlockLUT quantization on CPU."""
    w = weight.float().reshape(-1)
    N = w.shape[0]
    nb = (N + block_size - 1) // block_size
    pad = torch.zeros(nb * block_size)
    pad[:N] = w
    blocks = pad.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    norm = (blocks / amax).reshape(-1)[:N]
    dists = (norm.unsqueeze(1) - codebook.float().unsqueeze(0)).abs()
    idx = dists.argmin(dim=1)
    dq = codebook[idx] * amax.reshape(-1)[torch.arange(N) // block_size].clamp(max=nb-1)
    return dq.reshape(weight.shape).to(weight.dtype)


# =========================================================================
# 2. Dataset
# =========================================================================

def get_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    with open(DATASET) as f:
        text = " ".join(json.load(f))
    enc = tokenizer.encode(text)
    print(f"WikiText-2: {len(enc)} tokens")
    return torch.tensor(enc, dtype=torch.long), tokenizer


# =========================================================================
# 3. PPL evaluation: process model layer by layer
# =========================================================================

def eval_ppl_for_method(quant_fn, method_name, data, tokenizer):
    """
    Load model, quantize expert weights, evaluate PPL.
    quant_fn: (weight) -> quantized_weight
    """
    import sys
    sys.path.insert(0, MODEL_DIR)

    from transformers import AutoModelForCausalLM, AutoConfig
    import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen_mod
    from types import MethodType

    print(f"\n{'='*50}")
    print(f"Evaluating: {method_name}")
    print(f"{'='*50}")

    # Monkey-patch to quantize experts during model creation
    orig_init = qwen_mod.Qwen2MoeSparseMoeBlock.__init__

    def patched_init(self, config):
        orig_init(self, config)
        with torch.no_grad():
            for i in range(len(self.experts)):
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(self.experts[i], name).weight
                    w.data.copy_(quant_fn(w.data.cpu()).to(w.device, dtype=w.dtype))

    qwen_mod.Qwen2MoeSparseMoeBlock.__init__ = patched_init

    print("Loading model (device_map='auto', this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, device_map='auto',
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    # PPL with sliding window
    max_len = 2048
    stride = 1024
    nlls = []
    total_tok = 0
    print(f"Computing PPL (seq_len={max_len}, stride={stride})...")
    t0 = time.time()

    for i in range(0, data.shape[0], stride):
        end = min(i + max_len, data.shape[0])
        if end - i < 128:
            break
        inp = data[i:end].unsqueeze(0).to(DEVICE)
        labels = inp.clone()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=DTYPE):
            out = model(input_ids=inp, labels=labels)
        loss = out.loss

        n = inp.shape[1] - 1
        nlls.append(loss.item() * n)
        total_tok += n

        if (i // stride + 1) % 10 == 0:
            ppl = math.exp(sum(nlls) / total_tok)
            print(f"  step {i // stride + 1}: PPL={ppl:.4f}")

    ppl = math.exp(sum(nlls) / total_tok)
    print(f"\n{method_name}: PPL={ppl:.4f} ({total_tok} tokens, {time.time()-t0:.1f}s)")

    del model
    torch.cuda.empty_cache()

    return ppl


# =========================================================================
# 4. Main
# =========================================================================

if __name__ == '__main__':
    data, tokenizer = get_data()
    results = {}

    # LUT 4bit
    codebook = torch.from_numpy(np.load(LUT_CODEBOOK))
    results['LUT_4bit'] = eval_ppl_for_method(
        lambda w: quantize_lut(w, codebook), "LUT 4bit", data, tokenizer)

    # RTN Group 4bit
    results['RTN_Group4'] = eval_ppl_for_method(
        lambda w: quantize_rtn_group(w, GROUP_SIZE, 4), "RTN Group 4bit", data, tokenizer)

    # Summary
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    for name, ppl in results.items():
        print(f"  {name:20s}: PPL = {ppl:.4f}")
    if len(results) >= 2:
        v = list(results.values())
        diff = v[0] - v[1]
        better = "LUT" if diff < 0 else "RTN"
        print(f"  {'─'*35}")
        print(f"  Difference: {diff:.4f} ({better} better)")
    print("=" * 50)
