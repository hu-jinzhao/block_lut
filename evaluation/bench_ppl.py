#!/usr/bin/env python3
"""
PPL comparison: LUT 4bit vs RTN group 4bit on WikiText-2 (8k context).

Measures perplexity for both quantization methods on the same model/data.
"""

import os, sys, json, time, math, glob
sys.path.insert(0, '/home/hh/LUT-MoE')

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file, load_file

MODEL_DIR = "/home/hh/LUT-MoE/models/qwen"
DATASET = "/home/hh/LUT-MoE/evaluation/dataset/wikitext2_test.json"
MAX_LENGTH = 8192
STRIDE = 1024
DTYPE = torch.bfloat16
DEVICE = "cuda"


# =========================================================================
# 1. Quantization methods
# =========================================================================

def quantize_rtn_group(weight, group_size=128, bits=4):
    """
    RTN group-wise quantization.
    weight: [rows, cols] bf16
    Returns: (qweight_int32, scales, zeros)
    """
    w = weight.float()
    rows, cols = w.shape
    # Reshape for group quantization along cols
    assert cols % group_size == 0, f"cols={cols} not divisible by gs={group_size}"
    w_g = w.reshape(rows, -1, group_size)  # [rows, num_groups, gs]

    # min/max per group
    w_max = w_g.amax(dim=-1, keepdim=True)  # [rows, num_groups, 1]
    w_min = w_g.amin(dim=-1, keepdim=True)

    # Quantization levels
    qmax = 2**bits - 1
    scale = (w_max - w_min) / qmax
    scale = scale.clamp(min=1e-10)
    zero = -w_min / scale

    # Quantize
    q = (w_g / scale + zero).round().clamp(0, qmax)

    # Dequantize for reconstruction error check
    w_dq = (q - zero) * scale

    return w_dq.reshape(rows, cols), scale.reshape(rows, -1), zero.reshape(rows, -1)


def quantize_lut(weight, codebook, block_size=128):
    """BlockLUT quantization (our method)."""
    flat = weight.reshape(-1).to(torch.float32)
    N = flat.shape[0]
    num_blocks = (N + block_size - 1) // block_size
    padded = torch.zeros(num_blocks * block_size, dtype=torch.float32)
    padded[:N] = flat
    blocks = padded.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = (blocks / absmax).reshape(-1)[:N]

    # Encode
    cb_f = codebook.float()
    dists = (normalized.unsqueeze(1) - cb_f.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1)

    # Dequantize
    dq = codebook[indices] * absmax.reshape(-1)[torch.arange(N, device=weight.device) // block_size].clamp(max=absmax.shape[0]-1)
    return dq.reshape(weight.shape)


# =========================================================================
# 2. Data loading
# =========================================================================

def load_wikitext(path, max_length=MAX_LENGTH):
    """Load WikiText-2 test set, encode, return list of chunks."""
    with open(path) as f:
        texts = json.load(f)
    full_text = " ".join(texts)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    enc = tokenizer.encode(full_text)
    print(f"  WikiText-2: {len(enc)} tokens")

    # Split into overlapping chunks of max_length
    chunks = []
    for i in range(0, len(enc), max_length):
        chunk = enc[i:i + max_length]
        if len(chunk) < 100:
            break  # skip last fragment
        chunks.append(torch.tensor(chunk, dtype=torch.long))
    print(f"  {len(chunks)} chunks (max_len={max_length}, stride={max_length})")
    return chunks, tokenizer


# =========================================================================
# 3. Model loading and PPL evaluation
# =========================================================================

def load_qwen_model(quantize_fn=None):
    """Load Qwen2-MoE model, optionally quantize expert weights."""
    from transformers import AutoConfig
    from models.qwen import Qwen2MoEBlock

    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # Load with custom MoE that can apply quantization
    from models.model_utils import rotate_half
    from transformers.models.qwen2_moe.modeling_qwen2_moe import (
        Qwen2MoeForCausalLM, Qwen2MoeModel, Qwen2MoeDecoderLayer,
        Qwen2MoeSparseMoeBlock, Qwen2MoeRMSNorm, Qwen2MoeAttention
    )

    import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen_mod

    # Store original for restore
    orig_moe = qwen_mod.Qwen2MoeSparseMoeBlock

    class PatchedMoE(Qwen2MoeSparseMoeBlock):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self._quantize_fn = quantize_fn
            if quantize_fn is not None:
                self._apply_quantization()

        def _apply_quantization(self):
            """Quantize all expert weights."""
            with torch.no_grad():
                for i in range(len(self.experts)):
                    expert = self.experts[i]
                    for name in ['gate_proj', 'up_proj', 'down_proj']:
                        w = getattr(expert, name).weight
                        w_dq, _, _ = quantize_fn(w, group_size=128, bits=4)
                        w.data.copy_(w_dq.to(w.dtype))

    qwen_mod.Qwen2MoeSparseMoeBlock = PatchedMoE

    print("Loading model...")
    t0 = time.time()
    model = Qwen2MoeForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    model.eval()

    # Restore
    qwen_mod.Qwen2MoeSparseMoeBlock = orig_moe

    return model


def compute_ppl(model, chunks, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    """Compute perplexity using sliding window."""
    model = model.to(DEVICE)
    nlls = []
    total_tokens = 0

    print(f"Computing PPL ({len(chunks)} chunks, max_len={max_length})...")
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss * (input_ids.shape[1] - 1)

        n_tokens = input_ids.shape[1] - 1
        nlls.append(neg_log_likelihood.item())
        total_tokens += n_tokens

        if (idx + 1) % 5 == 0:
            ppl = math.exp(sum(nlls) / total_tokens)
            print(f"  [{idx+1}/{len(chunks)}] PPL={ppl:.2f}")

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    print(f"\nPerplexity: {ppl:.4f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {time.time()-t0:.1f}s")

    return ppl


# =========================================================================
# 4. Main
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LUT-MoE vs GPTQ PPL Comparison")
    print("=" * 60)

    # Load data
    print("\n[Data]")
    chunks, tokenizer = load_wikitext(DATASET)

    # Test 1: LUT 4bit
    print("\n--- LUT 4bit (BlockLUT K=16) ---")
    codebook_path = os.path.join(MODEL_DIR, "blocklut_256.npy")
    codebook = torch.from_numpy(np.load(codebook_path)).to(torch.bfloat16)

    from models.qwen import Qwen2MoEBlock
    orig_moe_block = Qwen2MoEBlock

    class LUTMoEBlock(Qwen2MoEBlock):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self._quantize_moe()

        def _quantize_moe(self):
            with torch.no_grad():
                for i in range(len(self.experts)):
                    expert = self.experts[i]
                    for name in ['gate_proj', 'up_proj', 'down_proj']:
                        w = getattr(expert, name).weight
                        w_dq = quantize_lut(w, codebook.to(w.device))
                        w.data.copy_(w_dq.to(w.dtype))

    # This approach requires monkey-patching the model class
    # Let's use a simpler method: patch after loading

    # Load model without quantization
    from transformers import AutoModelForCausalLM
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Apply LUT quantization to all expert weights
    print("Applying LUT 4bit quantization...")
    cb = codebook.to(DEVICE)
    n_quantized = 0
    for name, param in model.named_parameters():
        if 'experts.' in name and 'shared_expert' not in name and param.dim() >= 2:
            w_dq = quantize_lut(param.data.cpu(), cb.cpu()).to(param.device, dtype=param.dtype)
            param.data.copy_(w_dq)
            n_quantized += 1
    print(f"  Quantized {n_quantized} expert weight tensors")

    ppl_lut = compute_ppl(model, chunks, tokenizer)

    # Test 2: RTN group 4bit
    print("\n--- RTN Group 4bit (group_size=128) ---")
    # Reload model fresh
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Applying RTN Group 4bit quantization...")
    for name, param in model.named_parameters():
        if 'experts.' in name and 'shared_expert' not in name and param.dim() >= 2:
            w_dq, _, _ = quantize_rtn_group(param.data.cpu(), group_size=128, bits=4)
            param.data.copy_(w_dq.to(param.device, dtype=param.dtype))

    ppl_rtn = compute_ppl(model, chunks, tokenizer)

    # Test 3: Baseline (no quantization, if memory allows)
    # (Skip since model doesn't fit in 24GB without quantization)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LUT 4bit (BlockLUT K=16):  PPL = {ppl_lut:.4f}")
    print(f"  RTN Group 4bit (gs=128):   PPL = {ppl_rtn:.4f}")
    print(f"  Difference:                 PPL = {ppl_lut - ppl_rtn:.4f}")
    print(f"  {'LUT is better' if ppl_lut < ppl_rtn else 'RTN is better'}")
    print("=" * 60)
