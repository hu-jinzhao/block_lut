#!/usr/bin/env python3
"""Convert HF model directly to BlockLUT GGUF."""
import sys, os, json, gc, numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, '/home/hh/llama.cpp/gguf-py')
from gguf import GGUFWriter, GGMLQuantizationType, TensorNameMap, MODEL_TENSOR
from gguf.quants import quant_shape_to_byte_shape

# Simple Qwen2MoE tensor name mapping
def map_tensor_name(hf_name: str, n_blocks: int) -> str:
    # Direct mappings
    direct = {
        'model.embed_tokens.weight': 'token_embd.weight',
        'model.norm.weight': 'output_norm.weight',
        'lm_head.weight': 'output.weight',
    }
    if hf_name in direct:
        return direct[hf_name]
    # Layer patterns
    import re
    m = re.match(r'model\.layers\.(\d+)\.(.+)', hf_name)
    if m:
        bid = m.group(1)
        rest = m.group(2)
        suffix = '.weight' if hf_name.endswith('.weight') else ('.bias' if hf_name.endswith('.bias') else '')
        core = rest[:-len(suffix)] if suffix else rest

        # Expert tensors
        em = re.match(r'mlp\.experts\.(\d+)\.(.+)', core)
        if em:
            eid, wname = em.group(1), em.group(2)
            lut = {'gate_proj': 'ffn_gate_exps', 'up_proj': 'ffn_up_exps', 'down_proj': 'ffn_down_exps'}
            gguf_name = lut.get(wname)
            if gguf_name:
                return f'blk.{bid}.{gguf_name}.weight'
            # gate_up_proj merged
            if wname == 'gate_up_proj':
                return f'blk.{bid}.gate_up_exps.weight'

        # Shared expert
        for p, gguf in [('mlp.shared_expert.gate_proj', 'ffn_gate_shexp'),
                          ('mlp.shared_expert.up_proj', 'ffn_up_shexp'),
                          ('mlp.shared_expert.down_proj', 'ffn_down_shexp'),
                          ('mlp.shared_expert_gate', 'ffn_gate_inp_shexp')]:
            if core.startswith(p):
                return f'blk.{bid}.{gguf}.weight'

        # Attention
        attn_map = {
            'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
            'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
            'self_attn.qkv_proj': 'attn_qkv',
        }
        for p, gguf in attn_map.items():
            if core.startswith(p):
                return f'blk.{bid}.{gguf}.weight'

        # Normally patterns
        norm_map = {
            'input_layernorm': 'attn_norm',
            'post_attention_layernorm': 'ffn_norm',
        }
        for p, gguf in norm_map.items():
            if core == p:
                return f'blk.{bid}.{gguf}.weight'

        # MLP gate/ffn
        if core == 'mlp.gate_proj':
            return f'blk.{bid}.ffn_gate_inp.weight'

    return hf_name  # fallback

# Check the mapping works
for test, expected in [
    ('model.embed_tokens.weight', 'token_embd.weight'),
    ('model.layers.0.input_layernorm.weight', 'blk.0.attn_norm.weight'),
    ('model.layers.0.mlp.experts.5.gate_proj.weight', 'blk.0.ffn_gate_exps.weight'),
    ('model.layers.0.mlp.gate_proj.weight', 'blk.0.ffn_gate_inp.weight'),
]:
    got = map_tensor_name(test, 24)
    ok = '✅' if got == expected else '❌'
    print(f'  {ok} {test:60s} → {got}')
    if got != expected:
        print(f'      expected: {expected}')

MODEL = sys.argv[1]
OUT = sys.argv[2]

# Config
with open(f'{MODEL}/config.json') as f:
    cfg = json.load(f)

GGUF_ARCH = 'qwen2moe'
LUT_PATH = f'{MODEL}/blocklut_256.npy'

# Load LUT to GPU
lut = np.load(LUT_PATH)
mid_gpu = torch.from_numpy((lut[:-1] + lut[1:]) / 2.0).float().to('cuda')
print(f'LUT: {len(lut)} entries')

w = GGUFWriter(OUT, GGUF_ARCH)

# ── Write all config as metadata ──
# Architecture
w.add_architecture()
w.add_quantization_version(2)
w.add_file_type(32)  # MOSTLY_BF16
w.add_custom_alignment(32)
w.add_context_length(cfg.get('max_position_embeddings', 8192))
w.add_embedding_length(cfg['hidden_size'])
w.add_block_count(cfg['num_hidden_layers'])
w.add_feed_forward_length(cfg['intermediate_size'])
w.add_head_count(cfg['num_attention_heads'])
w.add_head_count_kv(cfg.get('num_key_value_heads', cfg['num_attention_heads']))
w.add_layer_norm_rms_eps(cfg.get('rms_norm_eps', 1e-6))
w.add_expert_count(cfg.get('num_experts', 60))
w.add_expert_used_count(cfg.get('num_experts_per_tok', cfg.get('top_k', 4)))
w.add_vocab_size(cfg.get('vocab_size', 151936))

# ROPE
w.add_rope_dimension_count(cfg.get('head_dim', 128))

# Qwen2MoE specific
moe_intermediate_size = cfg.get('moe_intermediate_size',
    cfg.get('intermediate_size', 5632) // cfg.get('num_experts_per_tok', cfg.get('top_k', 4)))
w.add_expert_feed_forward_length(moe_intermediate_size)
if 'shared_expert_intermediate_size' in cfg:
    w.add_expert_shared_feed_forward_length(cfg['shared_expert_intermediate_size'])

# LUT-MoE metadata
w.add_bool('lut-moe.enabled', True)
w.add_uint32('lut-moe.block_size', 128)
w.add_uint32('lut-moe.k', 256)

# ── Tokenizer ──
vocab_path = f'{MODEL}/vocab.json'
merges_path = f'{MODEL}/merges.txt'
if os.path.exists(vocab_path) and os.path.exists(merges_path):
    with open(vocab_path) as f: vocab = json.load(f)
    with open(merges_path) as f: merges = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    tokens = [None] * len(vocab)
    for token, tid in vocab.items():
        if tid < len(tokens): tokens[tid] = token
    w.add_tokenizer_model('gpt2')
    w.add_tokenizer_pre('default')
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_token_types([1] * len(tokens))
    w.add_token_type_count(4)
    w.add_bos_token_id(cfg.get('bos_token_id', 151643))
    w.add_eos_token_id(cfg.get('eos_token_id', 151643))
    print(f'Tokenizer: {len(tokens)} tokens, {len(merges)} merges')

# ── Process tensors ──
files = sorted(f for f in os.listdir(MODEL) if f.endswith('.safetensors'))
print(f'Processing {len(files)} shards...')

def is_expert(name):
    return any(p in name for p in ['experts.','.experts.','ffn_gate_exps','ffn_up_exps','ffn_down_exps','gate_up_exps']) \
        and not any(p in name for p in ['shared_expert','shexp','gate_inp','router'])

n_exp = 0
for sf in tqdm(files, desc='Shard'):
    with safe_open(f'{MODEL}/{sf}', framework='pt', device='cpu') as f:
        for name in tqdm(list(f.keys()), desc='  Tensor', leave=False):
            tensor = f.get_tensor(name)
            # Map to GGUF tensor name
            gguf_name = map_tensor_name(name, cfg['num_hidden_layers'])
            if is_expert(name):
                # BlockLUT quantize on GPU
                x = tensor.float().to('cuda').ravel()
                n = x.numel(); bs = 128; nb = (n + bs - 1) // bs
                if nb * bs > n: x = torch.nn.functional.pad(x, (0, nb * bs - n))
                b = x.view(nb, bs)
                a = b.abs().max(dim=1).values.clamp(min=1e-12)
                nrm = (b / a[:, None]).ravel()
                idx = torch.bucketize(nrm, mid_gpu).to(torch.uint8)[:n]
                abs_u8 = a.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint8).tobytes()
                idx_u8 = idx.cpu().numpy().tobytes()
                p = np.frombuffer(idx_u8 + abs_u8, dtype=np.uint8)
                w.add_tensor(gguf_name, p, raw_shape=(len(p),), raw_dtype=GGMLQuantizationType.BLOCKLUT8)
                n_exp += 1
                del x, b, a, nrm, idx, p
            else:
                # BF16: view as int16 then uint8 to avoid bf16→numpy issue
                t_bytes = tensor.view(torch.int16).numpy().view(np.uint8)
                w.add_tensor(gguf_name, t_bytes, raw_dtype=GGMLQuantizationType.BF16)
            del tensor
            gc.collect()

print(f'Writing GGUF ({n_exp} BlockLUT experts)...')
w.open_output_file()
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
print(f'Done: {OUT}')
