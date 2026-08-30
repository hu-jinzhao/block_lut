#!/usr/bin/env python3
"""GPU-accelerated BlockLUT GGUF converter."""
import torch, numpy as np, os, sys, json, gc
from safetensors import safe_open
from tqdm import tqdm

sys.path.insert(0, '/home/hh/llama.cpp/gguf-py')
from gguf import GGUFWriter, GGMLQuantizationType

MODEL = '/home/hh/LUT-MoE/models/qwen'
OUT = '/tmp/qwen_moe_blocklut.gguf'

print('[LUT-MoE] Loading LUT...')
lut = np.load(f'{MODEL}/blocklut_256.npy')
mid_gpu = torch.from_numpy((lut[:-1] + lut[1:]) / 2.0).float().to('cuda')

with open(f'{MODEL}/config.json') as f: cfg = json.load(f)

# Map HF architecture to GGUF architecture name
ARCH_MAP = {
    'Qwen2MoeForCausalLM': 'qwen2moe',
    'DeepseekV2ForCausalLM': 'deepseek2',
    'DeepseekV3ForCausalLM': 'deepseek2',
    'LlamaMoEForCausalLM': 'llama',
    'MixtralForCausalLM': 'mixtral',
}
arch_name = cfg.get('architectures',['Llama'])[0]
gguf_arch = ARCH_MAP.get(arch_name, arch_name.lower())
w = GGUFWriter(OUT, gguf_arch)
w.add_bool('lut-moe.enabled', True)
w.add_uint32('lut-moe.block_size', 128)
w.add_uint32('lut-moe.k', 256)
lut_bf16 = torch.from_numpy(lut).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)
w.add_array('lut-moe.lut_table', lut_bf16.tolist())
w.add_file_type(32)  # MOSTLY_BF16 (experts stored as BLOCKLUT, but ftype=BF16)

# Tokenizer (from model files)
vocab_path = os.path.join(MODEL, 'vocab.json')
merges_path = os.path.join(MODEL, 'merges.txt')
if os.path.exists(vocab_path) and os.path.exists(merges_path):
    import json as _json
    with open(vocab_path) as f: vocab = _json.load(f)
    with open(merges_path) as f: merges = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    # Sort vocabulary by token ID
    tokens = [None] * len(vocab)
    for token, tid in vocab.items():
        tokens[tid] = token
    w.add_tokenizer_model('gpt2')
    w.add_tokenizer_pre('default')
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_token_types([1] * len(tokens))  # 1=normal, 2=control, 3=unknown, 4=byte, 5=unused
    w.add_token_type_count(4)
    w.add_bos_token_id(cfg.get('bos_token_id', 151643))
    w.add_eos_token_id(cfg.get('eos_token_id', 151643))
    w.add_vocab_size(len(tokens))

# Standard llama.cpp hyperparameter keys
w.add_uint32(f'{gguf_arch}.context_length', cfg.get('max_position_embeddings', 8192))
w.add_uint32(f'{gguf_arch}.embedding_length', cfg['hidden_size'])
w.add_uint32(f'{gguf_arch}.block_count', cfg['num_hidden_layers'])
w.add_uint32(f'{gguf_arch}.feed_forward_length', cfg['intermediate_size'])
w.add_uint32(f'{gguf_arch}.head_count', cfg.get('num_attention_heads', 0))
w.add_uint32(f'{gguf_arch}.head_count_kv', cfg.get('num_key_value_heads', 0))
w.add_float32(f'{gguf_arch}.attention.layer_norm_rms_epsilon', cfg.get('rms_norm_eps', 1e-6))
w.add_uint32(f'{gguf_arch}.expert_count', cfg.get('num_experts', 0))
w.add_uint32(f'{gguf_arch}.expert_used_count', cfg.get('num_experts_per_tok', cfg.get('top_k', 2)))
w.add_uint32(f'{gguf_arch}.vocab_size', cfg.get('vocab_size', 0))
if 'intermediate_size_per_expert' in cfg:
    w.add_uint32(f'{gguf_arch}.expert_feed_forward_length', cfg['intermediate_size_per_expert'])
else:
    # Qwen2MoE: n_ff_exp = intermediate_size / expert_used_count
    n_ff_exp = cfg['intermediate_size'] // cfg.get('num_experts_per_tok', cfg.get('top_k', 2))
    w.add_uint32(f'{gguf_arch}.expert_feed_forward_length', n_ff_exp)
# Optional shared expert
if 'shared_expert_intermediate_size' in cfg:
    w.add_uint32(f'{gguf_arch}.expert_shared_feed_forward_length', cfg['shared_expert_intermediate_size'])

block_size = 128

def is_expert(name):
    return any(p in name for p in ['experts.','.experts.','ffn_gate_exps','ffn_up_exps','ffn_down_exps','gate_up_exps']) \
        and not any(p in name for p in ['shared_expert','shexp','gate_inp','router'])

files = sorted(f for f in os.listdir(MODEL) if f.endswith('.safetensors'))
n_exp = 0
for sf in tqdm(files, desc='Converting'):
    with safe_open(f'{MODEL}/{sf}', framework='pt', device='cpu') as f:
        for name in tqdm(list(f.keys()), desc=f'  {sf}', leave=False):
            t = f.get_tensor(name)
            if is_expert(name):
                x = t.float().to('cuda').ravel()
                n = x.numel(); nb = (n + block_size - 1) // block_size
                pad = nb * block_size - n
                if pad: x = torch.nn.functional.pad(x, (0, pad))
                b = x.view(nb, block_size)
                a = b.abs().max(dim=1).values.clamp(min=1e-12)
                nrm = (b / a[:, None]).ravel()
                idx = torch.bucketize(nrm, mid_gpu).to(torch.uint8)[:n]
                abs_u8 = a.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint8).tobytes()
                idx_u8 = idx.cpu().numpy().tobytes()
                p = np.frombuffer(idx_u8 + abs_u8, dtype=np.uint8)
                w.add_tensor(name, p, raw_shape=(len(p),), raw_dtype=GGMLQuantizationType.BLOCKLUT8)
                n_exp += 1
                del x, b, a, nrm, idx, p
            else:
                d = t.float().numpy().ravel().view(np.uint8)
                w.add_tensor(name, d, raw_dtype=GGMLQuantizationType.F32)
            del t; gc.collect()

print(f'Writing {n_exp} BlockLUT experts...')
w.open_output_file()
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
print(f'Done: {OUT}')
