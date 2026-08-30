# Copyright (c) 2026 <LUT_MoE / MINT, Nanjing University>.
# All rights reserved.
#
# Modifications and additions to this file are licensed under the
# Academic Non-Commercial License. See the LICENSE file in the
# project root for details.
#
# -------------------------------------------------------------------
# DERIVED FROM:
# EfficientMoE (Apache License 2.0)
# Copyright (c) EfficientMoE.
#
# The original code is licensed under the Apache License, Version 2.0.
# This file contains substantial modifications.
# -------------------------------------------------------------------



import functools
import gc
import json
import os
import re
from typing import Callable, Dict, Type, Union
from runtime.cache_planning import(
    PlanCache
)
import torch
import transformers
import numpy as np
from safetensors import safe_open
from tqdm import tqdm
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
import LUT_MoE
from experts import (
    ExpertExecutor,
    ExpertPrefetcher,
    ExpertPredictor
)
from customized import (
    DeepseekMoEBlock,
    SyncSwitchTransformersSparseMLP,
    Qwen2MoEBlock
)
import models
from runtime.hooks import *
from utils import (
    DELAY_PROFILE,
    COMPRESSION_RATIO_PROFILE,
    LUT_MoEConfig,
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param ,
    parse_expert_type,
) 
from utils.arguments import (
    copy_args_to_device,
    copy_kwargs_to_device
)

class LUT_MoEEngine(object):
    param_id = 0
    request_id = 0
    config = {}
    
    def __init__(
        self, 
        config: PretrainedConfig
    ):
        self.offload_exemption = set()
        self.expert_modules = []
        self.ckpt_files = []
        self.config = config
        
    def init(
        self,
        model_class: Type[PreTrainedModel],
        engine_config: Union[str, Dict, LUT_MoEConfig]
    ):
        self.model_class = model_class
        self.name_id_map = {}
        self.tensor_id_map = {} # Useless
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []
        self.offload_set = set()
        if isinstance(engine_config, str):
            self.lut_moe_config = LUT_MoEConfig.load_from_file(engine_config)
        elif isinstance(engine_config, Dict):
            self.lut_moe_config = LUT_MoEConfig.load_from_json(engine_config)
        elif isinstance(engine_config, LUT_MoEConfig):
            self.lut_moe_config = engine_config
        else:
            raise ValueError(
                "LUT_MoEConfig is not provided. Please provide a path to a config file or a dict."
            )
        self.checkpoint = self.lut_moe_config.offload_path
        os.makedirs(self.checkpoint, exist_ok=True)
        self.lut_moe_config.decompression_delay = DELAY_PROFILE[self.lut_moe_config.code_type] / self.lut_moe_config.num_file_chunks
        self.lut_moe_config.compression_ratio = COMPRESSION_RATIO_PROFILE[self.lut_moe_config.code_type]
        self.lut_sorted = None
        self.nested_lut_mapped64_uint16 = None
        self.nested_lut_mapped16_uint16 = None
        self._nested_lut_tier = getattr(self.lut_moe_config, 'lut_tier', 0)
        if self.lut_moe_config.code_type == "LUT":
            self._load_lut()
        if self.lut_moe_config.code_type in ("BLOCKLUT", "NESTEDLUT"):
            self._load_blocklut()
        if self.lut_moe_config.code_type == "NESTEDLUT":
            self._load_nested_lut_extras()
        self.lut_moe_config.gpu_pool_ratio, feasible = PlanCache(
            trace_path = self.lut_moe_config.trace_path,
            batch_size = self.lut_moe_config.batch_size,
            k = self.lut_moe_config.expert_topk,
            num_experts = self.lut_moe_config.num_experts,
            num_sparse_layers = self.lut_moe_config.num_expert_layers,
            tensors_per_expert = self.lut_moe_config.num_tensors_per_expert,
            num_elements_per_tensor = self.lut_moe_config.num_elements_per_expert,
            compression_ratio = self.lut_moe_config.compression_ratio,
            num_file_chunks = self.lut_moe_config.num_file_chunks,
            compute_pool_size = self.lut_moe_config.num_compute_threads,
            device_memory_ratio = self.lut_moe_config.device_memory_ratio,
            decompression_delay = self.lut_moe_config.decompression_delay,
            SM_IO_delay = self.lut_moe_config.sm_io_delay,
            step_size = 0.01
        )
        if not feasible or "switch" in self.lut_moe_config.offload_path:
            self.lut_moe_config.caching_algorithm = "LFU"
            print(F"[LUT_MoE Cache Planning] Switching to default LFU setting ...")
        self.lut_moe_engine = LUT_MoE.lut_moe_prefetch_handle(
            self.lut_moe_config.offload_path,
            self.lut_moe_config.offload_file_name,
            self.lut_moe_config.code_type,
            self.lut_moe_config.caching_algorithm,
            self.lut_moe_config.device_memory_ratio,
            self.lut_moe_config.gpu_pool_ratio,
            self.lut_moe_config.decompression_delay,
            self.lut_moe_config.sm_io_delay,
            self.lut_moe_config.num_compute_threads,
            self.lut_moe_config.num_file_chunks,
            self.lut_moe_config.prefetcher_topk,
            self.lut_moe_config.expert_topk,
            self.lut_moe_config.num_elements_per_expert,
            self.lut_moe_config.num_tensors_per_expert,
            self.lut_moe_config.num_expert_layers,
            self.lut_moe_config.num_experts,
            self.lut_moe_config.LZ4_accelerationLevel,
            self.lut_moe_config.LZ4HC_compressionLevel,
            self.lut_moe_config.ZSTD_compressionLevel,
            self.lut_moe_config.hyperparam_state_margin,
            self.lut_moe_config.bind_core
        )
        self.expert_executor = ExpertExecutor(
            lut_moe_config = self.lut_moe_config
        )
        if self.lut_moe_config.code_type == "LUT":
            self.lut_moe_engine.set_lut_table(self.lut_uint16)
        if self.lut_moe_config.code_type in ("BLOCKLUT", "NESTEDLUT"):
            self.lut_moe_engine.set_lut_table(self.lut_uint16)
        if self.lut_moe_config.code_type == "NESTEDLUT":
            self.lut_moe_engine.set_nested_lut_tables(
                self.lut_uint16,
                self.nested_lut_mapped64_uint16,
                self.nested_lut_mapped16_uint16
            )
            self._nested_lut_tier = getattr(self.lut_moe_config, 'lut_tier', 0)
        return self
        
    def __enter__(self):
        def torch_index_select_decorator( original_torch_index_select: Callable ):
            @functools.wraps( original_torch_index_select )
            def lut_moe_torch_index_select(input, dim, index):
                return original_torch_index_select(
                    input, dim, index.to(input.device)
                ).to("cuda:0")
            return lut_moe_torch_index_select

        def apply_to_model_decorator( original_apply_to_model: Callable ):
            @functools.wraps(original_apply_to_model)
            def lut_moe_apply_to_model(
                model_class,
                function
            ):
                for name, param in model_class.named_parameters(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    param.data = torch.zeros(
                        1,
                        dtype = param.dtype,
                        device = param.device,
                        pin_memory = True
                    )
                for name, buffer in model_class.named_buffers(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    buffer.data = torch.zeros(
                        1,
                        dtype=buffer.dtype,
                        device=buffer.device,
                        pin_memory=True
                    )
            return lut_moe_apply_to_model

        def cast_classifier_decorator( original_cast_classifier: Callable ):
            @functools.wraps( original_cast_classifier )
            def lut_moe_cast_classifier(
                model_class, *args, **kwargs
            ):
                original_data_ptr = model_class.classifier.weight.data.data_ptr()
                if original_data_ptr in self.offload_set:
                    self.offload_set.remove(
                        model_class.classifier.weight.data.data_ptr()
                    )
                    original_cast_classifier(
                        model_class, *args, **kwargs
                    )
                    new_data_ptr = model_class.classifier.weight.data.data_ptr()
                    self.offload_set.add(
                        model_class.classifier.weight.data.data_ptr()
                    )
                    self.lut_moe_engine.update_tensor_map(
                        original_data_ptr, new_data_ptr
                    )
                else:
                    original_cast_classifier(
                        model_class, *args, **kwargs
                    )
                    self.offload_set.add(
                        model_class.classifier.weight.data.data_ptr()
                    )
            return lut_moe_cast_classifier
            
        self.model_class._old_init = self.model_class.__init__
        self.model_class.__init__ = do_nothing_decorator(self.model_class._old_init)
        torch.nn.modules.module.Module._old_apply = (
            torch.nn.modules.module.Module.apply
        )
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )
        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select
        )
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )
        self.model_class._old_post_init = self.model_class.post_init
        self.model_class.post_init = do_nothing_decorator(self.model_class._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = do_nothing_decorator(
            PreTrainedModel._old_post_init
        )
        activate_empty_init()
        models.deepseek.modeling_deepseek._old_sparse_mlp = (
            models.deepseek.modeling_deepseek.DeepseekV2MoE
        )
        models.deepseek.modeling_deepseek.DeepseekV2MoE = DeepseekMoEBlock
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._old_cast_classifier = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier = cast_classifier_decorator(
            transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier
        )
        transformers.models.switch_transformers.modeling_switch_transformers._old_sparse_mlp = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP = SyncSwitchTransformersSparseMLP
        transformers.models.qwen2_moe.modeling_qwen2_moe._old_sparse_mlp = transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock
        transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = Qwen2MoEBlock        
        def from_pretrained_decorator(
            original_from_pretrained: Callable
        ):
            @functools.wraps( original_from_pretrained )
            def lut_moe_from_pretrained(
                model_class, *args, **kwargs
            ):
                name_id_map_file = os.path.join( self.checkpoint , "name_id_map.json" )
                self.model_name = model_name = args[0]
                self.num_layers, self.num_experts, self.num_encoder_layers = parse_moe_param(self.config)
                self.dtype = parse_expert_dtype( self.config )
                self.dtype_class = self.config.torch_dtype
                if (
                    not self.lut_moe_engine.is_tensor_index_initialized()
                    or not os.path.exists( name_id_map_file )
                ):
                    print("[LUT_MoE] Creating model from scratch ...", flush=True)
                    self.model_class.__init__ = self.model_class._old_init
                    empty_state_dict = {}
                    self.name_id_map = {}
                    for ckpt in tqdm(
                        self.ckpt_files, desc="Loading checkpoint files", smoothing=0
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(ckpt, framework="pt", device="cpu") as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)
                        batch_size = 0
                        max_batch_size = int(
                            self.lut_moe_config.num_compute_threads/self.lut_moe_config.num_file_chunks
                        )
                        tensor_id_list = []
                        tensor_list = []
                        batch_exponent_chunks = []
                        batch_sign_mantissa = []

                        for k,v in state_dict.items():
                            state_dict[k] = v.to(self.dtype).to("cpu")
                            is_sparse = True if ( "expert" in k and "shared_expert" not in k ) else False 
                            exponents_chunks, sign_mantissa = (
                                self.split_and_extract_from_tensor(
                                    state_dict[k],
                                    self.lut_moe_config.num_file_chunks
                                )
                            )
                            if is_sparse:
                                self.name_id_map[k] = self._generate_param_id()
                                tensor_id_list.append(self.name_id_map[k])
                                tensor_list.append(state_dict[k])
                                batch_exponent_chunks.append(exponents_chunks)
                                batch_sign_mantissa.append(sign_mantissa)
                                batch_size += 1
                            else:
                                if len(tensor_id_list) > 0:
                                    self.lut_moe_batch_offload_to_disk(
                                        tensor_id_list,
                                        tensor_list,
                                        batch_exponent_chunks, 
                                        batch_sign_mantissa
                                    )
                                    tensor_id_list = []
                                    tensor_list = []
                                    batch_exponent_chunks = []
                                    batch_sign_mantissa = []
                                    batch_size = 0
                                self.lut_moe_offload_to_disk(
                                    k,
                                    state_dict[k],
                                    exponents_chunks, 
                                    sign_mantissa,
                                    is_sparse
                                )

                            if batch_size == max_batch_size:
                                self.lut_moe_batch_offload_to_disk(
                                    tensor_id_list,
                                    tensor_list,
                                    batch_exponent_chunks, 
                                    batch_sign_mantissa
                                )
                                tensor_id_list = []
                                tensor_list = []
                                batch_exponent_chunks = []
                                batch_sign_mantissa = []
                                batch_size = 0
                        if len(tensor_id_list) > 0:
                            self.lut_moe_batch_offload_to_disk(
                                tensor_id_list, tensor_list, batch_exponent_chunks, batch_sign_mantissa
                            )
                            tensor_id_list = []
                            tensor_list = []
                            batch_exponent_chunks = []
                            batch_sign_mantissa = []
                            batch_size = 0

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    with open(name_id_map_file, "w") as f:
                        json.dump(self.name_id_map, f)
                        
                else:
                    print("[LUT_MoE] Loading model from offload_path ...", flush=True)
                    self.model_class.__init__ = self.model_class._old_init
                    with open(name_id_map_file, "r") as f:
                        self.name_id_map = json.load(f)
                        

                self.lut_moe_engine.open_offload_file()
                print("[LUT_MoE] Offload file opened successfully!")

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False
                )
                model = model_class._from_config(
                    self.config,
                    torch_dtype = torch.bfloat16,
                    attn_implementation = "flash_attention_2" if is_flash_attn_available else "eager"
                )
                self.expert_prefetcher = ExpertPrefetcher( self.config )
                self.expert_prefetcher.set_lut_moe_engine( self.lut_moe_engine )
                self.expert_dispatcher = LUT_MoE.expert_dispatcher(
                    self.num_experts,
                    self.lut_moe_config.num_expert_layers,
                    parse_expert_type( self.config ),
                    self.dtype
                )
                if self.lut_moe_config.prefetcher_topk > 0:
                    self.expert_predictor = ExpertPredictor(
                        self.lut_moe_config.num_expert_layers,
                        self.lut_moe_config.prefetcher_topk
                    )
                else:
                    self.expert_predictor = None
                base_model_prefix = model.base_model_prefix
                for name, param in model.named_parameters(recurse = True):
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[len(base_model_prefix)+1:]
                        if name_without_prefix in self.name_id_map:
                            self.name_id_map[name] = self.name_id_map[name_without_prefix]
                            self.name_id_map.pop(name_without_prefix)
                    param.lut_moe_id = self.name_id_map.get(name)
                if "lm_head.weight" not in self.name_id_map:
                    print(
                        "lm_head.weight not in name_id_map, add it as embed_tokens"
                    )
                    self.name_id_map["lm_head.weight"] = 0
                    self.name_id_map["encoder.embed_tokens.weight"] = 0
                    self.name_id_map["decoder.embed_tokens.weight"] = 0
                    model.lm_head.weight.ar_id = 0
                    model.model.encoder.embed_tokens.weight.ar_id = 0
                    model.model.decoder.embed_tokens.weight.ar_id = 0

                self.expert_tensor_map = dict()
                for name, id in self.name_id_map.items():
                    layer_id, expert_id = parse_expert_id(name, self.config)
                    if expert_id is not None:
                        self.expert_tensor_map[(layer_id, expert_id)] = id
                self.expert_prefetcher.expert_tensor_map = (
                    self.expert_tensor_map
                )
                self.expert_executor.set_expert_dispatcher( self.expert_dispatcher )
                self.expert_executor.set_expert_predictor( self.expert_predictor )
                module_idx = 0
                self.expert_layer_modules = []
                for module in model.modules():
                    if (
                        isinstance(module, DeepseekMoEBlock)
                        or isinstance(module, SyncSwitchTransformersSparseMLP)
                        or isinstance(module, Qwen2MoEBlock)
                    ):
                        module.lut_moe_engine = self.lut_moe_engine
                        module.lut_moe_config = self.lut_moe_config
                        module.expert_predictor = self.expert_predictor
                        module.expert_prefetcher = self.expert_prefetcher
                        module.expert_tensor_map = self.expert_tensor_map
                        module.expert_executor = self.expert_executor
                        module.layer_id = module_idx
                        self.expert_modules.append(module)
                        self.expert_layer_modules.append(module)
                        module_idx += 1
                self.setup_lut_moe_hooks(model)
                return model
            return lut_moe_from_pretrained
        
        self.model_class._old_from_pretrained = self.model_class.from_pretrained
        self.model_class.from_pretrained = classmethod(
            from_pretrained_decorator(self.model_class.from_pretrained)
        )
        return self
                
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.model_class.__init__ = self.model_class._old_init
        self.model_class.from_pretrained = self.model_class._old_from_pretrained
        torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply 
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select
        self.model_class.post_init = self.model_class._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init
        deactivate_empty_init()        

    def get_topology(self, model):
        name_lst = []
        ret_dict = {}
        for name, _ in model.named_parameters(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                print(f"{name} not in self.name_id_map")
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(self.name_id_map[name])
                        else:
                            ret_dict[stored_name][expert_name] = [ self.name_id_map[name] ]
                    else:
                        ret_dict[stored_name] = { expert_name: [ self.name_id_map[name] ] }
                        name_lst.append(stored_name)
                else:
                    match = re.match(r"(.*\.\d+\.)", name)
                    last_number_position = match.end() - 2
                    stored_name = name[:last_number_position+1]
                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]
                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for name, _ in model.named_buffers(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]
                        }
                        name_lst.append(stored_name)

                else:
                    matches = [match for match in re.finditer(r"\d", name)]
                    last_number_position = (
                        matches[-1].start() if matches else -1
                    )
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)
        for i in ret_dict.keys():
            if isinstance(ret_dict[i], dict):
                ret_dict[i] = list(ret_dict[i].values())

        topology = list(ret_dict.items())
        return topology        
                    
                    
    def setup_lut_moe_hooks(self, model):
        for name, param in model.named_parameters(recurse=True):
            if name not in self.name_id_map:
                continue
            self.lut_moe_engine.register( param.data , self.name_id_map[name] )
            self.offload_set.add( param.data.data_ptr() )
            if "shared" in name:
                self.offload_exemption.add( param.data.data_ptr() )
        for name, buffer in model.named_buffers(recurse=True):
            if name not in self.name_id_map:
                continue
            self.lut_moe_engine.register( buffer.data , self.name_id_map[name] )
            self.offload_set.add( buffer.data.data_ptr() )

        topo = self.get_topology(model)
        self.lut_moe_engine.set_topology(topo)
        
        @torch.no_grad()
        def _pre_forward_input_hook( module, input, kwargs, device, tensors ):
            self.lut_moe_engine.fetch_tensors( self.request_id, tensors )
            new_args = copy_args_to_device( device , input )
            new_kwargs = copy_kwargs_to_device( device , kwargs )
            return new_args, new_kwargs
        
        @torch.no_grad()
        def _post_forward_output_hook( module, input, output, device, tensors ):
            if isinstance(output, tuple):
                new_args = copy_args_to_device( device , output )
            elif isinstance(output, dict):
                new_args = copy_kwargs_to_device( device , output )
            else:
                new_args = output.to(device)
            return new_args

        def gen_args_hook(
            key , input_device_index , output_device_index , tensors
        ):
            keys = key.split(".")
            m = model
            for k in keys:
                if k.isdigit():
                    m = m[int(k)]
                else:
                    m = getattr( m, k )
            m.register_forward_pre_hook(
                functools.partial(
                    _pre_forward_input_hook,
                    device = input_device_index,
                    tensors = tensors
                ),
                prepend = True,
                with_kwargs = True
            )
            if "lm_head" in key:
                m.register_forward_hook(
                    functools.partial(
                        _post_forward_output_hook,
                        device = 0,
                        tensors = tensors
                    ),
                    prepend = False
                )
        expert_layer_id = 0
        output_device_index = None
        for key, tensors in topo:
            if "shared" in key or "lm_head" in key:
                key = key.split(".")[0]
                output_device_index = 0
            if "expert" in key:
                for expert_idx, expert_tensors in enumerate(tensors):
                    input_device_index = self.lut_moe_engine.get_node_default_device( expert_tensors )
                    self.expert_dispatcher.register_expert(expert_layer_id, expert_idx, expert_tensors)
                expert_layer_id += 1
        self._register_hooks_recursively(model)
        
        

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively( child, count = count )

        # Skip hook registration for modules without offloaded params
        has_offloaded = False
        for name, param in module.named_parameters(recurse = False):
            if param.data.data_ptr() in self.offload_set:
                has_offloaded = True
                break
        if not has_offloaded:
            for name, buf in module.named_buffers(recurse = False):
                if buf.data.data_ptr() in self.offload_set:
                    has_offloaded = True
                    break
        if not has_offloaded:
            return

        @torch.no_grad()
        def _pre_forward_module_hook( module, args, kwargs ):
            
            device_list = []
            for name, param in module.named_parameters(recurse = False):
                if param.data.data_ptr() not in self.offload_set:
                    # Non-expert params are already on GPU from setup
                    if param.data.device.type != 'cuda':
                        param.data = param.data.to("cuda:0")
                    continue
                self.offload_set.remove( param.data.data_ptr() )
                tensor_id = self.lut_moe_engine.begin( self.request_id , param)
                self.offload_set.add( param.data.data_ptr() )
                device_list.append( param.data.device )
            for name, buf in module.named_buffers(recurse = False):
                if buf.data.data_ptr() not in self.offload_set:
                    buf.data = buf.data.to("cuda:0")
                    continue
                self.offload_set.remove( buf.data_ptr() )
                tensor_id = self.lut_moe_engine.begin( self.request_id , buf )
                self.offload_set.add( buf.data_ptr() )
                
                device_list.append( buf.data.device )

        @torch.no_grad()
        def _post_forward_module_hook( module, input, output ):
            
            device_list = []
            param_not_offload = set()
            for param in module.parameters(recurse = False):
                if param.data.data_ptr() not in self.offload_set:
                    param_not_offload.add( param.data.data_ptr() )
                    continue
                self.offload_set.remove( param.data.data_ptr() )
                self.lut_moe_engine.end( self.request_id , param )
                self.offload_set.add( param.data.data_ptr() )
                
                device_list.append( param.data.device )
                
            for buf in module.buffers(recurse = False):
                if buf.data.data_ptr() not in self.offload_set:
                    continue
                    
                self.offload_set.remove( buf.data_ptr() )
                self.lut_moe_engine.end( self.request_id , buf )
                self.offload_set.add( buf.data_ptr() )
                
                device_list.append( buf.data.device )
                
            if param_not_offload:
                if isinstance(output, torch.Tensor):
                    return output.to("cuda:0")

        self.forward_hooks.append(
            module.register_forward_pre_hook( _pre_forward_module_hook, with_kwargs = True )
        )

        self.forward_hooks.append(
            module.register_forward_hook( _post_forward_module_hook )
        )


    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _generate_request_id(self):
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def split_and_extract_from_tensor(
        self,
        weight_tensor: torch.Tensor,
        num_file_chunks: int
    ):

        if self.lut_moe_config.code_type in ("BLOCKLUT", "NESTEDLUT"):
            indices, absmax_uint16 = self._quantize_weight_to_blocklut(weight_tensor)
            indices = np.ascontiguousarray(indices)
            num_elements = indices.size
            # Progressive 3-section bit-plane storage
            low = (indices & 0x0F).astype(np.uint8)
            mid = ((indices >> 4) & 0x03).astype(np.uint8)
            high = ((indices >> 6) & 0x03).astype(np.uint8)
            packed_low = (low[0::2] | (low[1::2] << 4)).astype(np.uint8)
            packed_mid = (mid[0::4] | (mid[1::4] << 2) | (mid[2::4] << 4) | (mid[3::4] << 6)).astype(np.uint8)
            packed_high = (high[0::4] | (high[1::4] << 2) | (high[2::4] << 4) | (high[3::4] << 6)).astype(np.uint8)
            packed = np.concatenate([packed_low, packed_mid, packed_high])
            elements_per_chunk = (packed.size + num_file_chunks - 1) // num_file_chunks
            indices_chunks = []
            for chunk_idx in range(num_file_chunks):
                start = chunk_idx * elements_per_chunk
                end = min(packed.size, (chunk_idx + 1) * elements_per_chunk)
                indices_chunks.append(packed[start:end].copy())
            absmax_bytes = np.ascontiguousarray(absmax_uint16).view(np.uint8)
            return indices_chunks, absmax_bytes

        weight_uint16 = weight_tensor.detach().view(torch.int16).numpy()
        exponents = ( (weight_uint16 >> 7) & 0xFF ).astype(np.uint8)
        sign_mantissa = (((weight_uint16 >> 15) & 0x1) << 7 | (weight_uint16 & 0x7F)).astype(np.uint8)
        exponents = np.ascontiguousarray( exponents.ravel() )
        sign_mantissa = np.ascontiguousarray( sign_mantissa.ravel() )
        num_elements = exponents.size
        elements_per_chunk = (num_elements + num_file_chunks - 1)//num_file_chunks
        exponents_chunks = []
        exponents = exponents.ravel()
        for chunk_idx in range(num_file_chunks):
            start = chunk_idx * elements_per_chunk
            end = min( num_elements , (chunk_idx+1) * elements_per_chunk )
            chunk_copy = exponents[start:end].copy()
            exponents_chunks.append( chunk_copy )
        sign_mantissa = sign_mantissa.copy()
        return exponents_chunks, sign_mantissa

    def lut_moe_offload_to_disk(
        self,
        param_name,
        tensor: torch.Tensor,
        exponents_chunks, 
        sign_mantissa,
        is_sparse
    ):
        self.name_id_map[param_name] = self._generate_param_id()
        if not self.lut_moe_engine.is_tensor_offloaded(
            self.name_id_map[param_name]
        ):
            self.lut_moe_engine.offload(
                self.name_id_map[param_name],
                tensor,
                exponents_chunks, 
                sign_mantissa,
                is_sparse
            )
            print(f"Successfully offloaded: {self.name_id_map[param_name]}")
        gc.collect()
        torch.cuda.empty_cache()
        
    def lut_moe_batch_offload_to_disk(
        self,
        tensor_id_list,
        tensor_list,
        batch_exponents_chunks, 
        batch_sign_mantissa
    ):
        if not tensor_id_list:
            return
        
        self.lut_moe_engine.batch_offload(
            tensor_id_list,
            tensor_list,
            batch_exponents_chunks, 
            batch_sign_mantissa
        )
        for idx in tensor_id_list:
            print(f"Successfully offloaded: {idx}")
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            self.name_id_map[param_name] = self._generate_param_id()
            if not self.lut_moe_engine.is_tensor_offloaded(
                self.name_id_map[param_name]
            ):
                self.lut_moe_engine.offload(
                    state_dict[param_name], self.name_id_map[param_name]
                )

        gc.collect()
        torch.cuda.empty_cache()


    def _load_lut(self):
        lut_path = self.lut_moe_config.lut_path
        if not lut_path or not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}")
        lut = np.load(lut_path)
        self.lut_sorted = lut.astype(np.float32)
        self.lut_sorted.sort()
        lut_bf16 = torch.from_numpy(self.lut_sorted.copy()).to(torch.bfloat16)
        self.lut_uint16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)
        midpoints = (self.lut_sorted[:-1] + self.lut_sorted[1:]) / 2.0
        mid_bf16 = torch.from_numpy(midpoints).to(torch.bfloat16)
        mid_u16 = mid_bf16.view(torch.int16).numpy().astype(np.uint16)
        self.lut_thresholds = np.where(mid_u16 & 0x8000, ~mid_u16, mid_u16 ^ np.uint16(0x8000)).astype(np.uint16)
        print(f"[LUT_MoE] LUT loaded: {len(self.lut_sorted)} entries")

    def _load_blocklut(self):
        lut_path = self.lut_moe_config.lut_path
        if not lut_path or not os.path.exists(lut_path):
            raise FileNotFoundError(f"BlockLUT file not found: {lut_path}")
        lut = np.load(lut_path)
        self.blocklut_sorted = lut.astype(np.float32)
        self.blocklut_sorted.sort()
        n_orig = len(self.blocklut_sorted)
        lut_for_gpu = self.blocklut_sorted.copy()
        if len(lut_for_gpu) < 256:
            lut_for_gpu = np.pad(lut_for_gpu, (0, 256 - len(lut_for_gpu)), constant_values=lut_for_gpu[-1])
        lut_bf16 = torch.from_numpy(lut_for_gpu).to(torch.bfloat16)
        self.lut_uint16 = lut_bf16.view(torch.int16).numpy().astype(np.uint16)
        print(f"[LUT_MoE] BlockLUT loaded: {n_orig} entries, padded to {len(lut_for_gpu)} for GPU")

    def _load_nested_lut_extras(self):
        lut_path = self.lut_moe_config.lut_path
        base = os.path.dirname(lut_path) if lut_path else '/home/hh/LUT-MoE/models/qwen'
        mapped64_path = os.path.join(base, "nested_lut_mapped64.npy")
        mapped16_path = os.path.join(base, "nested_lut_mapped16.npy")
        if not os.path.exists(mapped64_path):
            raise FileNotFoundError(f"Nested mapped64 LUT not found: {mapped64_path}")
        if not os.path.exists(mapped16_path):
            raise FileNotFoundError(f"Nested mapped16 LUT not found: {mapped16_path}")
        self.nested_lut_mapped64_uint16 = np.load(mapped64_path)
        self.nested_lut_mapped16_uint16 = np.load(mapped16_path)
        print(f"[LUT_MoE] Nested LUT extras loaded: mapped64={len(self.nested_lut_mapped64_uint16)} entries, mapped16={len(self.nested_lut_mapped16_uint16)} entries")

    def set_all_experts_lut_tier(self, tier):
        self.lut_moe_engine.set_all_sparse_nodes_lut_tier(tier)

    def _quantize_weight_to_blocklut(self, weight_tensor):
        x = weight_tensor.detach().to(torch.float32).numpy().ravel()
        n = x.size; bs = 128; nb = (n + bs - 1) // bs; pad = nb * bs - n
        if pad > 0: x = np.pad(x, (0, pad))
        blocks = x.reshape(nb, bs)
        absmax_vals = np.max(np.abs(blocks), axis=1)
        absmax_vals = np.maximum(absmax_vals, 1e-12)
        normalized = blocks / absmax_vals[:, np.newaxis]
        midpoints = (self.blocklut_sorted[:-1] + self.blocklut_sorted[1:]) / 2.0
        indices = np.searchsorted(midpoints, normalized.ravel()).astype(np.uint8)
        absmax_bf16 = torch.from_numpy(absmax_vals).to(torch.bfloat16)
        absmax_uint16 = absmax_bf16.view(torch.int16).numpy().astype(np.uint16)
        return indices[:n], absmax_uint16

    def _quantize_weight_to_lut(self, weight_tensor):
        flat_u16 = weight_tensor.detach().view(torch.int16).numpy().astype(np.uint16).ravel()
        flat_mono = np.where(flat_u16 & 0x8000, ~flat_u16, flat_u16 ^ np.uint16(0x8000)).astype(np.uint16)
        indices = np.searchsorted(self.lut_thresholds, flat_mono).astype(np.uint8)
        return indices

    def _post_forward_output_hook_setup(self):
        pass

    def clean_up(self):
        models.deepseek.modeling_deepseek.DeepseekV2MoE = models.deepseek.modeling_deepseek._old_sparse_mlp
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._cast_classifier = transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router._old_cast_classifier
        transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP = transformers.models.switch_transformers.modeling_switch_transformers._old_sparse_mlp
        transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = transformers.models.qwen2_moe.modeling_qwen2_moe._old_sparse_mlp