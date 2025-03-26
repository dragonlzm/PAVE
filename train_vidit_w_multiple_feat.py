# Adopted from https://github.com/haotian-liu/LLaVA. 
# This training script aims to handle multiple side-channel at the same time 

import os
import pathlib
import torch
import transformers
import ipdb
import sys
from transformers import AutoConfig, BitsAndBytesConfig, AutoTokenizer
from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libs.utils.model_trainer import VideoModelTrainer
from libs import conversation_lib as conversation_lib
from libs.model import *
from libs.utils.train_utils import parse_argument_classes, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_videotrainer, get_peft_state_non_lora_maybe_zero_3_with_state_dict
from libs.mm_utils import tokenizer_vision_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from libs.dataset.base_dataset import make_video_supervised_data_module
from libs.utils.train_utils import MODEL_ARGUMENTS_MAPPING, DATA_ARGUMENTS_MAPPING
# from utils import prepare_video_model
import warnings


def filter_the_state_dict(state_dict, keyword):
    # filter the state dict using the keyword
    new_state_dict = {key: state_dict[key] for key in state_dict if keyword in key}
    return new_state_dict


def prepare_video_model_multiple_input(training_args, model_args, data_args, compute_dtype, attn_implementation, model_arg_name, data_arg_name):
    '''
        prepare the model 
        1. load the base model and the patches
        2. freeze part of the model, unfreeze the lora
        3. init the audio encoder part
        
    '''
    kwargs = {"device_map": {"": training_args.device}}
    if training_args.bits == 8:
        kwargs['load_in_8bit'] = True
    elif training_args.bits == 4:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    kwargs['attn_implementation'] = attn_implementation
    
    # ipdb.set_trace() # check the model
    model_path = model_args.model_path
    model_name = get_model_name_from_path(model_path)
    model_base = model_args.model_base

    if 'pave' in model_name.lower():
        # ipdb.set_trace()
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. \
                          If you are loading a LoRA model, please provide the `model_base` argument. \
                          Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
        if 'lora' in model_name.lower() and model_base is not None:
            from libs.model.language_model.pave_qwen2_multi_sides import PAVEQwen2MultiSidesConfig, PAVEQwen2MultiSidesForCausalLM
            # from libs.model.language_model.hyper_vidit_qwen2 import HyperViditQwen2Config, HyperViditQwen2ForCausalLM
            # init the base LLM model
            # ipdb.set_trace() # check the name of the model
            base_model_cfg = PAVEQwen2MultiSidesConfig.from_pretrained(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = PAVEQwen2MultiSidesForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_model_cfg, **kwargs)
            lora_cfg_pretrained = PAVEQwen2MultiSidesConfig.from_pretrained(model_path)
        
            # reshaping the language head of the model
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                print('re-initing the lm_head')
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            ## init the vision module 
            print('Init the vision module ...')
            # merge the training config with the lora config
            # the lora_cfg_pretrained contains the parameters sended in through command line
            # the default_model_arg contains the default model parameters
            default_model_arg = MODEL_ARGUMENTS_MAPPING[model_arg_name]
            default_data_args = DATA_ARGUMENTS_MAPPING[data_arg_name]
            print('Warning: we are using MODEL_ARGUMENTS_MAPPING:', model_arg_name, 'DATA_ARGUMENTS_MAPPING:', data_arg_name)
            
            # set the value in lora_cfg_pretrained as default_model_arg, we should use lora_cfg_pretrained latter on
            for key in default_model_arg.__dict__:
                if not key.startswith('__'):
                    if not hasattr(lora_cfg_pretrained, key):
                        setattr(lora_cfg_pretrained, key, default_model_arg.__dict__[key])

            # re-instantiate the Video backbone and the SSM
            # ipdb.set_trace() # check the video module init, and the merged config
            if default_model_arg.video_tower is not None:
                lora_cfg_pretrained.image_size = default_data_args.image_size
                model.get_model().initialize_vision_modules(
                    model_args=lora_cfg_pretrained,
                    fsdp=None,
                ) 
                
                # load the pretrained temporal aggregator weights
                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    print('Loading additional LLaVA weights..., from:', model_path)
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download
                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            subfolder=subfolder)
                        return torch.load(cache_file, map_location='cpu')
                    non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(filter_the_state_dict(non_lora_trainables, 'temporal_aggregator'), strict=False)
                
                # handle the special case for the mplug-owl3
                if 'mplug' in model_name.lower():
                    print('loading additional param for the mplug.')
                    additional_params_1 = filter_the_state_dict(non_lora_trainables, 'self_attn.v_kv_proj')
                    model.load_state_dict(additional_params_1, strict=False)
                    additional_params_2 = filter_the_state_dict(non_lora_trainables, 'self_attn.gate_proj')
                    model.load_state_dict(additional_params_2, strict=False)                
                # ipdb.set_trace() # check the loading of the mplug-owl3 

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            # import ipdb
            # ipdb.set_trace() # check before merge
            print('Merging LoRA weights...')
            # model = model.merge_and_unload()
            print('Model is loaded...')
            
            model.initialize_vision_tokenizer(lora_cfg_pretrained, tokenizer=tokenizer)
            # ipdb.set_trace() # check the loading of the tokenizer, the size of the tokenizer
            
        elif 'adaptor' in model_name.lower() and model_base is not None: # for the case we only train the adaptor
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # set the tokenizer details for the data preparation
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version.startswith("Qwen2Tokenizer"):
        # ipdb.set_trace()
        tokenizer.pad_token = '<|endoftext|>'
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else: # for other version
        # TODO: could be a issue here we use the padding tokens of the original tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # # handle the init of the special tokens
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if model_args.video_tower is not None:
        # copy some data size info into the model config
        # model_args.num_frames = data_args.num_frames
        model_args.image_size = data_args.image_size

        # ipdb.set_trace()
        # convert the video tower to a specified data type
        video_tower = model.get_video_tower()
        # ipdb.set_trace() # check the loading of the model
        if video_tower is not None:
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            video_tower.data_type = torch.bfloat16 if training_args.bf16 else torch.float16

        # prepare the dataset config
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        if model_args.tune_temporal_aggregator:
            for p in model.get_model().temporal_aggregator.parameters():
                p.requires_grad = True
        # ipdb.set_trace() # check the grad again
        # only lora and the temporal aggregator2 should be un frozen
        for p in model.get_model().temporal_aggregator2.parameters():
            p.requires_grad = True        
        for p_n, p in model.named_parameters():
            if 'lora_' in p_n:
                p.requires_grad = True 

    # handle the image processor 
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    return model, tokenizer


def train_vidit_func(attn_implementation=None):
    global local_rank
    model_args, data_args, training_args, model_class_name, data_class_name, training_class_name = parse_argument_classes(sys.argv[1:], return_arg_name=True)
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # prepare the model
    model, tokenizer = prepare_video_model_multiple_input(training_args, model_args, data_args, compute_dtype, attn_implementation,
                                                          model_class_name, data_class_name)

    # make the dataset and the trainer
    data_module = make_video_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = VideoModelTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    # TODO: Handle the resume of the training. 
    # HOWEVER, since the training just ONE epoch.
    # It may not reasonable to resume the training.
    # We should restart the training at the begining.
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        torch.use_deterministic_algorithms(False) # for the 3d pooling layer
        trainer.train()
    
    # save the state dict and the model after training
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable: # this is for step2 training
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_with_state_dict(
            model.state_dict(), special_key=['temporal_aggregator', 'temporal_aggregator2', 'self_attn.v_kv_proj', 'self_attn.gate_proj']
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # this is for step 1 training
        safe_save_model_for_hf_videotrainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train_vidit_func(attn_implementation="flash_attention_2")
