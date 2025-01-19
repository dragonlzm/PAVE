# This script aims to inference the model LLaVA and store the prediction into special format which the COCOeval can read

# This script aims to eval the LLaVA-OneVision using the ActivityNet dataset
# Should put this file under the LLaVA-NeXT folder to run
# use the following environment:
# module use /depot/schaterj/data/3d/modules
# module load conda-env/lla-py3.8.5 


import argparse
import torch
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import shortuuid
import ipdb
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ipdb
import argparse
import json
import os

from transformers import AutoConfig
import decord
from decord import VideoReader, cpu
import numpy as np
from transformers import AutoConfig, BitsAndBytesConfig, AutoTokenizer
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
import torchvision
from einops import rearrange, repeat
from PIL import Image
from PIL import Image as im 
import requests
import copy
import torch
from decord import VideoReader, cpu
import sys
import warnings

import math
from libs.conversation_lib import conv_templates, SeparatorStyle
from libs.utils.train_utils import MODEL_ARGUMENTS_MAPPING, DATA_ARGUMENTS_MAPPING
from libs.mm_utils import tokenizer_vision_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(video_path, for_get_frames_num):
    # extract 32 video frames
    # if args.for_get_frames_num == 0:
    #     return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    # fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    # frame_time = [i/fps for i in frame_idx]
    # ipdb.set_trace()
    # if len(frame_idx) > args.for_get_frames_num or args.force_sample:
    sample_frame_num = for_get_frames_num
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_frame_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # ipdb.set_trace()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


def load_images_from_folder(folder_path):
    # ipdb.set_trace()
    # try loading the npy files
    npy_file_path = os.path.join(folder_path, 'stacked_images.npy')
    if os.path.exists(npy_file_path):
        images_array = np.load(npy_file_path)
        return images_array
    
    # if the npy files not exist, list all the files
    all_files_under_folder = os.listdir(folder_path)
    
    # filter and sort the file along the temporal
    jpg_files = sorted(
        [f for f in all_files_under_folder if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )    
    
    # load the image 
    images = []
    for filename in jpg_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
        img_array = np.array(img)
        images.append(img_array)
    
    # handle the special case
    if len(images) > 32:
        print(folder_path, len(images))
        images = images[:32]
    elif len(images) < 32:
        gap = 32 - len(images)
        print(folder_path, len(images))
        images = images + [images[-1]] * gap
    
    # stack the image
    if images:
        # Convert list of images to a 4D NumPy array
        images_array = np.stack(images, axis=0)  # Shape: (number_of_images, H, W, C)
        # save the npy file
        # ipdb.set_trace()
        save_file_name = os.path.join(folder_path, "stacked_images.npy")
        print('stacked numpy array is saved to:', save_file_name)
        np.save(save_file_name, images_array)
        return images_array
    else:
        print("No JPG images found in the specified folder.")
        return None


# temporal_aggregator_parameters = [name for name, _ in opt_model.named_parameters() if "temporal_aggregator" in name]
def filter_the_state_dict(state_dict, keyword):
    # filter the state dict using the keyword
    new_state_dict = {key: state_dict[key] for key in state_dict if keyword in key}
    return new_state_dict



def load_trained_model_for_eval(model_path, model_base, model_name, 
                                model_arg_name='default',
                                data_arg_name='default',
                                load_8bit=False, load_4bit=False, 
                                device_map="auto", device="cuda", 
                                use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    if 'vidit' in model_name.lower():
        # ipdb.set_trace()
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. \
                          If you are loading a LoRA model, please provide the `model_base` argument. \
                          Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
        if 'lora' in model_name.lower() and model_base is not None:
            from libs.model.language_model.vidit_qwen2 import ViditQwen2Config, ViditQwen2ForCausalLM

            base_model_cfg = ViditQwen2Config.from_pretrained(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = ViditQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_model_cfg, **kwargs)
            lora_cfg_pretrained = ViditQwen2Config.from_pretrained(model_path)
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
            
            # for key in lora_cfg_pretrained.__dict__:
            #     if not key.startswith('__'):
            #         print(key)
            
            # re-instantiate the Video backbone and the SSM
            # ipdb.set_trace() # check the video module init
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
            model = model.merge_and_unload()
            print('Model is loaded...')
            
            ### handle the loading of the tokenizer
            # lora_cfg_pretrained.pretrain_temporal_aggregator = os.path.join(model_path, 'non_lora_trainables.bin')
            
            model.initialize_vision_tokenizer(lora_cfg_pretrained, tokenizer=tokenizer)
            # ipdb.set_trace() # check the loading of the tokenizer, the size of the tokenizer
            
        elif 'adaptor' in model_name.lower() and model_base is not None: # for the case we only train the adaptor
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    image_processor = None
    # ipdb.set_trace()
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        image_processor = vision_tower.image_processor
    
    ### TODO: handle the special tokens
    if 'mplug' in  model_name.lower():
        print('adding additional token for mplug')
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    # ipdb.set_trace()
    return tokenizer, model, image_processor, context_len




####################### load the downloaded checkpoints ####################################
#the model path for each version
# model_base = 'lmms-lab/llava-onevision-qwen2-7b-ov'
# conv_mode = 'conv_llava_ov_qwen'

# for general video
# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_v5_1_3_lora_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/huggingface/videomme/data/m4qhFFdHTCc.mp4' 
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/videomme/languagebind_feat/m4qhFFdHTCc.pt' 
# task_type = 'video'
# question = 'please describe the video ?'

# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_v5_1_3_lora_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/academic_source/NextQA/0023/6971535794.mp4'
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/languagebind_feat/NextQA/0023/6971535794.pt' 
# task_type = 'video'
# question = 'please describe the video ?'


# for 0.5B model test
model_base = 'lmms-lab/llava-onevision-qwen2-0.5b-ov'
conv_mode = 'conv_llava_ov_qwen'
model_path = '/video_datasets/zhuoming/checkpoints/vidit_v5_1_2_lora_full'
model_arg_name = 'VideoFeatModelArgumentsV5_1_2'
video_path = '/video_datasets/zhuoming/checkpoints/6971535794.mp4'
feat_path = '/video_datasets/zhuoming/checkpoints/6971535794.pt' 
task_type = 'video'
question = 'please describe the video ?'


# for audio
# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_v5_1_3_lora_avsd_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_audio_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/avsd/Charades_vu17_test_480/VC5RZ.mp4' 
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/avsd/Charades_vu17_test_audio_feat/VC5RZ.pt' 
# task_type = 'audiovideo'
# question = 'are there any people in the video ?'    

# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_v5_1_3_lora_avsd_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_audio_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/audio_dataset/1_2_m_0032_3080070070_walking_traffic_sound.mp4' 
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/audio_dataset/testaudio_feat/1_2_m_0032_3080070070_walking_traffic_sound.pt' 
# task_type = 'audiovideo'
# # question = 'please describe the sound you hear from the video'
# question = 'Do you hear the sound of the cat meow'  
# # answer： 'No I do not hear the cat meow.'



# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_v5_1_3_lora_avsd_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_audio_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/audio_dataset/1_2_m_0048_3846475848_man_talking_and_water.mp4' 
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/audio_dataset/testaudio_feat/1_2_m_0048_3846475848_man_talking_and_water.pt' 
# task_type = 'audiovideo'
# question = 'please describe the sound you hear from the video'    
# # answer： "The sound is of water running in the sink and the person's hands moving."

# # for 3dqa
# model_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/vidit_scanqa_v5_1_3_3d_lora_7B'
# model_arg_name = 'VideoFeatModelArgumentsV5_1_3_3d_7B'
# video_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/scannet/posed_images_new/scene0011_00' 
# feat_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/scannet/video_features_new/scene0011_00/video_features.pt' 
# task_type = '3d'
# question = 'What color is the chair in the kitchen?'  




model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, max_length = load_trained_model_for_eval(model_path, model_base, model_name, model_arg_name=model_arg_name)
model.to('cuda')

### adding back the config 
model.config.mm_newline_position = 'grid'
model.config.mm_spatial_pool_mode = 'bilinear'



# load the video 
if not os.path.isdir(video_path): 
    video,frame_time,video_time = load_video(video_path, for_get_frames_num=32)
else: # check if it's a folder for the 3d case
    video = load_images_from_folder(video_path) # (B, H, W, C) (32, 320, 480, 3)

video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
video = [video]  

# load the feature
if task_type == 'audiovideo':
    video_feat = torch.load(feat_path, map_location=torch.device('cpu'))
    video_feat = video_feat.unsqueeze(dim=0).permute([0, 2, 1]).unsqueeze(dim=-1).unsqueeze(dim=-1) # (B, T, C) -> (B, C, T, 1, 1)
    
elif task_type == '3d':
    # load the feature and reshape    
    video_feat = torch.load(feat_path, map_location=torch.device('cpu')) #torch.Size([1, 18432, 1024]) B, V*H*W, 3
    B, _ , D = video_feat.shape
    V, H, W = 32, 24, 24
    video_feat = video_feat.view(B, V, H, W, D).squeeze(dim=0) # (B, V, H, W, D) -> (V, H, W, D)
    video_feat = video_feat.permute([3,0,1,2]).unsqueeze(dim=0) # (V, H, W, D) -> (B, C, T, H, W)
    # ipdb.set_trace() # check the loading
    # video_feat_fps = 1
    # feat_frame_num = video_feat.shape[1]    
else: # for general video
    video_feat = torch.load(feat_path, map_location=torch.device('cpu')) # torch.Size([280, 5, 1024]) T, C, D
    # exclude the cls tokens
    video_feat = video_feat[:, 1:,]
    # reshape
    S = video_feat.shape[1]
    assert int(math.sqrt(S)) ** 2 == S # assert is a square
    W = H = int(math.sqrt(S))
    video_feat = rearrange(video_feat, 't (h w) c -> c t h w', h = H).unsqueeze(dim=0) # video_feat should be in the shape of (B, C, T, H, W)
    # print('fast_feat:', fast_feat.shape)             

# vid = ele['vid'][0]
# video = ele['video'] 
video = [video[0].squeeze(dim=0).half().cuda()]
video_feat = video_feat.half().cuda() # (B, T, C) -> (B, C, T, H, W)
feat_frame_num = video_feat.shape[2]

# ipdb.set_trace() # audio feature shape, feat_frame_num
### prepare the question
qs = question
if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
# '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
# <|im_start|>user\n<image>\nis the athlete wearing trousers<|im_end|>\n
# <|im_start|>assistant\n'
gen_kwargs = {}
if "max_new_tokens" not in gen_kwargs:
    gen_kwargs["max_new_tokens"] = 1024

if "temperature" not in gen_kwargs:
    gen_kwargs["temperature"] = 0

if "do_sample" not in gen_kwargs:
    gen_kwargs["do_sample"] = False

if "top_p" not in gen_kwargs:
    gen_kwargs["top_p"] = None

if "num_beams" not in gen_kwargs:
    gen_kwargs["num_beams"] = 1



input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors="pt").unsqueeze(0).cuda()
# tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
#     151645,    198, 151644,    872,    198,   -200,    198,    285,    279,
#     33780,  12233,  67676, 151645,    198, 151644,  77091,    198]],
# device='cuda:0')

if tokenizer.pad_token_id is None:
    if "qwen" in tokenizer.name_or_path.lower():
        print("Setting pad token to bos token for qwen model.")
        tokenizer.pad_token_id = 151643

attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

gen_kwargs["modalities"] = ["video"]
gen_kwargs["stopping_criteria"] = [stopping_criteria]

# stopping_criteria = None

# image_sizes = [(ele.shape[0], ele.shape[1]) for ele in video]
# ipdb.set_trace() # test the image_sizes

# ipdb.set_trace() # check the input shape
try:
    # import ipdb
    # ipdb.set_trace()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, 
                                    video_feats=video_feat,
                                    video_feat_fps=torch.tensor([1]).cuda(), # meaningless info
                                    feat_frame_nums=torch.tensor([feat_frame_num]).cuda(),
                                    images=video,
                                    image_sizes=[200], # -1 as an indicator of using the slow feature
                                    attention_mask=attention_masks, 
                                    pad_token_id=tokenizer.pad_token_id, 
                                    use_cache=True, 
                                    cache_position=None,
                                    **gen_kwargs)
        # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
except Exception as e:
    raise e


outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print('outputs:', outputs)


