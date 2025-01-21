# This script aims to inference the pave and store the prediction into special format which the evaluator can read


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


def load_video(video_path, args):
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
    sample_frame_num = args.for_get_frames_num
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_frame_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    # spare_frames = vr.get_batch(frame_idx).numpy()

    spare_frames = vr.get_batch(frame_idx)
    if isinstance(spare_frames, decord.ndarray.NDArray):
        spare_frames = spare_frames.asnumpy()
    else:
        spare_frames = spare_frames.numpy()    
    
    # ipdb.set_trace()
    # import pdb;pdb.set_trace()

    return spare_frames,frame_time,video_time

# define the dataloader
class EvalDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 annotation_file,    # path to the mapping between the video id and the video path
                 video_folder,
                 feature_folder,
                 image_processor,
                 feat_type,
                 ):
        
        # load the annotation
        annotations = json.load(open(annotation_file))
        self.feat_type = feat_type
        # all_dialogs = annotations['dialogs']
        
        
        # load all the question and answer pair
        all_qa_pair = []
        for ele in annotations:
            vid = ele['video_id']
            curr_video_name = vid + '.mp4'
            curr_feat_name = vid + '.pt'
            video_subfolder = 'MUSIC-AVQA-videos-Real' if vid[0].isdigit() else 'MUCIS-AVQA-videos-Synthetic'
            video_path = os.path.join(video_folder, video_subfolder, curr_video_name)
            if self.feat_type == 'languagebind':
                feat_subfolder = 'MUSIC-AVQA-videos-Real_audio_languagebind_feat' if vid[0].isdigit() else 'MUCIS-AVQA-videos-Synthetic_audio_languagebind_feat'
            elif self.feat_type == 'imagebind':
                feat_subfolder = 'MUSIC-AVQA-videos-Real_audio_imagebind_feat' if vid[0].isdigit() else 'MUCIS-AVQA-videos-Synthetic_audio_imagebind_feat'                
            else:
                feat_subfolder = 'MUSIC-AVQA-videos-Real_audio_feat' if vid[0].isdigit() else 'MUCIS-AVQA-videos-Synthetic_audio_feat'
            feat_path = os.path.join(feature_folder, feat_subfolder, curr_feat_name)
            # print('feat_path:', feat_path)
            
            # refine the question
            curr_qa_dict = {'video_path': video_path,
                            'video_id': vid,
                            'question_id': ele['question_id'],
                            'question': ele['question_content'],
                            'answer': ele['anser'],
                            'feat_path': feat_path,
                            }
            all_qa_pair.append(curr_qa_dict)
            
        
        self.all_qa_pair = all_qa_pair
        self.image_processor = image_processor

    def __len__(self):
        return len(self.all_qa_pair)

    def __getitem__(self, i):
        all_ele_content = self.all_qa_pair[i]
        curr_vid = all_ele_content['video_id']
        curr_question = all_ele_content['question']
        curr_answer = all_ele_content['answer']
        curr_video_path = all_ele_content['video_path']
        curr_feat_path = all_ele_content['feat_path']
        
        # import ipdb
        # ipdb.set_trace() # check the whether the path exist
        # Check if the video exists
        if os.path.exists(curr_video_path):
            if "gpt4v" != args.model_path:
                video,frame_time,video_time = load_video(curr_video_path, args)
                # ipdb.set_trace() # check model.config.image_aspect_ratio (checked)

                # reference the playground\demo\video_demo.py in LLaVA-NeXT
                video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
                video = [video]
                # ipdb.set_trace() # check the video shape # torch.Size([32, 3, 384, 384])
            else:
                raise NotImplementedError

        # load the feature
        if os.path.exists(curr_feat_path):
            audio_feature = torch.load(curr_feat_path)
        
        # some other config for the feature vector
        if audio_feature.requires_grad:
            audio_feature.requires_grad = False

        data_dict = {
            'vid': curr_vid,
            'video': video,
            'audio_feature': audio_feature,
            'curr_feat_path': curr_feat_path,
            'video_file_path': curr_video_path,
            'question': curr_question,
            'answer':curr_answer,
        }
        
        # ipdb.set_trace()
        return data_dict
    
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

    if 'pave' in model_name.lower():
        # ipdb.set_trace()
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. \
                          If you are loading a LoRA model, please provide the `model_base` argument. \
                          Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        
        if 'lora' in model_name.lower() and model_base is not None:
            from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM

            base_model_cfg = PAVEQwen2Config.from_pretrained(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = PAVEQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=base_model_cfg, **kwargs)
            lora_cfg_pretrained = PAVEQwen2Config.from_pretrained(model_path)
            
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

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    # ipdb.set_trace()
    return tokenizer, model, image_processor, context_len



def eval_model(args):
    ####################### load the downloaded checkpoints ####################################
    # ipdb.set_trace()
    # model_name = "llava_qwen"
    # device = "cuda"
    # device_map = "auto"
    # warnings.filterwarnings("ignore")
    # tokenizer, model, image_processor, max_length = load_pretrained_model(args.pretrained_path, None, model_name, device_map=device_map, )  # Add any other thing you want to pass in llava_model_args
    # model.eval()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, max_length = load_trained_model_for_eval(args.model_path, args.model_base, model_name, model_arg_name=args.model_arg_name)
    model.to('cuda')
    
    ### adding back the config 
    model.config.mm_newline_position = args.mm_newline_position
    model.config.mm_spatial_pool_mode = args.mm_spatial_pool_mode
    
    eval_dataset = EvalDataset(args.annotation_file,    # path to the mapping between the video id and the video path
                 args.video_folder,
                 args.feature_folder,
                 image_processor,
                 args.feat_type,
    )

    # define the dataloader
    eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=None,)


    # from_qid_to_question = {ele['question_id']:ele for ele in question_annotations}
    # from_qid_to_answer = {ele['question_id']:ele for ele in answer_annotations}
    
    # # build the mapping from video id to video path
    # # list all the video
    # all_videos = os.listdir(args.video_folder)
    # # build the mapping
    # from_vid_to_video_name = {ele.split('.')[0][2:] : ele for ele in all_videos}
    
    all_prediction = []
    count_i = 0
    for ele in tqdm(eval_dataloader):
        count_i += 1
        vid = ele['vid'][0]
        video = ele['video'] 
        video = [video[0].squeeze(dim=0).half().cuda()]
        question = ele['question'][0]
        answer = ele['answer'][0]
        if len(ele['audio_feature'].shape) == 3: # gama feature, imagebind feature
            audio_feature = ele['audio_feature'].permute([0, 2, 1]).unsqueeze(dim=-1).unsqueeze(dim=-1).half().cuda() # (B, T, C) -> (B, C, T, H, W)
        elif len(ele['audio_feature'].shape) == 4: # languagebind feature (B, 32, S, C)
            feat_dim = ele['audio_feature'].shape[-1]
            temp_bs = ele['audio_feature'].shape[0]
            assert temp_bs == 1
            audio_feature = ele['audio_feature'].view([temp_bs, -1, feat_dim]).permute([0, 2, 1]).unsqueeze(dim=-1).unsqueeze(dim=-1).half().cuda() # (B, T, C) -> (B, C, T, H, W)
        else:
            raise NotImplementedError
        
        feat_frame_num = audio_feature.shape[2]

        # ipdb.set_trace() # audio feature shape, feat_frame_num
        ### prepare the question
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
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
        # cur_prompt = curr_question
        
        # stopping_criteria = None
        
        # image_sizes = [(ele.shape[0], ele.shape[1]) for ele in video]
        # ipdb.set_trace() # test the image_sizes

        # ipdb.set_trace() # check the input shape
        try:
            # import ipdb
            # ipdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(input_ids, 
                                            video_feats=audio_feature,
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
        # ipdb.set_trace() # check prediction

        # ans_id = shortuuid.uuid()
        # dump the result as the format of the video-Chatgpt
        
        # TODO: What kind of store format we need
        # {'image_id': 404464, 'caption': 'black and white photo of a man standing in front of a building'}
        all_prediction.append({'image_id': vid,
                                'pred': outputs,
                                'question': question,
                                'answer': answer,
                                })         

    # save the result
    file = open(args.pred_save, 'w')
    file.write(json.dumps(all_prediction))
    file.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest='model_path', type=str, default=None)
    parser.add_argument("--model-base", dest='model_base', type=str, default=None)
    parser.add_argument("--model-arg-name", dest='model_arg_name', type=str, default=None)
    parser.add_argument("--annotation-file", dest='annotation_file', type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")    
    # parser.add_argument("--mapping-file", dest='mapping_file', type=str, default="answer.jsonl")    
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--feature-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--original-fps", type=int, default="24")
    parser.add_argument("--eval-fps", type=int, default="4")
    parser.add_argument("--pred-save", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    parser.add_argument("--feat-type", dest='feat_type', type=str, default='gama')
    
    
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--add_time_instruction", type=str, default=False)
    parser.add_argument("--for_get_frames_num", type=int, default=32) # for llava onevision use 32 frames
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)    
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")    
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    args = parser.parse_args()

    eval_model(args)
