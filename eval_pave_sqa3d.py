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



# define the dataloader
class EvalDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 question_file,    # path to the mapping between the video id and the video path
                 video_folder,
                 feature_folder,
                 image_processor,
                 ):
        
        # load the annotation
        question_file_content = json.load(open(question_file))
    
        # loop over all the questions and answer
        all_qa_pair = []
        for curr_q in question_file_content:
            if os.path.exists(os.path.join(video_folder, curr_q['scene_id'])) and \
                os.path.exists(os.path.join(feature_folder, curr_q['scene_id'], 'video_features.pt')):
                curr_qa_dict = {'question_id': curr_q['question_id'],
                                'scene_id': curr_q['scene_id'],
                                'scene_folder': os.path.join(video_folder, curr_q['scene_id']),
                                'scane_feature_path': os.path.join(feature_folder, curr_q['scene_id'], 'video_features.pt'),
                                'question': curr_q['text'],
                                'answer': curr_q['answers'],
                                }
                all_qa_pair.append(curr_qa_dict)
            
        print('total qa: ', len(question_file_content), ' video exist qa:', len(all_qa_pair))
        self.all_qa_pair = all_qa_pair
        self.image_processor = image_processor

    def __len__(self):
        return len(self.all_qa_pair)

    def __getitem__(self, i):
        all_ele_content = self.all_qa_pair[i]
        question_id = all_ele_content['question_id']
        scene_folder = all_ele_content['scene_folder']
        question = all_ele_content['question']
        answer = all_ele_content['answer']
        scane_feature_path = all_ele_content['scane_feature_path']
        
        # import ipdb
        # ipdb.set_trace() # check the whether the path exist
        # Check if the video exists
        if os.path.exists(scene_folder):
            if "gpt4v" != args.model_path:
                video = load_images_from_folder(scene_folder) # (B, H, W, C) (32, 320, 480, 3)
                # ipdb.set_trace() # check loaded image shape

                # reference the playground\demo\video_demo.py in LLaVA-NeXT
                video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
                video = [video]
                # ipdb.set_trace() # check the video shape # torch.Size([32, 3, 384, 384])
            else:
                raise NotImplementedError
        
        # load the feature and reshape    
        video_feat = torch.load(scane_feature_path, map_location=torch.device('cpu')) #torch.Size([1, 18432, 1024]) B, V*H*W, 3
        B, _ , D = video_feat.shape
        V, H, W = 32, 24, 24
        video_feat = video_feat.view(B, V, H, W, D).squeeze(dim=0) # (B, V, H, W, D) -> (V, H, W, D)
        video_feat = video_feat.permute([3,0,1,2]) # (V, H, W, D) -> (C, T, H, W)
        # ipdb.set_trace() # check the loading
        # video_feat_fps = 1
        # feat_frame_num = video_feat.shape[1]

        # send out the feature
        data_dict = {
            'question_id': question_id,
            'video': video,
            'question': question,
            'answer': answer,
            'video_feat': video_feat,
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
    
    eval_dataset = EvalDataset(args.question_file,    # path to the mapping between the video id and the video path
                 args.video_folder,
                 args.feature_folder,
                 image_processor,
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
    
    ans_file = open(args.pred_save, "w")
    count_i = 0
    for ele in tqdm(eval_dataloader):
        # ipdb.set_trace() # check the question and answer
        question_id = ele["question_id"][0]
        video = ele['video'] 
        video = [video[0].squeeze(dim=0).half().cuda()]
        question = ele['question'][0]
        answer = ele['answer'][0]
        video_feature = ele['video_feat'].half().cuda() # it should be in the shape of (B, C, T, H, W)
        video_feat_fps = 1
        feat_frame_num = video_feature.shape[2]
        
        count_i += 1

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
        
        # stopping_criteria = None
        
        # image_sizes = [(ele.shape[0], ele.shape[1]) for ele in video]
        # ipdb.set_trace() # test the image_sizes

        # ipdb.set_trace() # check the input shape
        try:
            # import ipdb
            # ipdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(input_ids, 
                                            video_feats=video_feature,
                                            video_feat_fps=torch.tensor([video_feat_fps]).cuda(), # meaningless info
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

        ans_id = shortuuid.uuid()
        # dump the result as the format of the video-Chatgpt
        
        # TODO: What kind of store format we need
        # {'image_id': 404464, 'caption': 'black and white photo of a man standing in front of a building'}
        ans_file.write(json.dumps({"question_id": question_id,
                                   "prompt": question,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        
        # save the result
        ans_file.flush()
    ans_file.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest='model_path', type=str, default=None)
    parser.add_argument("--model-base", dest='model_base', type=str, default=None)
    parser.add_argument("--model-arg-name", dest='model_arg_name', type=str, default=None)
    parser.add_argument("--question-file", dest='question_file', type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")    
    # parser.add_argument("--mapping-file", dest='mapping_file', type=str, default="answer.jsonl")    
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--feature-folder", type=str, default="")
    
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--original-fps", type=int, default="24")
    parser.add_argument("--eval-fps", type=int, default="4")
    parser.add_argument("--pred-save", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    
    
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
