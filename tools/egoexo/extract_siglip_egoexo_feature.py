# This script aims to extract the image feature of the exo view of the ego-exo dataset using the siglip

# Should put this file under the LLaVA-NeXT folder to run
# use environment that run pave to run this script

import ipdb
import os
import argparse
import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json
from decord import VideoReader, cpu

from transformers import AutoConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


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
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # ipdb.set_trace()
    # import pdb;pdb.set_trace()

    return spare_frames,frame_time,video_time


# define the dataloader
class FeatureExtractDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 annotation_file,    # path to the mapping between the video id and the video path
                 video_folder,
                 feature_folder,
                 image_processor,
                 reverse_dataset,
                 ):

        # load the annotation
        question_file_content = json.load(open(annotation_file))['annotations']
    
        # loop over all the questions and answer
        all_video_info = []
        folder_exist = 0
        for curr_q in question_file_content:
            # refine the path for the downscaled video 
            splited_path = curr_q['video_paths']['ego'].split('/')
            updated_splited_path = splited_path[:-1] + ['downscaled', '448']
            updated_merged_path = '/'.join(updated_splited_path)
            
            curr_video_folder = os.path.join(video_folder, updated_merged_path)
            curr_feature_folder = os.path.join(feature_folder, updated_merged_path)
            # ipdb.set_trace() # check the folder
            # check whether the folder exist
            if not os.path.exists(curr_video_folder):
                continue
            
            folder_exist += 1
            # handle each exo video
            for video_key in curr_q['video_paths']:
                if 'exo' in video_key:
                    curr_video_name = curr_q['video_paths'][video_key].split('/')[-1]
                    curr_feature_name = curr_video_name.split('.')[0] + '.pt'
                    curr_video_path = os.path.join(curr_video_folder, curr_video_name)
                    curr_feature_path = os.path.join(curr_feature_folder, curr_feature_name)
                    
                    # ipdb.set_trace()
                    # ipdb.set_trace() # check the video name
                    if os.path.exists(curr_video_path):
                        curr_qa_dict = {'take_id': curr_q['take_uid'],
                                        'video_path': curr_video_path,
                                        'feature_path': curr_feature_path,
                                        'feature_folder': curr_feature_folder,
                                        }
                        all_video_info.append(curr_qa_dict)
            
        print('total folder: ', len(question_file_content), 'folder exist:', folder_exist, 'total exo video:', len(all_video_info))
        self.all_video_info = all_video_info
        
        if reverse_dataset:
            self.all_video_info.reverse()
        
        self.image_processor = image_processor

    def __len__(self):
        return len(self.all_video_info)

    def __getitem__(self, i):
        all_ele_content = self.all_video_info[i]
        curr_take_id = all_ele_content['take_id']
        curr_video_path = all_ele_content['video_path']
        curr_feature_path = all_ele_content['feature_path']
        curr_feature_folder = all_ele_content['feature_folder']
        
        # if video not exist
        if os.path.exists(curr_feature_path):
            data_dict = {}
            data_dict['video'] = -1
            data_dict['fps'] = -1  
            data_dict['frame_num'] = -1      
            data_dict['video_file_path'] = curr_video_path    
            data_dict['feature_save_path'] = curr_feature_path
            data_dict['feature_saving_root'] = curr_feature_folder
            # ipdb.set_trace()
            return data_dict        
        
        # import ipdb
        # ipdb.set_trace() # check the whether the path exist
        # Check if the video exists
        if os.path.exists(curr_video_path):
            if "gpt4v" != args.model_path:
                video, frame_time, video_time = load_video(curr_video_path, args)
                # ipdb.set_trace() # check model.config.image_aspect_ratio (checked)

                # reference the playground\demo\video_demo.py in LLaVA-NeXT
                video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
                #video = [video]
                # ipdb.set_trace() # check the video shape # torch.Size([32, 3, 384, 384])
            else:
                raise NotImplementedError

        data_dict = {}
        data_dict['images'] = video # # image: torch.Size([32, 3, 384, 384])
        data_dict['fps'] = 0  
        data_dict['frame_num'] = video.shape[0]      
        data_dict['video_file_path'] = curr_video_path    
        data_dict['feature_save_path'] = curr_feature_path
        data_dict['feature_saving_root'] = curr_feature_folder
        # ipdb.set_trace()
        return data_dict
    


def main(args):
    ####################### load the downloaded checkpoints ####################################
    compute_dtype = torch.bfloat16    
    
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass    
    
    
    # init the dataset
    # video_mapping_path = args.video_mapping_path
    extract_dataset = FeatureExtractDataset(
                            annotation_file=args.annotation_file,
                            video_folder=args.video_folder,
                            feature_folder=args.feature_folder,
                            image_processor=image_processor,
                            reverse_dataset=args.reverse_dataset)
    
    # init the dataloader
    extract_dataloader = DataLoader(
                extract_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=None,)    
    
    # m = nn.AdaptiveAvgPool2d((14, 14)) # NCHW
    
    # use for loop to extract and save the feature
    for count_i, data in enumerate(extract_dataloader):
        # ipdb.set_trace()
        curr_video_path = data['video_file_path'][0]

        # define the feature saving path
        # temp = curr_video_path.split('/')
        # # temp[1] = 'video_datasets'
        # feature_save_root = os.path.join(args.feature_saving_root, '/'.join(temp[:-1]))
        # feature_save_name = temp[-1].split('.')[0] + '.pt'
        feature_save_path = data['feature_save_path'][0]
        feature_save_root = data['feature_saving_root'][0]
        
        # if feature exist then skip
        if os.path.exists(feature_save_path):
            #ipdb.set_trace()
            try:
                assert data['video'].item() == -1
            except:
                ipdb.set_trace()
            print(curr_video_path, 'is skipped.')
            continue
        
        # if dir not exist create
        if not os.path.exists(feature_save_root):
            os.makedirs(feature_save_root)
        
        # extract the feature
        images = data['images']
        images.to(model.device, dtype=compute_dtype) # torch.Size([1, 320, 3, 384, 384]) -> torch.Size([320, 3, 384, 384])
        images = images.squeeze(dim=0) # torch.Size([32, 3, 384, 384])

        # ipdb.set_trace() # check the datatype
        # extract the feature
        with torch.no_grad():
            all_feats = model.get_model().get_vision_tower()(images) # torch.Size([32, 729, 1152]) 
        
        stride = 2
        height = width = model.get_model().get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = all_feats.shape
        all_feats = all_feats.view(num_frames, height, width, -1)
        all_feats = all_feats.permute(0, 3, 1, 2).contiguous()
        # all_feats = nn.functional.max_pool2d(all_feats, self.config.mm_spatial_pool_stride)
        if model.get_model().config.mm_spatial_pool_mode == "average":
            all_feats = nn.functional.avg_pool2d(all_feats, stride)
        elif model.get_model().config.mm_spatial_pool_mode == "max":
            all_feats = nn.functional.max_pool2d(all_feats, stride)
        elif model.get_model().config.mm_spatial_pool_mode == "bilinear":
            height, weight = all_feats.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            all_feats = nn.functional.interpolate(all_feats, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {model.get_model().config.mm_spatial_pool_mode}")
        all_feats = all_feats.permute(0, 2, 3, 1)
        all_feats = all_feats.view(num_frames, -1, num_dim)        
        
        all_feats = all_feats.detach().to(dtype=torch.bfloat16).cpu()
        # ipdb.set_trace() # check the dimnesion feature after downsampling
        torch.save(all_feats, feature_save_path)

        print('total:', len(extract_dataloader.dataset), 'curr:', count_i, ' ', curr_video_path, 'is proceeded.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", dest='model_path', type=str, default=None)
    parser.add_argument("--model-base", dest='model_base', type=str, default=None)
    parser.add_argument("--annotation-file", dest='annotation_file', type=str, default="tables/question.jsonl")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--feature-folder", type=str, default="")
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)  
    parser.add_argument('--reverse-dataset', dest='reverse_dataset', action='store_true')  

    
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
    main(args)

# Sample usage
# CUDA_VISIBLE_DEVICES=1 python extract_siglip_egoexo_feature.py \
# --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
# --annotation-file /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/annotations/proficiency_demonstrator_train.json \
# --video-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/ \
# --feature-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/sigclip_feature \
# --for_get_frames_num 32 \
# --mm_spatial_pool_stride 2 \
# --mm_spatial_pool_mode bilinear \
# --mm_newline_position grid \
# --overwrite True \
# --num-workers 8 \
# --reverse-dataset


# CUDA_VISIBLE_DEVICES=0 python extract_siglip_egoexo_feature.py \
# --model-path lmms-lab/llava-onevision-qwen2-7b-ov \
# --annotation-file /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/annotations/proficiency_demonstrator_val.json \
# --video-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/ \
# --feature-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin/sigclip_feature \
# --for_get_frames_num 32 \
# --mm_spatial_pool_stride 2 \
# --mm_spatial_pool_mode bilinear \
# --mm_newline_position grid \
# --overwrite True \
# --num-workers 8
