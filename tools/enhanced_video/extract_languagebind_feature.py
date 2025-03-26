# This script should be put this file under the Video-LLaVA folder to use
# This script aims to use the languagebind to extract the feature

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ipdb
import argparse
import json
import os

from einops import rearrange, repeat

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import decord
from decord import VideoReader, cpu
import numpy as np

from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
import torchvision

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample



# define the dataloader
class LanguageBindFeatureExtractDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 annotation_path,    # path to the mapping between the video id and the video path
                 data_root,
                 feature_saving_root,
                 reverse=False,
                 not_skip_the_first=True,
                 min_frame_num=None
                 ):
        
        self.feature_saving_root = feature_saving_root
        self.data_root = data_root
        self.not_skip_the_first = not_skip_the_first
        self.min_frame_num = min_frame_num
        
        # annotation
        if annotation_path is not None:
            all_anno = json.load(open(annotation_path))
            # get all the distinct video
            self.all_video = [ele['video'] for ele in all_anno]
            self.all_video = list(set(self.all_video))
            origin_len = len(self.all_video)
            # filter the dataset
            self.all_video = [ele for ele in self.all_video if os.path.exists(os.path.join(self.data_root, ele))]             
            
        else: # if the annotation is None then list all the videos on the folder
            self.all_video = os.listdir(data_root)
            origin_len = len(self.all_video)
        
        print('total unique video:', origin_len, 'video exist:', len(self.all_video))

        # reverse
        if reverse:
            self.all_video.reverse()

        # define the transform: refer videollava\model\multimodal_encoder\languagebind\video\processing_video.py
        self.transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.0),
            ]
        )


    def __len__(self):
        return len(self.all_video)


    def __getitem__(self, i):
        
        # determine the source path
        curr_video_path = os.path.join(self.data_root, self.all_video[i])
        
        # import ipdb
        # ipdb.set_trace() # check the path
        # determine the save path
        if '/' in self.all_video[i] and self.not_skip_the_first:
            saving_file_name = '.'.join(('/'.join(self.all_video[i].split('/')[1:])).split('.')[:-1]) + '.pt'
        else:
            saving_file_name = '.'.join(self.all_video[i].split('.')[:-1]) + '.pt'
        curr_saving_path = os.path.join(self.feature_saving_root, saving_file_name)
        
        # if the feature file exist then skip
        if os.path.exists(curr_saving_path):
            data_dict = {}
            data_dict['video'] = -1
            data_dict['video_file_path'] = curr_video_path    
            data_dict['feature_save_path'] = curr_saving_path
            data_dict['feature_saving_root'] = '/'.join(curr_saving_path.split('/')[:-1])
            # ipdb.set_trace()
            return data_dict        
        
        # load the video using decord
        try:
            decord.bridge.set_bridge('torch')
            decord_vr = VideoReader(curr_video_path, ctx=cpu(0))
            duration = len(decord_vr)
            
            # sample the frame using 2 fps 
            video_time = duration / decord_vr.get_avg_fps()
            target_frame_number = video_time * 2
            
            if self.min_frame_num is not None and target_frame_number < self.min_frame_num:
                # import ipdb
                # ipdb.set_trace()
                target_frame_number = self.min_frame_num
            
            # map the number of frame number to something divideable by 8
            exact_chunk = target_frame_number // 8
            if exact_chunk <= 0:
                exact_chunk = 1
            total_frame = int(exact_chunk * 8)

            # extract the frame 
            frame_id_list = np.linspace(0, duration-1, total_frame, dtype=int)
            video_data = decord_vr.get_batch(frame_id_list)
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W) # torch.Size([3, 8, 360, 640])
        except:

            print('using the torch vision to load the files:', curr_video_path)
            # load the full video 
            vframes, _, vinfo = torchvision.io.read_video(filename=curr_video_path, pts_unit="sec", output_format="THWC")

            duration = len(vframes)
            # import ipdb
            # ipdb.set_trace()
            if 'video_fps' in vinfo:
                fps = vinfo["video_fps"]
            else:
                fps = 24
            video_time = duration / fps
            # sample the frame using 2 fps 
            target_frame_number = video_time * 2
            
            # map the number of frame number to something divideable by 8
            exact_chunk = target_frame_number // 8
            if exact_chunk <= 0:
                exact_chunk = 1            
            total_frame = int(exact_chunk * 8)
            
            # extract the frame 
            frame_id_list = np.linspace(0, duration-1, total_frame, dtype=int).tolist()
            video_data = vframes[frame_id_list]
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W) # torch.Size([3, 8, 360, 640])
            
        # call the proprocessing
        image_features = self.transform(video_data) # torch.Size([3, 8, 224, 224]) / torch.Size([3, 8, 224, 224])
        video_feat_split_sizes = [8] * int(exact_chunk)
        
        splited_video_feat = torch.split(image_features, video_feat_split_sizes, dim=1)
        video_tensor = torch.stack(splited_video_feat) # torch.Size([1, 3, 8, 224, 224]) # checked

        # ipdb.set_trace() # check the 'feature_saving_root'
        data_dict = {
            'video': video_tensor,
            'video_file_path': curr_video_path,
            'feature_save_path': curr_saving_path,
            'feature_saving_root': '/'.join(curr_saving_path.split('/')[:-1]) 
        }
        
        # ipdb.set_trace()
        return data_dict

def main(args):
    disable_torch_init()
    # define the dataset
    extract_dataset = LanguageBindFeatureExtractDataset(annotation_path=args.annotation_path,    # path to the mapping between the video id and the video path
                                                        data_root=args.data_root,
                                                        feature_saving_root=args.feature_saving_root,
                                                        reverse=args.reverse_dataset,
                                                        not_skip_the_first=args.not_skip_the_first,
                                                        min_frame_num=args.min_frame_num)    
    
    # define the dataloader
    extract_dataloader = DataLoader(
                extract_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=None,)
    
    # load the model
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    # video_processor = processor['video']

    # defind the downsampling
    m = nn.AdaptiveAvgPool2d((2, 2)) # NCHW
    
    for i, sample in enumerate(extract_dataloader):
        
        # check the file is exist or not 
        curr_feature = sample['video'][0] # torch.Size([1, 40, 3, 8, 224, 224]) -> torch.Size([40, 3, 8, 224, 224])
        curr_saving_path = sample['feature_save_path'][0]
        curr_saving_root = sample['feature_saving_root'][0]
        
        # check the feature is exist or not
        if len(curr_feature.shape) == 0:
            assert curr_feature == -1
            assert os.path.exists(curr_saving_path)
            print(curr_saving_path, ' exist, skipped.')
            continue
        
        # check the path is exist or not, make the dir
        # ipdb.set_trace() # check the curr_saving_root
        if not os.path.exists(curr_saving_root):
            os.makedirs(curr_saving_root)
        
        # ipdb.set_trace() # check the tensor dimension
    
        # prepare the device and type
        tensor = curr_feature.to(model.device, dtype=torch.float16)

        # call the feature extraction
        feature = model.get_model().get_video_tower()(tensor) # torch.Size([1, 8, 257, 1024]) / torch.Size([2, 8, 257, 1024])
        
        # do some downsampling
        # in videollava\model\multimodal_encoder\languagebind\video\modeling_video.py: line 61 we verify the first token is the a special tokens
        global_feat = feature[:, :, 0]
        spatial_feat = feature[:, :, 1:]
        # reshape te sptial
        B, T, _ , D = spatial_feat.shape
        H, W = 16, 16
        spatial_feat = spatial_feat.view(B, T, H, W, D) # NCHW
        # do the adptive pooling on the sptial
        spatial_feat = rearrange(spatial_feat, 'b t h w c -> (b t) c h w')
        spatial_feat = m(spatial_feat) # torch.Size([16, 1024, 2, 2])
        spatial_feat = rearrange(spatial_feat, '(b t) c h w -> b t (h w) c', b=B)
        downsampled_feat = torch.cat([global_feat.unsqueeze(dim=2), spatial_feat], dim=2)
        # ipdb.set_trace() # check the feature
        downsampled_feat = rearrange(downsampled_feat, ' b t s c -> (b t) s c')
        
        torch.save(downsampled_feat, curr_saving_path)
        
        # ipdb.set_trace() # check the feature
        print('total: ', len(extract_dataset), ' curr:', i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract image feature from image backebon')
    parser.add_argument('--data-root', dest='data_root', type=str, default=None)
    parser.add_argument('--feature-saving-root', dest='feature_saving_root', type=str, default=None)     
    parser.add_argument('--annotation-path', dest='annotation_path', type=str, default=None)   
    parser.add_argument('--reverse-dataset', dest='reverse_dataset', action='store_true')  
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=2) 
    parser.add_argument('--not-skip-the-first', dest='not_skip_the_first', action='store_false')  
    parser.add_argument('--min-frame-num', dest='min_frame_num', type=int, default=None)  
    
    # parser.add_argument('--dataset-filter-key', dest='dataset_filter_key', type=str, default=None)
    args = parser.parse_args()
    main(args)
    
## sample usage
# python extract_languagebind_feat.py \
# --data-root /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1 \
# --feature-saving-root /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1/languagebind_feat \
# --annotation-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_oe_v0_1_qa_processed.json \
