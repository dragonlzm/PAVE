import os
import copy
from dataclasses import dataclass, field
import json
import time
from typing import  Any, Dict, Optional, Sequence, List, Tuple, Union
import ipdb
from PIL import Image
import numpy as np
import math
from copy import deepcopy

import decord
from decord import VideoReader, cpu

from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers
import torchvision

from ..constants import IGNORE_INDEX, VID_EXTENSIONS, IMG_EXTENSIONS
from ..utils.train_utils import DataArguments, rank0_print
from .image_dataset import preprocess_multimodal, preprocess, preprocess_qwen, preprocess_video_multimodal
from .video_dataset import read_video, temporal_random_crop, get_transforms_video, fps_base_temporal_sampling, frame_base_temporal_sampling, process_video_with_decord
from .dataset_utils import load_data


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_vision=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


def load_video_vae_feature(video_feat_file_path, sources, original_fps, training_feat_fps, min_frame_num=None):
    # video_feat_file_path = os.path.join(video_feat_file_path)
    try:
        video_feat = torch.load(video_feat_file_path).squeeze(dim=0)  # torch.Size([1, 4, 20, 28, 28]) ->  torch.Size([4, 20, 28, 28]) (C, T, H, W)
    except:
        print('video_feat_file_path:', video_feat_file_path, 'data info:', sources)
        torch.save({'video_feat_file_path:': video_feat_file_path, 'data info:': sources}, 'datat_error_' + str(time.time) + '.pt')
        raise NotImplementedError
    # video_fps = self.original_feat_fps  # default fps
    # further downsample the feature if the feature is too long in T axis
    if training_feat_fps != original_fps:
        video_feat, frame_num, final_fps = fps_base_temporal_sampling(video_feat.permute([1, 0, 2, 3]), 
                                                                      original_fps, 
                                                                      target_fps=training_feat_fps, 
                                                                      video_feat_file_path=video_feat_file_path,
                                                                      min_frame_num=min_frame_num) # (C, T, H, W) -> (T, C, H, W) -> (C, T, H, W)
        video_feat = video_feat.permute([1, 0, 2, 3])
        video_feat_fps = final_fps
    else:
        video_feat_fps = original_fps
    
    return video_feat, video_feat_fps


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



class LazySupervisedVideoDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self, 
                 anno_path: str,            # path to the instruction annotation json file
                 fast_path_mapping_path: str,    # path to the mapping between the video id and the video feature path
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedVideoDataset, self).__init__()
        # handle the hyper
        self.tokenizer = tokenizer
        self.data_root = data_args.data_root
        self.data_args = data_args
        self.prepare_qid = data_args.prepare_qid
        self.data_sample_ratio = data_args.data_sample_ratio
        
        # handle the video features
        self.use_fast = data_args.use_fast
        self.use_fast_feat = data_args.use_fast_feat
        
        # handle for the image feature
        self.use_slow = data_args.use_slow
        self.use_slow_feat = data_args.use_slow_feat
        slow_path_mapping_path = data_args.slow_path_mapping_path
        self.slow_path_data_root = data_args.slow_path_data_root
        
        # handle the fast feature type and the tokens
        self.fast_feat_type = data_args.fast_feat_type
        self.exclude_languagebind_cls_token = data_args.exclude_languagebind_cls_token
        
        
        # assertion about the video path
        assert (self.data_args.use_fast == True and self.data_args.use_fast_feat == False) or \
               (self.data_args.use_fast == False and self.data_args.use_fast_feat == True) or \
               (self.data_args.use_fast == False and self.data_args.use_fast_feat == False)
        assert (self.data_args.use_slow == True and self.data_args.use_slow_feat == False) or \
               (self.data_args.use_slow == False and self.data_args.use_slow_feat == True) or \
               (self.data_args.use_slow == False and self.data_args.use_slow_feat == False)
        
        # process the self.data_sample_ratio
        if self.data_sample_ratio is not None:
            self.data_sample_ratio = self.data_sample_ratio.split(',')
            self.data_sample_ratio = [float(ele) for ele in self.data_sample_ratio]
            assert len(self.data_sample_ratio) == len(slow_path_mapping_path)
            # ipdb.set_trace() #check the data_sample_ratio
        
        prefilter_video_ids = ['009151_009200/1057949419', '077251_077300/14911927', # for step1 training
                               '00013654', # for step2 sharegptvideo traning
                               ] 
        
        # handle the special case for multiple dataset
        if isinstance(anno_path, list):
            assert isinstance(fast_path_mapping_path, list)
            assert isinstance(self.data_root, list)
            
            if self.use_slow or self.use_slow_feat: # use raw video frames
                assert isinstance(slow_path_mapping_path, list)
                assert isinstance(self.slow_path_data_root, list)
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root) == len(slow_path_mapping_path) == len(self.slow_path_data_root)

                # load annotation
                all_filtered_anno = []
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root, curr_slow_path_mapping_path, curr_slow_path_data_root) in \
                    enumerate(zip(anno_path, fast_path_mapping_path, self.data_root, slow_path_mapping_path, self.slow_path_data_root)):
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root, 
                                                        prefilter_video_ids=prefilter_video_ids,
                                                        slow_path_mapping_path=curr_slow_path_mapping_path,
                                                        slow_path_data_root=curr_slow_path_data_root)
                    if self.data_sample_ratio is not None:
                        # ipdb.set_trace()
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(curr_anno_path, 'sample ratio:', curr_ratio, 'remaining len:', len(curr_anno))
                    all_filtered_anno += curr_anno    
            else:
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root)
            
                # load annotation
                all_filtered_anno = []
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root) in enumerate(zip(anno_path, fast_path_mapping_path, self.data_root)):
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root, 
                                                        prefilter_video_ids=prefilter_video_ids)
                    if self.data_sample_ratio is not None:
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(curr_anno_path, 'sample ratio:', curr_ratio, 'remaining len:', len(curr_anno))
                    all_filtered_anno += curr_anno
        else:
            all_filtered_anno = load_annotation_and_filter(anno_path, fast_path_mapping_path, self.data_root, 
                                                           prefilter_video_ids=prefilter_video_ids,
                                                           slow_path_mapping_path=slow_path_mapping_path,
                                                           slow_path_data_root=self.slow_path_data_root)

        # ipdb.set_trace() # check the loaded list_data_dict
        self.list_data_dict = all_filtered_anno

        # set the transform
        self.transforms = {
            # "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(self.data_args.transform_name, self.data_args.image_size),
        }
        
        # handle the additional information of the feature loading and downsampling
        if self.use_fast_feat:
            self.original_feat_fps = self.data_args.original_feat_fps
            self.training_feat_fps = self.data_args.training_feat_fps

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'video' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    # @property
    # def modality_lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
    #         cur_len = cur_len if 'video' in sample else -cur_len
    #         length_list.append(cur_len)
    #     return length_list

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i):
        ## Note about the sampling
        ## The image loading         : if we set 'frames_upbound' > 0 and 'force_sample' == True, then the frame we extract will be exactly 32
        ## The image feature loading : will be 32 frame feature and do not need any additional params
        ## The video frame loading   : we set the frames_upbound = 0 thus by default the len of the feature is unbounded, 
        ##                             the lower bound of the frame is set to 32 (send the 'min_frame_num=32' to process_video_with_decord), we have at least 32 frames, [32, +infinity]
        ##                             We set 'video_fps' to 1, which we sample one frame each second.
        ## The video feature loading : we set the 'min_frame_num' in function fps_base_temporal_sampling to 32, [32, + infinity]
        
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # ipdb.set_trace()
        if 'video' in sources[0]:
            # load feature
            # ipdb.set_trace() # check the video loading
            if self.use_fast: # This is for a special test which directly load the video frames as the video feature
                # TODO: different fast backbone need different loading strategy, we need to implement this part later
                raise NotImplementedError
                # try:
                #     video_file_path = self.list_data_dict[i]['feat_path']
                #     # TODO: change this temporate change
                #     # ipdb.set_trace() # check the loading
                #     temp_args = deepcopy(self.data_args)
                #     temp_args.video_fps = 1 # select the frame in 4 fps
                #     temp_args.frames_upbound = 0
                #     min_frame_num = 32
                    
                #     # load the video and select the frames
                #     video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file_path, temp_args, min_frame_num=min_frame_num) # (653, 720, 1280, 3)
                #     video = torch.tensor(video).permute([0, 3, 1, 2]) # T H W C -> T C H W
                #     # preprocessing of the frames
                #     transform = get_transforms_video(self.data_args.transform_name, self.data_args.image_size) 
                #     # ipdb.set_trace()
                    
                #     if len(video) > 100:
                #         all_res = []
                #         start = 0
                #         end = 1
                #         while start * 100 < len(video):
                #             curr_chunks = video[start*100:end*100]
                #             curr_chunks = transform(curr_chunks)
                #             all_res.append(curr_chunks)
                #             start += 1
                #             end += 1
                #         video = torch.cat(all_res, dim=0)
                #     else:
                #         video = transform(video) # image: torch.Size([32, 3, 224, 224]) T, C, H, W -> C, T, H, W
                #     video_feat = video.permute([1,0,2,3]) # T, C, H, W -> C, T, H, W
                #     # ipdb.set_trace() # check size of the size of the image # expect (C, T, H, W)

                #     # papare other params
                #     video_feat_fps = temp_args.video_fps
                #     feat_frame_num = video_feat.shape[1] # expect (C, T, H, W)
                # except Exception as e:
                #     print(f"Error: {e}")
                #     print(f"Failed to read video file: {video_file_path}")
                #     return self._get_item(i + 1)
            elif self.use_fast_feat: # This is for loading the fast features
                # load the video
                video_feat_file_path = self.list_data_dict[i]['feat_path']
                if self.fast_feat_type == 'video_vae':
                    video_feat, video_feat_fps = load_video_vae_feature(video_feat_file_path, sources, 
                                                                        self.data_args.original_feat_fps, 
                                                                        self.data_args.training_feat_fps,
                                                                        min_frame_num=self.data_args.min_fast_frame_num if hasattr(self.data_args, 'min_fast_frame_num') else None)
                    feat_frame_num = video_feat.shape[1]
                    # video_feat should be in the shape of (C, T, H, W)
                elif self.fast_feat_type == 'languagebind' or self.fast_feat_type == 'languagebind_14x14' or self.fast_feat_type == 'internvideo2' or self.fast_feat_type == 'siglip':
                    # ipdb.set_trace() # check the loading
                    # load the feature 
                    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu')) # torch.Size([280, 5, 1024]) T, C, D / torch.Size([32, 196, 1024]) / torch.Size([280, 4, 1024])
                    # exclude the cls tokens
                    if self.exclude_languagebind_cls_token:
                        video_feat = video_feat[:, 1:,]
                    # reshape
                    S = video_feat.shape[1]
                    assert int(math.sqrt(S)) ** 2 == S # assert is a square
                    W = H = int(math.sqrt(S))
                    video_feat = rearrange(video_feat, 't (h w) c -> c t h w', h = H) # video_feat should be in the shape of (C, T, H, W)
                    video_feat_fps = 2
                    feat_frame_num = video_feat.shape[1]
                elif self.fast_feat_type == 'audio':
                    # load the feature 
                    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu')) #torch.Size([19474, 768]) T, C
                    if len(video_feat.shape) == 3: # special case for the model languagebind feature: torch.Size([32, 593, 1024])
                        # ipdb.set_trace() # check the loading, check the view
                        feat_dim = video_feat.shape[-1]
                        video_feat = video_feat.view(-1, feat_dim) # (T*S, C)
                    # special handle for the grad
                    if video_feat.requires_grad:
                        video_feat.requires_grad = False
                    
                    video_feat = video_feat.permute([1,0]) # (C, T)
                    # unsqueeze dim 
                    video_feat = video_feat.unsqueeze(dim=-1).unsqueeze(dim=-1) # (C, T, 1, 1) (C, T, H, W)
                    # reshape
                    video_feat_fps = 100 / 16
                    feat_frame_num = video_feat.shape[1]
                elif self.fast_feat_type == '3d_feature':
                    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu')) #torch.Size([1, 18432, 1024]) B, V*H*W, 3
                    B, _ , D = video_feat.shape
                    V, H, W = 32, 24, 24
                    video_feat = video_feat.view(B, V, H, W, D).squeeze(dim=0) # (B, V, H, W, D) -> (V, H, W, D)
                    video_feat = video_feat.permute([3,0,1,2]) # (V, H, W, D) -> (C, T, H, W)
                    # ipdb.set_trace() # check the loading
                    video_feat_fps = 1
                    feat_frame_num = video_feat.shape[1]
                else:
                    raise NotImplementedError
                    
            # handle the video frame loading
            # ipdb.set_trace() # check the video loading
            if self.use_slow: 
                video_file_path = self.list_data_dict[i]['video_path']
                if os.path.isdir(video_file_path): # is bunch of image
                    video = load_images_from_folder(video_file_path)
                    processor = self.data_args.image_processor
                    image = processor.preprocess(video, return_tensors="pt")["pixel_values"] # image: torch.Size([32, 3, 384, 384])
                    image = [(image, 100, "video")]
                    # ipdb.set_trace() # check the image loading
                else: # is video
                    try:
                        # load the video
                        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file_path, self.data_args)
                        # preprocess the video frames
                        processor = self.data_args.image_processor
                        image = processor.preprocess(video, return_tensors="pt")["pixel_values"] # image: torch.Size([32, 3, 384, 384])
                        # ipdb.set_trace() # check size of the size of the image, test loading the video
                        
                        # prepare the video frames, the original video frame size, type of modality
                        image = [(image, video[0].size, "video")]
                        # ipdb> video[0].size
                        # 2764800
                        # ipdb> video[0].shape
                        # (720, 1280, 3)
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Failed to read video file: {video_file_path}")
                        return self._get_item(i + 1)
            elif self.use_slow_feat: # load the video feature
                # ipdb.set_trace() # check the loading of the slow
                video_file_path = self.list_data_dict[i]['video_path']
                image = torch.load(video_file_path)
                image = [(image, -1, "video")] # use -1 to mark it as a feature instead of a raw frames

            # ipdb.set_trace() # check the loading of the text and the preprocessing
            # handle the text    
            sources = preprocess_video_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        # tokenize the text (turn it into index), 
        # get training target (by masking out the question and context)
        # replace the special token () with idx -200
        # ipdb.set_trace()
        old_data_dict = preprocess(
            sources,
            self.tokenizer,
            has_vision=('video' in self.list_data_dict[i]),
            for_video=True,
            prepare_qid=self.prepare_qid) # special handle for the video
        if isinstance(i, int):
            data_dict = dict(input_ids=old_data_dict["input_ids"][0],
                             labels=old_data_dict["labels"][0])
            if 'question_ids' in old_data_dict:
                data_dict['question_ids'] = old_data_dict["question_ids"][0]
                data_dict['question_len'] = old_data_dict["question_len"]

        # video exist in the data
        if 'video' in self.list_data_dict[i]:
            if self.use_fast or self.use_fast_feat:
                data_dict['video_feat'] = video_feat
                data_dict['video_feat_fps'] = video_feat_fps  
                data_dict['feat_frame_num'] = feat_frame_num  
            
            # Put the video frame information in
            if self.use_slow or self.use_slow_feat:
                data_dict["image"] = image
            
            # put the meta information in
            data_dict['video_meta'] = self.list_data_dict[i]
            
        elif self.data_args.is_multimodal:
            raise NotImplementedError
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            
        # ipdb.set_trace()
        return data_dict        


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


@dataclass
class DataCollatorForSupervisedVideoDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        # labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]        
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video_feat_fps' in instances[0]:
            batch['video_feat_fps'] = torch.tensor([ele['video_feat_fps'] for ele in instances])
        
        if 'feat_frame_num' in instances[0]:
            batch['feat_frame_nums'] = torch.tensor([ele['feat_frame_num'] for ele in instances])

        if 'video_feat' in instances[0]:
            video_feats = [instance['video_feat'] for instance in instances]
            if all(x is not None and x.shape == video_feats[0].shape for x in video_feats):
                batch['video_feats'] = torch.stack(video_feats)
            else:
                # We do the padding here to accerate the training
                all_lens = [ele['feat_frame_num'] for ele in instances]
                max_len = max(all_lens) 
                C, T, H, W = instances[0]['video_feat'].shape
                padded_tensor = torch.zeros([len(instances), C, max_len, H, W]) # (B, C, T, H, W)
                for i, (v, v_len) in enumerate(zip(video_feats, all_lens)):
                    padded_tensor[i][:, :v_len] = v
                batch['video_feats'] = padded_tensor
        if 'question_ids' in instances[0]:
            question_ids = [instance['question_ids'].squeeze(dim=0) for instance in instances]
            
            question_ids = torch.nn.utils.rnn.pad_sequence(
                question_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            question_ids = question_ids[:, :self.tokenizer.model_max_length]
            batch['question_ids'] = question_ids
            batch['question_lens'] = torch.tensor([instance['question_len'] for instance in instances])

        batch['video_metas'] = [ele['video_meta'] for ele in instances]
        
        ## handle the image frames
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_video_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                        data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedVideoDataset(tokenizer=tokenizer,
                                               anno_path=data_args.annotation_path,
                                               fast_path_mapping_path=data_args.fast_path_mapping_path,
                                               data_args=data_args)
    data_collator = DataCollatorForSupervisedVideoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def load_annotation_and_filter(anno_path, fast_path_mapping_path, data_root, 
                               prefilter_video_ids=None,
                               slow_path_mapping_path=None,
                               slow_path_data_root=None):
    list_data_dict = load_data(anno_path) # It will be a list and each ele in the list is a dictionary
    # load the mapping
    feat_path_mapping = json.load(open(fast_path_mapping_path))
    
    if slow_path_mapping_path is not None:
        video_path_mapping = json.load(open(slow_path_mapping_path))
    else:
        video_path_mapping = None
    
    # filter the id that video not exist
    filtered_anno_list = []
    remaining_video = []
    video_not_exist = []
    # ipdb.set_trace()
    for ele in list_data_dict:    
        # ipdb.set_trace()   # check the loading 
        # decode the video id
        vid = ele['video']
        # handle the spcial case in the LLaVA-video 178K
        if '/' in vid:
            vid = vid.split('/')[-1]        
        # handle the case that for the activitynet video of sharegpt-video
        if 'Scene' in vid: 
            vid = vid[:13]
        # handle the special case for the valley and video-chatgpt dataset       
        if '.' in vid:
            vid = vid.split('.')[0]
        # filter some issue video for video-chatgpt dataset
        if prefilter_video_ids is not None and vid in prefilter_video_ids:
            print(vid, 'in prefilter list, filtered.')
            video_not_exist.append(vid)
            continue        
        if vid in feat_path_mapping:
            feat_file_path = os.path.join(data_root, feat_path_mapping[vid])
            if not os.path.exists(feat_file_path): # filter the not exist video
                video_not_exist.append(vid)
                continue
            # merge the root with the path
            ele['feat_path'] = feat_file_path
            
            if video_path_mapping is not None: # handle the the special case that also consider raw video
                video_file_path = os.path.join(slow_path_data_root, video_path_mapping[vid])
                if not os.path.exists(video_file_path): # filter the not exist video
                    video_not_exist.append(vid)
                    continue
                ele['video_path'] = video_file_path
                
            filtered_anno_list.append(ele)
            remaining_video.append(vid)
        else:
            video_not_exist.append(vid)
    # ipdb.set_trace()
    print('dataset:', anno_path, 
          'total annotation:', len(list_data_dict), 
          'remaining anno:', len(filtered_anno_list), 
          'existing video:', len(set(remaining_video)),
          'video_not_exist:', len(set(video_not_exist)))
    
    return filtered_anno_list
