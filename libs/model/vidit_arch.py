#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import math
import ipdb
from einops import rearrange, repeat

from .multimodal_encoder.builder import build_temporal_aggregator, build_video_tower, build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from libs.mm_utils import get_anyres_image_grid_shape, split_list_lengths
from libs.utils.train_utils import rank0_print


def get_weight(weights, keyword):
    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}


class ViditMetaModel:

    def __init__(self, config):
        super(ViditMetaModel, self).__init__(config)

        # The Fast-Path
        if hasattr(config, "mm_video_tower"):
            #build the encoder (MAE + diffusion)
            self.video_tower = build_video_tower(config, delay_load=True)
            #TODO: build the compresser (SSM) We may not need to instantiate it now
            self.temporal_aggregator = build_temporal_aggregator(config)

        # The Slow-Path
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            # self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        #### The Fast-path Init ###
        
        video_tower = model_args.video_tower
        pretrain_temporal_aggregator = model_args.pretrain_temporal_aggregator
        self.config.mm_video_tower = video_tower

        ### init and load the pretrained video tower backbone
        if self.get_video_tower() is None:
            video_tower = build_video_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.video_tower = [video_tower]
            else:
                self.video_tower = video_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                video_tower = self.video_tower[0]
            else:
                video_tower = self.video_tower
            # load the checkpoint (for step 1 and step 2)
            video_tower.load_model()

        ### build the temporal aggregator again
        self.config.temporal_aggregator_type = getattr(model_args, 'temporal_aggregator_type', 'ssm')
        if getattr(self, 'temporal_aggregator', None) is None:
            self.temporal_aggregator = build_temporal_aggregator(model_args)
        else:
            # In case it is frozen by LoRA
            for p in self.temporal_aggregator.parameters():
                p.requires_grad = True

        # load the existing temporal aggregator checkpoint (for step 2 training)
        if pretrain_temporal_aggregator is not None:
            temporal_aggregator_weights = torch.load(pretrain_temporal_aggregator, map_location='cpu')
            # ipdb.set_trace()
            def get_w(weights, keyword, module):
                temp = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                
                # handle the running means
                module_state_dict = module.state_dict()
                # check the loading 
                if len(module_state_dict) >= len(temp):
                    missed_key = [ele for ele in module_state_dict if ele not in temp]
                    
                    # hacky way to get the data type
                    data_type = None
                    for key in temp:
                        data_type = temp[key].dtype
                    
                    for key in missed_key:
                        print('ele:', key, 'is reinitialized.')
                        temp[key] = module_state_dict[key].to(data_type)  
                return temp
                    
            self.temporal_aggregator.load_state_dict(get_w(temporal_aggregator_weights, 'temporal_aggregator', self.temporal_aggregator))

        ### The Slow-path Init. This section should be skipped ### 
        # ipdb.set_trace() # check this part is skipped
        if hasattr(model_args, 'vision_tower') and self.get_vision_tower() is None:
            print('You are reloading/reiniting the vision_tower in function initialize_vision_modules')
            vision_tower = model_args.vision_tower
            mm_vision_select_layer = model_args.mm_vision_select_layer
            mm_vision_select_feature = model_args.mm_vision_select_feature
            pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter     # check whether we train the adapter in the step1 training
            mm_patch_merge_type = model_args.mm_patch_merge_type
            pretrain_vision_modules = model_args.pretrain_vision_modules     # Considering the case we directly load the whole vision module from other model
            # ipdb.set_trace() # check pretrain_vision_modules

            self.config.mm_vision_tower = vision_tower
            self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

            # Load the vision backbone (Image Backbone)
            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)
                # vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
                # for k, v in vision_resampler.config.items():
                #     setattr(self.config, k, v)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                    # self.vision_resampler = [vision_resampler]
                else:
                    self.vision_tower = vision_tower
                    # self.vision_resampler = vision_resampler
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                    # vision_resampler = self.vision_resampler[0]
                else:
                    vision_tower = self.vision_tower
                    # vision_resampler = self.vision_resampler
                # if pretrain_vision_modules is not None: # if we has the pretrain model then further delay the loading
                vision_tower.load_model()

                # In case it is frozen by LoRA
                # for p in self.vision_resampler.parameters():
                #     p.requires_grad = True

            self.config.use_mm_proj = True
            self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
            # self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
            self.config.mm_vision_select_layer = mm_vision_select_layer
            self.config.mm_vision_select_feature = mm_vision_select_feature
            self.config.mm_patch_merge_type = mm_patch_merge_type
            
            if not hasattr(self.config, 'add_faster_video'):
                if hasattr(model_args, 'add_faster_video')  and model_args.add_faster_video:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.faster_token = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            
            if getattr(self, "mm_projector", None) is None:
                self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

                if "unpad" in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

            # load the adaptor
            if pretrain_mm_mlp_adapter is not None:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

                incompatible_keys = self.mm_projector.load_state_dict(get_weight(mm_projector_weights, "mm_projector"))
                print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
                incompatible_keys = self.vision_resampler.load_state_dict(get_weight(mm_projector_weights, "vision_resampler"), strict=False)
                print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            
            # ipdb.set_trace()
            # load the vision backbone, adaptor and the self.image_newline
            if pretrain_vision_modules is not None:
                assert pretrain_mm_mlp_adapter is None, "You give the pretrain_mm_mlp_adapter and pretrain_vision_modules at the same time"
                # load the full model
                whole_vision_weights = torch.load(pretrain_vision_modules, map_location="cpu")
                
                # load the backbone #'model.vision_tower.vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight' =>  vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight
                incompatible_keys = self.vision_tower.load_state_dict({'.'.join(k.split(".")[2:]): v for k, v in whole_vision_weights.items() if "vision_tower" in k})
                print(f"ReLoaded vision_tower weights from {pretrain_vision_modules}. Incompatible keys: {incompatible_keys}")
                
                # load the adaptor
                incompatible_keys = self.mm_projector.load_state_dict(get_weight(whole_vision_weights, "mm_projector"))
                print(f"Loaded mm projector weights from {pretrain_vision_modules}. Incompatible keys: {incompatible_keys}")
                
                # load the newline
                self.image_newline.load_state_dict(whole_vision_weights['model.image_newline'])
                print(f'Loaded image_newline weights from {pretrain_vision_modules}.')
                
            # ipdb.set_trace() # check the loading
        
        ## handle other config
        self.config.mm_newline_position = model_args.mm_newline_position
        self.config.feat_combine_method = model_args.feat_combine_method
        self.config.train_addition_start_end_tokens = model_args.train_addition_start_end_tokens

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class ViditMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()


    def encode_videos(self, 
                      video_feats, 
                      q_text_embeds=None, 
                      video_feat_fps=None, 
                      feat_frame_nums=None,
                      q_text_nums=None,
                      chunk_num=None,
                      slow_feats=None):
        # video_feats         : Tensor [B, C, T, H, W] 
        # video_features : Tensor [B, T, H, W, C]
        # feat_frame_nums     : Tensor [B, ]
        # q_text_embeds  : Tensor [B, Max_squence_len] is the question embedding in the instruction pairs 
        # q_text_nums    : Tensor [B, ] 
        # slow_feats : list of tensor [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
        
        # Using the question text embedding in the diffusion module of the OpenSora
        # we add one extra layer in temporal_aggregator to keep all trainable params in temporal_aggregator
        if hasattr(self.get_model().temporal_aggregator, 'diffusion_mlp') and self.get_model().temporal_aggregator.diffusion_mlp is not None:
            assert self.get_model().get_video_tower().opensora_diffusion is not None , "The MLP defined but the diffusion is not used"
            q_text_embeds = q_text_embeds.to(dtype=self.get_model().temporal_aggregator.diffusion_mlp.layers[0].weight.dtype)
            diffusion_text_embedding = self.get_model().temporal_aggregator.diffusion_mlp(q_text_embeds)
            # ipdb.set_trace() # check the embedding dimension
        else:
            diffusion_text_embedding = None

        # encode video feature
        # video_features, new_frame_num = self.get_model().get_video_tower()(video_feats, 
        #                                                                    diffusion_control_texts=None, # this suppose to send in the question to refine the diffusion model
        #                                                                    fps=video_feat_fps, 
        #                                                                    frame_nums=feat_frame_nums,
        #                                                                    diffusion_control_text_embedding=diffusion_text_embedding, 
        #                                                                    diffusion_control_text_embedding_len=q_text_nums)
        
        video_features = video_feats
        new_frame_num = feat_frame_nums        
        video_features = video_features.permute([0,2,3,4,1]) # [B, C, T, H, W] -> [B, T, H, W, C]
        # return x, new_frame_nums            
        
        # pass the feature to the temporal aggregator
        if hasattr(self.get_model().temporal_aggregator, 'use_slow_as_query') and self.get_model().temporal_aggregator.use_slow_as_query:
            video_features = self.get_model().temporal_aggregator(video_features, new_frame_num, 
                                                                q_text_embeds=q_text_embeds, 
                                                                q_text_nums=q_text_nums,
                                                                chunk_num=chunk_num,
                                                                slow_feats=slow_feats)
        else:
            video_features = self.get_model().temporal_aggregator(video_features, new_frame_num, 
                                                                q_text_embeds=q_text_embeds, 
                                                                q_text_nums=q_text_nums,
                                                                chunk_num=chunk_num)
        
        # handle output feature number for the case with the cross-attn 
        # ipdb.set_trace() # handle the number of tokens here
        # ipdb.set_trace() # check the new_frame_num in the following lines
        if hasattr(self.get_model().temporal_aggregator, 'use_query_tokens') and self.get_model().temporal_aggregator.use_query_tokens: # v2 version
            new_frame_num = torch.tensor([self.get_model().temporal_aggregator.num_query_tokens] * video_features.shape[0]).to(video_features.device)
        if type(self.get_model().temporal_aggregator).__name__ == 'SSMTemporalAggregatorV4': # for v4 version, although this may not neccessary
            new_frame_num = torch.tensor([self.get_model().temporal_aggregator.number_of_query*chunk_num] * video_features.shape[0]).to(video_features.device)
        
        return video_features, new_frame_num

    def encode_images(self, images, return_feat_before_mlp=False):
        image_features_before_mlp = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features_before_mlp = image_features_before_mlp.to(dtype=self.dtype) # update the data type for eval
        image_features = self.get_model().mm_projector(image_features_before_mlp)
        if return_feat_before_mlp:
            return image_features, image_features_before_mlp
        else:
            return image_features

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        # if self.config.add_faster_video:
        #     # import pdb; pdb.set_trace()
        #     # (3584, 832, 14) -> (3584, 64, 13, 14)
        #     image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
        #     #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
        #     image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
        #     # (64, 13, 14, 3584) -> (64, 13*14, 3584)
        #     image_feature = image_feature.flatten(1, 2)
        #     # import pdb; pdb.set_trace()
        #     return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_image_features(self, images, image_sizes, modalities, return_feat_before_mlp=False):
        '''
            This function is for encode the image feature.
            images: list[tensor]: shape of the tensor is torch.Size([32, 3, 384, 384]), len of list = batchsize
            image_sizes: list[int]: represent the H*W*C of the origin video frames
            modalities: list[string]: string should be "video"
        '''
        # ipdb.set_trace() # check the input format
        
        if images is None: # We do not have image as input
            return None
        
        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0) # torch.Size([64, 3, 384, 384])
            split_sizes = [image.shape[0] for image in images_list] # [32, 32]
            # ipdb.set_trace() # check the feature before mlp
            if return_feat_before_mlp:
                encoded_image_features, encoded_image_features_before_mlp = self.encode_images(concat_images, return_feat_before_mlp=return_feat_before_mlp) # torch.Size([64, 729, 896])
                
                encoded_image_features = torch.split(encoded_image_features, split_sizes) 
                encoded_image_features_before_mlp = torch.split(encoded_image_features_before_mlp, split_sizes) 
                image_features = []
                image_features_before_mlp = []
                for idx, (image_feat, image_feat_before_mlp) in enumerate(zip(encoded_image_features, encoded_image_features_before_mlp)): # [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
                    if idx in video_idx_in_batch: # video_idx_in_batch: [0]
                        image_features.append(self.get_2dPool(image_feat)) ## Video call this torch.Size([32, 169, 896])
                        image_features_before_mlp.append(self.get_2dPool(image_feat_before_mlp)) ## Video call this torch.Size([32, 169, 896])
                    else:
                        image_features.append(image_feat)
                        image_features_before_mlp.append(image_feat_before_mlp)
                
                return image_features, video_idx_in_batch, image_features_before_mlp
            else:
                encoded_image_features = self.encode_images(concat_images, return_feat_before_mlp=return_feat_before_mlp) # torch.Size([64, 729, 896])
                # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
                # ipdb.set_trace() # check the encoded_image_features
                # This is a list, each element is [num_images, patch * patch, dim]
                # rank_print(f"Concat images : {concat_images.shape}")
                ### For image
                # encoded_image_features: torch.Size([512, 729, 896]) 
                # 729 = 27 * 27 => the feature of one patches
                # 512 = 32 * 16 => total_frames * patches_for_each_frame
                ### For video
                # torch.Size([32, 729, 896])
                encoded_image_features = torch.split(encoded_image_features, split_sizes) 
                image_features = []
                for idx, image_feat in enumerate(encoded_image_features): # [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
                    if idx in video_idx_in_batch: # video_idx_in_batch: [0]
                        image_features.append(self.get_2dPool(image_feat)) ## Video call this torch.Size([32, 169, 896])
                    else:
                        image_features.append(image_feat)
                
                return image_features, video_idx_in_batch
        else:
            raise NotImplementedError
            # image_features = self.encode_images(images)

        

    def post_processing_of_image_feature(self, image_features, video_idx_in_batch):
        '''
            This function is for some post-processing of the image feature, 
            like flatten and adding special tokens
        '''
        # ipdb.set_trace() # check the hyper
        # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
        # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
        # image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat") # for video and image: 'spatial_unpad'
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square") # for video and image: 'anyres_max_9'
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token") # for image:'one_token', for video: 'no_token'

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"): # INTO HERE
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)
                # rank0_print("At least we are reaching here")
                # import pdb; pdb.set_trace()
                if image_idx in video_idx_in_batch:  # video operations
                    # rank0_print("Video")
                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        # if self.config.add_faster_video:
                        #     faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                        #     # Add a token for each frame
                        #     concat_slow_fater_token = []
                        #     # import pdb; pdb.set_trace()
                        #     for _ in range(image_feature.shape[0]):
                        #         if _ % self.config.faster_token_stride == 0:
                        #             concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                        #         else:
                        #             concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                        #     # import pdb; pdb.set_trace()
                        #     image_feature = torch.cat(concat_slow_fater_token)
                    
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature) # torch.Size([32, 169, 896]) -> torch.Size([32, 170, 896])

                        new_image_features.append(image_feature.flatten(0, 1))
                        
                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1) # torch.Size([32, 169, 896])
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0) # torch.Size([6273, 896])
                        new_image_features.append(image_feature)      
                    elif mm_newline_position == "no_token": 
                        new_image_features.append(image_feature.flatten(0, 1)) # torch.Size([5408, 896])
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                else:  # single image operations # multi patches and multi images operations
                    raise NotImplementedError
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        
        return image_features


    def prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                video_feats,
                video_feat_fps=None,
                feat_frame_nums=None,
                question_ids=None,
                question_lens=None,
                # for the image frames
                images=None,
                image_sizes=None,
                modalities=None,
                video_metas=None,
            ):
        # input_ids       : [B, text_len]  torch.Size([4, 80])
        # labels          : [B, text_len]  torch.Size([4, 80])
        # video_feats     : [B, C, T, H, W] torch.Size([4, 3, 340, 256, 256]) / If is feature torch.Size([4, 4, 27, 28, 28])
        # video_feat_fps  : [B, ] torch.Size([4])
        # feat_frame_nums : [B, ] torch.Size([4])
        # question_ids    : [B, Max_seq_len] torch.Size([4])
        # question_lens   : [B, ] torch.Size([4])
        # images          : if it is feature it should be list of [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
        # image_sizes
        # modalities      : Should be a list, len(list) = batch_size, each element in the list is 'video'
        # ipdb.set_trace() # check video_feats should be None
        
        video_tower = self.get_video_tower()
        vision_tower = self.get_vision_tower()
        
        if (video_tower is None and vision_tower is None) or (video_feats is None and images is None) or input_ids.shape[1] == 1: # this could be used for the inference
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # embed the question id using our LLM
        if question_ids is not None:
            question_embeds = self.get_model().embed_tokens(question_ids).detach()
        else:
            question_embeds = None
        
        # figure out the chunk size
        if images is not None:
            chunk_num = images[0].shape[0]
        else:
            chunk_num = None
            
        # special control for the slow feature
        if hasattr(self.get_model(), 'temporal_aggregator') and getattr(self.get_model().temporal_aggregator, 'use_slow_feat_before_mlp', False): 
            assert images is not None and (-1 not in image_sizes) # if using the slof-_feature_before_MLP current implementation only support the training with the raw video
            self.use_slow_feat_before_mlp = True
        else:
            self.use_slow_feat_before_mlp = False

        # get the image feature (The slow feature)
        # ipdb.set_trace() # check feature before mlp
        if images is not None and (-1 not in image_sizes):
            if self.use_slow_feat_before_mlp:
                image_features, video_idx_in_batch, image_features_before_mlp = self.prepare_image_features(images, image_sizes, modalities, return_feat_before_mlp=self.use_slow_feat_before_mlp) # [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
            else:
                image_features, video_idx_in_batch = self.prepare_image_features(images, image_sizes, modalities) # [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
        else: # the image feature is loaded
            # ipdb.set_trace() # test the handle of the image feature
            assert sum(image_sizes) == -len(image_sizes) # assert all ele in this video is -1
            image_features = images
            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)
        
        # import ipdb
        # ipdb.set_trace()
        # get the video feature (The fast feature)
        if video_feats is None:
            video_features, new_frame_num = None, None
        else:
            video_features, new_frame_num = self.encode_videos(video_feats, 
                                                            q_text_embeds=question_embeds,
                                                            video_feat_fps=video_feat_fps, 
                                                            feat_frame_nums=feat_frame_nums,
                                                            q_text_nums=question_lens,
                                                            chunk_num=chunk_num,
                                                            slow_feats=image_features if not self.use_slow_feat_before_mlp else image_features_before_mlp) # torch.Size([2, 6272, 896])

        # add up the video and image features.
        # ipdb.set_trace() # check the size of the video feature
        feat_combine_method = getattr(self.config, 'feat_combine_method', 'concat')
        if video_features is not None and feat_combine_method == 'add':
            assert image_features[0].shape[1]*image_features[0].shape[0] == video_features.shape[1]
            
            updated_image_feat = []
            # add up the feature
            for curr_video_feat, curr_image_feat in zip(video_features, image_features):
                curr_video_feat = rearrange(curr_video_feat, "(k s) d -> k s d", k=chunk_num)  # flatten the spatial
                updated_image_feat.append(curr_video_feat + curr_image_feat)
            image_features = updated_image_feat
        
        # proprocessing of the image feature
        if images is not None:
            image_features = self.post_processing_of_image_feature(image_features, video_idx_in_batch) # [torch.Size([6720, 896]), torch.Size([6720, 896])]
        
        # ipdb.set_trace() # before combine
        # Combine image and video feature, and update the new_frame_num
        # Do the very first version by concatenating the feature
        train_addition_start_end_tokens = getattr(self.config, 'train_addition_start_end_tokens', False)
        if train_addition_start_end_tokens:
            assert self.get_model().temporal_aggregator.start_end_tokens is not None
            start_end_token_set = self.get_model().temporal_aggregator.start_end_tokens
        
        # ipdb # check feat_combine_method
        if video_features is not None:
            if feat_combine_method == 'concat':
                image_features = torch.cat([ele.unsqueeze(dim=0) for ele in image_features], dim=0) # torch.Size([2, 6273, 896])
                video_features = torch.cat([image_features, video_features], dim=-2) # torch.Size([2, 225, 896]) + torch.Size([2, 6273, 896]) = torch.Size([2, 6498, 896])
                new_frame_num += image_features.shape[1]
            elif feat_combine_method == 'interleave':
                # ipdb.set_trace() # check no tokens, check the split, check the merging
                interleaved_feat = []
                interleaved_frame_num = []
                frame_number_per_video = images[0].shape[0] # this should be fixed for all the frames
                for curr_img_feat, curr_vid_feat, curr_vid_feat_len in zip(image_features, video_features, new_frame_num):
                    # image_features = torch.cat([ele.unsqueeze(dim=0) for ele in image_features], dim=0) # torch.Size([2, 6273, 896])
                    
                    #### handle the image features
                    total_image_tokens = curr_img_feat.shape[0]
                    assert total_image_tokens % frame_number_per_video == 0 # assert all the image tokens could be equally splited into intervals
                    tokens_per_frame = total_image_tokens // frame_number_per_video
                    image_feat_split_sizes = [tokens_per_frame for i in range(frame_number_per_video)]
                    ## seperated the image features using the frame number
                    splited_image_feat = torch.split(curr_img_feat, image_feat_split_sizes) 
                    
                    #### handle the video features
                    updated_video_feat = curr_vid_feat[:curr_vid_feat_len]
                    # split the size equally to each frames
                    video_feat_split_sizes = split_list_lengths(curr_vid_feat_len, frame_number_per_video)
                    splited_video_feat = torch.split(updated_video_feat, video_feat_split_sizes)
                    
                    #### combine the feature
                    combined_feat = []
                    for i_f, v_f in zip(splited_image_feat, splited_video_feat):
                        if train_addition_start_end_tokens:
                            combined_feat.append(start_end_token_set[0].unsqueeze(dim=0))
                            combined_feat.append(v_f)
                            combined_feat.append(start_end_token_set[1].unsqueeze(dim=0))
                            combined_feat.append(start_end_token_set[2].unsqueeze(dim=0))
                            combined_feat.append(i_f)
                            combined_feat.append(start_end_token_set[3].unsqueeze(dim=0))
                        else:
                            combined_feat.append(v_f)
                            combined_feat.append(i_f)
                    combined_feat = torch.cat(combined_feat)
                    interleaved_feat.append(combined_feat)
                    # update the new_frame_num
                    interleaved_frame_num.append(combined_feat.shape[0])
                    # ipdb.set_trace() # check no tokens, check the split, check the merging
                # ipdb.set_trace()
                new_frame_num = interleaved_frame_num
                video_features = interleaved_feat
                # ipdb.set_trace() # check no tokens, check the split, check the merging            
            elif feat_combine_method == 'add':
                # ipdb.set_trace() # check the shape the feature + new_frame_num, also check the following see whether the token selection is correct
                video_features = image_features
                new_frame_num = [ele.shape[0] for ele in image_features]
            else:
                raise NotImplementedError
        else: # IF we are not using the fast path
            # ipdb.set_trace() # the postprocessing
            video_features = image_features
            for ele in video_features: # add grad to the forward pass
                ele.requires_grad = True
            new_frame_num = [ele.shape[0] for ele in image_features]
        

        # TODO: video start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        # input_ids & labels: [torch.Size([42]), torch.Size([52]), torch.Size([40]), torch.Size([44]), torch.Size([46]), torch.Size([36]), torch.Size([42]), torch.Size([30]), torch.Size([38]), torch.Size([39]), torch.Size([52]), torch.Size([47]), torch.Size([43]), torch.Size([47]), torch.Size([42]), torch.Size([49])]
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # ipdb.set_trace() # check the forwards
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # determine how many image we have 
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = video_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            # cut the input ids and the label by the position of the image_token (-200)
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # [torch.Size([14]), torch.Size([27])]
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]]) # [14, 27]
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # convert the text idx to the text embedding: cur_input_embeds: torch.Size([65, 4096]), and split the text embedding
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # merge the text embedding with the image embedding
            # also prepare the label (the training target of the model)
            # the number of image should be 1, since the whole conversation is splited by the image
            # thus we have image + 1 of the conversation
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = video_features[cur_image_idx]
                    cur_feature_len = new_frame_num[cur_image_idx]
                    cur_image_features = cur_image_features[:cur_feature_len] # (T, C) or (T, H*W, C)
                    if len(cur_image_features.shape) == 3:
                        cur_image_features = cur_image_features.view(-1, cur_image_features.shape[-1])
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # ipdb.set_trace()
        # Combine them by padding
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        # do the padding
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # match the output with the input
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        # ipdb.set_trace()
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # use additional image token between the image and the text
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                # input_embeddings = self.get_input_embeddings().weight.data
                # output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = self.get_input_embeddings().weight.data[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = self.get_input_embeddings().weight.data[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                # tensor([[ 0.0057,  0.0160, -0.0067,  ..., -0.0011, -0.0156, -0.0035],
                #     [ 0.0057,  0.0160, -0.0067,  ..., -0.0011, -0.0156, -0.0035]],
                # dtype=torch.bfloat16)
                self.get_input_embeddings().weight.data[-num_new_tokens:] = input_embeddings_avg
                self.get_input_embeddings().weight.data[-num_new_tokens:] = output_embeddings_avg

            # ipdb.set_trace() # self.get_input_embeddings().weight.requires_grad
            if model_args.tune_temporal_aggregator or model_args.tune_addition_token_embeddings:
                # self.get_input_embeddings().requires_grad_(True)
                # self.get_output_embeddings().requires_grad_(False)
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_temporal_aggregator:
                mm_projector_weights = torch.load(model_args.pretrain_temporal_aggregator, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if self.get_input_embeddings().weight.data.shape == embed_tokens_weight.shape:
                    self.get_input_embeddings().weight.data[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    self.get_input_embeddings().weight.data[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {self.get_input_embeddings().weight.data.shape}. Numer of new tokens: {num_new_tokens}.")
        
            # ipdb.set_trace() # self.get_input_embeddings().weight.requires_grad, (check) input_embeddings_avg, self.get_input_embeddings().weight.data[-num_new_tokens:]
        elif model_args.mm_use_im_patch_token:
            # freeze the input and output tokens
            if model_args.tune_temporal_aggregator:
                self.get_input_embeddings().requires_grad_(False)
                self.get_output_embeddings().requires_grad_(False)       
                # for p in self.get_input_embeddings().parameters():
                #     p.requires_grad = False
                # for p in self.get_output_embeddings().parameters():
                #     p.requires_grad = False
        
        ### handle the special case which the len(tokenizer) != self.get_input_embeddings().weight.data.shape[0]
        if len(tokenizer) != self.get_input_embeddings().weight.data.shape[0]:
            self.resize_token_embeddings(len(tokenizer))
            # ipdb.set_trace()