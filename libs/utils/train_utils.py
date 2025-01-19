import os
import ipdb
from dataclasses import dataclass, field
import logging
from typing import Dict, Optional, Sequence, List

import torch
import transformers

from transformers.trainer import logger


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


@dataclass
class ModelArguments:
    '''
        For LLaVA model  
    '''
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class VideoModelArguments:
    '''
        For the normal video insturction tuning model
    '''
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ## LLM config
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                           # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ## video tower config
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = dict(
        type="OpenSoraVAE_V1_2",
        from_pretrained="./checkpoints/opensora/OpenSora-VAE-v1.2",
        micro_frame_size=17, # was 17
        micro_batch_size=4, # was 4
    )    # the config of the VAE in the OpenSora
    text_encoder = dict(
        type="fix_text_embedding",
        from_pretrained="./checkpoints/opensora/t5_sample_encoding.pt",
    )
    # text_encoder = dict(
    #     type="t5",
    #     from_pretrained="DeepFloyd/t5-v1_1-xxl",
    #     model_max_length=300,
    #     shardformer=True,
    # )    # the config of the text encoder in the OpenSora
    diffu = dict(
        type="STDiT3-XL/2",
        from_pretrained='./checkpoints/opensora/OpenSora-STDiT-v3/',
        qk_norm=True,
        enable_flash_attn=True,
        enable_layernorm_kernel=True,
        freeze_y_embedder=True,
    )    # the config of the diffusion module in the OpenSora
    diffu_t = 1 # the amount of noise we add on the video
    
    ## temporal_aggregator config
    tune_temporal_aggregator: bool = field(default=False)             # It will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # It will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None) # whether we have the pretrained temporal_aggregator (for step 2 training)
    mm_vision_select_layer: Optional[int] = field(default=-1)         # default to the last layer
    temporal_aggregator_type: Optional[str] = field(default='ssm')    # The model type of the temporal_aggregator
    temporal_aggregator_config = dict(
        input_dim=1152,
        embd_dim=1152,
        output_dim=896,
    )  


@dataclass
class VideoFeatModelArgumentsV5:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It only use the opensora vae feature, do not use the diffusion module
        2. It still use conv layer to handle the vae feature
        3. it use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=4,
        embed_dim=256,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        
        number_of_input_mapping_layer=3,      # conv layers
        number_of_block_each_layer=[1,2,2],
        sptial_stride=[1,1,2], 
        dim_scaling = [1,1,2],       
        
        query_input_dim=896,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,        
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. it use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=896,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,        
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)

@dataclass
class VideoFeatModelArgumentsV5_1_1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. it use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. this special version use the learnable query tokens
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_type='learnable',
        chunks_number=32,
        number_of_query=196,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,        
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_2:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)

@dataclass
class VideoFeatModelArgumentsV5_1_2_siglip:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1152,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)




@dataclass
class VideoFeatModelArgumentsV5_1_3:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_h1024:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=1024,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=1024,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)

@dataclass
class VideoFeatModelArgumentsV5_1_3_7B:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_7B_h1024:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=1024,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=1024,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_7B_hyper:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. using the 3d rope
        5. use 1024 cross-attn hidden, splited into 8 heads
        6. use 4 cross-attn
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=1024,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=1024,
        num_cross_attn_head=8,
        num_cross_attn_layer=4,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_3d:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the 3d input is 1024
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_3d_7B:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the 3d input is 1024
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_3d_v1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
        5. only one layer cross-attn
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the 3d input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=1,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_3d_v1_1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the 3d input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_3d_learnable_query:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the audio input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        query_type='learnable',
        chunks_number=32,
        number_of_query=196,

        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=False         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_audio:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_learnable_query:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        query_type='learnable',
        chunks_number=32,
        number_of_query=196,

        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=False         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_7B:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=3584, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_7B_3layers:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=3584, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=3,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the audio input is 1024
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_3layers:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the audio input is 1024
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=3,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_1layers:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,  # the audio input is 1024
        output_dim=3584, # for 7B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=1,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_v1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=1,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=False,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_1_3_audio_v1_1:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,  # the audio input is 768
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)




@dataclass
class VideoFeatModelArgumentsV5_1_4:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. 3 cross-attn layers
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=3,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_1_5:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. 1 cross-attn layers
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')    # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=1152,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=1,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,
        use_slow_feat_before_mlp=True         # when using the slow-feat for cross-attn training, we use the slow-feature before MLP
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)




@dataclass
class VideoFeatModelArgumentsV5_1_mplug:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. it use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. this version use the mplug-owl3 way to inject hte vision information
    ''' 
    ############################ For setting the base model ##############################
    use_hyper_attention: bool = field(default=True)
    
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='pmv5')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        embed_dim=512,
        
        fast_input_mapping_type='linear',   
        
        query_input_dim=896,
        cross_attn_hidden_dim=512,
        num_cross_attn_head=4,
        num_cross_attn_layer=2,
        # query_per_chunk=False,
        # chunks_number=32,
        
        use_output_mlp=True,
        use_dropout=False,
        use_output_norm=True,
        train_addition_start_end_tokens=False,        
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)



@dataclass
class VideoFeatModelArgumentsV5_2:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. This version use the global average pooling and MLP 
        3. Conbine method use interleave 
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='ssmv1_2')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        downsample_factor=2,         # the downsample factor of the adaptive pooling
        input_shape=(1024, 2, 2),     # the input shape of the feature (C, H, W)     
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="no_token")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="interleave")
    train_addition_start_end_tokens: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsV5_2_audio:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. This version use the global average pooling and MLP 
        3. Conbine method use interleave 
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='ssmv1_2')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=768,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        downsample_factor=1,         # the downsample factor of the adaptive pooling
        input_shape=(768, 1, 1),     # the input shape of the feature (C, H, W)     
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="no_token")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="interleave")
    train_addition_start_end_tokens: bool = field(default=False)






@dataclass
class VideoFeatModelArgumentsV5_2_3d:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. This version use the global average pooling and MLP 
        3. Conbine method use interleave 
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='ssmv1_2')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        downsample_factor=2,         # the downsample factor of the adaptive pooling
        input_shape=(1024, 24, 24),     # the input shape of the feature (C, H, W)    
        flatten_spatial=False, 
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="no_token")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="interleave")
    train_addition_start_end_tokens: bool = field(default=False)





@dataclass
class VideoFeatModelArgumentsV5_2_3d_add:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. This version use the global average pooling and MLP 
        3. Conbine method use interleave 
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='ssmv1_2')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896, # for 0.5B model
        # output_dim=1536, # for 1.5B model
        downsample_factor=1.7,         # the downsample factor of the adaptive pooling
        input_shape=(1024, 24, 24),     # the input shape of the feature (C, H, W)    
        flatten_spatial=False, 
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="no_token")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)






@dataclass
class VideoFeatModelArgumentsV5_3:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        3. This version only use mlp layer 
        4. combine method use add
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ temporal_aggregator config ##########################################
    tune_temporal_aggregator: bool = field(default=False)               # If True, it will make all params freeze expect the temporal_aggregator
    # freeze_temporal_aggregator: bool = field(default=False)           # If True, it will make all params freeze
    pretrain_temporal_aggregator: Optional[str] = field(default=None)   # whether we have the pretrained temporal_aggregator (for step 2 training)
    pretrain_model_checkpoint: Optional[str] = field(default=None)      # Whether we have the pretrained model (for the ppo training) 
                                                                        # it expect the path has the checkpoint of the temporal aggregator and the LoRA weights    
    temporal_aggregator_type: Optional[str] = field(default='ssmv1_2')  # The model type of the temporal_aggregator
    # The first version with input embed + flatten feature
    temporal_aggregator_config = dict(
        input_dim=1024,
        output_dim=896,                 # for 0.5B model
        # output_dim=1536,              # for 1.5B model
        downsample_factor=1,            # the downsample factor of the adaptive pooling (here is not downsampling)
        input_shape=(1024, 14, 14),     # the input shape of the feature (C, H, W)     
        flatten_spatial=False,
        use_output_norm=True,
    ) 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="no_token")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    feat_combine_method: str = field(default="add")
    train_addition_start_end_tokens: bool = field(default=False)






@dataclass
class DataArguments:
    '''
        For the LLaVA dataset
    '''
    
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    

@dataclass
class VideoDataArguments:
    '''
        For the normal video instruction tuning dataset
    '''    
    
    annotation_path: str = field(default=None, metadata={"help": "Path to the annotation."})
    feat_mapping_path: str = field(default=None, metadata={"help": "Path to the mapping."})
    data_root: Optional[str] = field(default=None)
    use_video_feat: bool = False
    lazy_preprocess: bool = False
    is_multimodal: bool = True

    image_size = (224, 224)
    transform_name = 'resize_crop'
    num_frames = 50 # was the 17 * 10
    # num_frames=32
    # frame_interval=1



@dataclass
class VideoFeatModelArguments_finetuning:
    '''
        For the video insturction tuning model trained from the LLaVA-OneVision 
        It consist of two paths: the slow-path, which is the image-path, which directly comes from LLaVA-OneVision
        and the fast-path which use the pre-extracted Video-VAE features.
        Video Path (Fast-path) Features:
        1. It use the languagebind as the fast path input
        2. It use cross-attention beween the slowfeatre and the fast feature to aggregate the information in the fast
        3. This is a special version of which will load the video and use the feature before the MLP for cross-attn
        4. not using the 3d rope
    ''' 
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = None               # the type of the vision backbone
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
    
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    # rope_scaling_factor: Optional[float] = field(default=None)
    # rope_scaling_type: Optional[str] = field(default=None)

    # s2: Optional[bool] = field(default=False)
    # s2_scales: Optional[str] = field(default="336,672,1008")

    # use_pos_skipping: Optional[bool] = field(default=False)
    # pos_skipping_range: Optional[int] = field(default=4096)

    # mm_newline_position: Optional[str] = field(default="grid")
    # delay_load: Optional[bool] = field(default=True)
    # add_faster_video: Optional[bool] = field(default=False)
    # faster_token_stride: Optional[int] = field(default=10)
    ######################### feature merging config ############################################
    train_addition_start_end_tokens: bool = field(default=False)




@dataclass
class VideoFeatModelArguments_finetuning_videollava:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    # ===================================================================


@dataclass
class VideoFeatDataArguments:
    '''
        For the video instruction tuning dataset with pre-extracted feature.
        Probably we will also use the video data, if we use the llava-onevision.
    ''' 
    ### if and only if we define like this
    # use_fast_feat: bool = True, the variable could be recognized by the hugging face
    
    annotation_path: str = field(default=None, metadata={"help": "Path to the annotation."})
    fast_path_mapping_path: str = field(default=None, metadata={"help": "Path to the fast path data mapping file."})
    slow_path_mapping_path: str = field(default=None, metadata={"help": "Path to the video mapping."})
    data_root: Optional[str] = field(default=None, metadata={"help": "Path to the video feature."})
    slow_path_data_root: Optional[str] = field(default=None, metadata={"help": "Path to the slowpath data."})
    data_sample_ratio: Optional[str] = field(default=None, metadata={"help": "ratio of each dataset sampled"})
    # defined some hyper of the dataset
    use_fast: bool = False      # a special version which directly send the video frame in
    use_fast_feat: bool = True  # use pre-extracted video feature in the training
    use_slow: bool = False      # use the image and the image backbone for the training
    use_slow_feat: bool = False # use pre-extracted video feature in the training
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    prepare_qid: bool = True

    # For fast loading
    fast_feat_type: str = 'video_vae'
    original_feat_fps: int = 24                 # for video-vae feature
    training_feat_fps: int = 4
    min_fast_frame_num: int = 32
    exclude_languagebind_cls_token: bool = True # for languagebind feature: this is for excluding the first token from the languagebind during the training
    
    # For slow loading
    frames_upbound: int = 32
    force_sample: int = True # setting frames_upbound and force_sample will force the number of frame to be 32
    video_fps: int = 1       # like a scaling factor 
    
    image_size = (224, 224) # TODO: be careful about this part check meaningless config
    transform_name = 'resize_crop'  # TODO: be careful about this part check meaningless config
    # num_frames = 50 # was the 17 * 10
    




@dataclass
class VideoTrainingArguments(transformers.TrainingArguments):
    '''
        For the normal video instruction tuning training argument
    '''        
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False) # Alert: this should not be removed
    # freeze_mm_mlp_adapter: bool = field(default=False)
    # mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    temporal_aggregator_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    max_grad_norm: float = 0.1
    
    full_determinism: bool = True
    seed: int = 42 
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=0.0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    '''
        For the LLaVA training argument
    ''' 
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3_with_state_dict(state_dict, special_key=[]):
    # this is a special version of the get_peft_state_non_lora_maybe_zero_3
    # it use the state dict to save the model,
    # this will save the 'running_mean', 'running_var', 'num_batches_tracked' in batch norm
    # special_key = ['running_mean', 'running_var', 'num_batches_tracked']
    non_lora = {k: state_dict[k] for k in state_dict if "lora_" not in k}
    # filter the state dict with the key
    to_return = {}
    for k, t in non_lora.items():
        if any(key_match in k for key_match in special_key):
            to_return[k] = t

    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3_with_state_dict(state_dict, keys_to_match):
    # this is a special version of saving the params with given key
    # it will use the state_dict to save the parameters
    # this will also save the 'running_mean', 'running_var', 'num_batches_tracked' in the batch norm
    to_return = {k: state_dict[k] for k in state_dict if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_temporal_aggregator", False):
        # Only save Adapter
        keys_to_match = ['temporal_aggregator', 
                         'self_attn.v_kv_proj', # for the mplug-owl3
                         'self_attn.gate_proj']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def safe_save_model_for_hf_videotrainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_temporal_aggregator", False):
        # Only save Adapter
        keys_to_match = ['temporal_aggregator', 
                        'self_attn.v_kv_proj', # for the mplug-owl3
                        'self_attn.gate_proj']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3_with_state_dict(trainer.model.state_dict(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

# the mapping between the name of the class and the class function
MODEL_ARGUMENTS_MAPPING = {
    'default': VideoFeatModelArgumentsV5_1_3,
    'VideoFeatModelArgumentsV5': VideoFeatModelArgumentsV5,
    'VideoFeatModelArgumentsV5_1': VideoFeatModelArgumentsV5_1,
    'VideoFeatModelArgumentsV5_2': VideoFeatModelArgumentsV5_2,
    'VideoFeatModelArgumentsV5_2_audio': VideoFeatModelArgumentsV5_2_audio,
    'VideoFeatModelArgumentsV5_2_3d': VideoFeatModelArgumentsV5_2_3d,
    'VideoFeatModelArgumentsV5_2_3d_add': VideoFeatModelArgumentsV5_2_3d_add,
    
    'VideoFeatModelArgumentsV5_1_mplug': VideoFeatModelArgumentsV5_1_mplug,
    'VideoFeatModelArgumentsV5_1_1': VideoFeatModelArgumentsV5_1_1,
    'VideoFeatModelArgumentsV5_1_2': VideoFeatModelArgumentsV5_1_2,
    'VideoFeatModelArgumentsV5_1_2_siglip': VideoFeatModelArgumentsV5_1_2_siglip,
    
    'VideoFeatModelArgumentsV5_1_3': VideoFeatModelArgumentsV5_1_3,
    'VideoFeatModelArgumentsV5_1_3_h1024': VideoFeatModelArgumentsV5_1_3_h1024,
    'VideoFeatModelArgumentsV5_1_3_7B': VideoFeatModelArgumentsV5_1_3_7B,
    'VideoFeatModelArgumentsV5_1_3_7B_h1024': VideoFeatModelArgumentsV5_1_3_7B_h1024,
    'VideoFeatModelArgumentsV5_1_3_7B_hyper': VideoFeatModelArgumentsV5_1_3_7B_hyper,
    'VideoFeatModelArgumentsV5_1_3_audio': VideoFeatModelArgumentsV5_1_3_audio,
    'VideoFeatModelArgumentsV5_1_3_audio_v1': VideoFeatModelArgumentsV5_1_3_audio_v1,
    'VideoFeatModelArgumentsV5_1_3_audio_v1_1': VideoFeatModelArgumentsV5_1_3_audio_v1_1,
    'VideoFeatModelArgumentsV5_1_3_audio_learnable_query': VideoFeatModelArgumentsV5_1_3_audio_learnable_query,
    'VideoFeatModelArgumentsV5_1_3_audio_7B': VideoFeatModelArgumentsV5_1_3_audio_7B,
    'VideoFeatModelArgumentsV5_1_3_audio_7B_3layers': VideoFeatModelArgumentsV5_1_3_audio_7B_3layers,
    'VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B': VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B,
    'VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_3layers': VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_3layers,
    'VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_1layers': VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_1layers,
    'VideoFeatModelArgumentsV5_1_3_3d': VideoFeatModelArgumentsV5_1_3_3d,
    'VideoFeatModelArgumentsV5_1_3_3d_v1': VideoFeatModelArgumentsV5_1_3_3d_v1,
    'VideoFeatModelArgumentsV5_1_3_3d_v1_1': VideoFeatModelArgumentsV5_1_3_3d_v1_1,
    'VideoFeatModelArgumentsV5_1_3_3d_learnable_query': VideoFeatModelArgumentsV5_1_3_3d_learnable_query,
    'VideoFeatModelArgumentsV5_1_3_3d_7B': VideoFeatModelArgumentsV5_1_3_3d_7B,
    
    'VideoFeatModelArgumentsV5_1_4': VideoFeatModelArgumentsV5_1_4,
    'VideoFeatModelArgumentsV5_1_5': VideoFeatModelArgumentsV5_1_5,
    'VideoFeatModelArgumentsV5_3': VideoFeatModelArgumentsV5_3,
    'VideoFeatModelArguments_finetuning': VideoFeatModelArguments_finetuning,
    'VideoFeatModelArguments_finetuning_videollava': VideoFeatModelArguments_finetuning_videollava,
}

DATA_ARGUMENTS_MAPPING = {
    'default': VideoFeatDataArguments,
    'VideoFeatDataArguments': VideoFeatDataArguments
}

TRAINING_ARGUMENTS_MAPPING = {
    'default': VideoTrainingArguments,
    'VideoTrainingArguments': VideoTrainingArguments
}


def parse_argument_classes(sys_args):
    # This function aims to takes all sys arguments input,
    # figure out the model_class, data_class, training_class 
    # and retunr the remaining arguments
    
    # parse the arugment for the model
    remaining_args = []
    model_class_name = 'default'
    data_class_name = 'default'
    training_class_name = 'default'
    i = 0
    while i < len(sys_args):
        ele = sys_args[i]
        if ele == '--model_class':
            assert i+1 < len(sys_args) # assert is not empty
            model_class_name = sys_args[i+1]
            i += 2
        elif ele == '--data_class':
            assert i+1 < len(sys_args) # assert is not empty
            data_class_name = sys_args[i+1]
            i += 2
        elif ele == '--training_class':
            assert i+1 < len(sys_args) # assert is not empty
            training_class_name = sys_args[i+1]
            i += 2
        else:
            remaining_args.append(ele)
            i += 1
    
    # find the class through mapping
    model_arg_class = MODEL_ARGUMENTS_MAPPING[model_class_name]
    data_arg_class = DATA_ARGUMENTS_MAPPING[data_class_name]
    training_arg_class = TRAINING_ARGUMENTS_MAPPING[training_class_name]
    print('model_class_name:', model_class_name, 'data_class_name:', data_class_name, 'training_class_name:', training_class_name)

    logger.info('model_class_name: ' + model_class_name + ' data_class_name: ' + data_class_name + ' training_class_name: ' + training_class_name)
    
    ##################################################### parse the dataset information #################################################################
    # find all the argument head, and mark the position of the --fast_path_mapping_path and --data_root
    all_argument_head_pos = []
    annotation_path_pos = None            # the position in the remaining_args
    annotation_path_pos_in_list = None    # the position in the all_argument_head_pos
    fast_path_mapping_path_pos = None          # the position in the remaining_args
    fast_path_mapping_path_pos_in_list = None  # the position in the all_argument_head_pos
    slow_path_mapping_path_pos = None         # the position in the remaining_args
    slow_path_mapping_path_pos_in_list = None # the position in the all_argument_head_pos
    data_root_pos = None                  # the position in the remaining_args
    data_root_pos_in_list = None          # the position in the all_argument_head_pos
    slow_path_data_root_pos = None            # the position in the remaining_args
    slow_path_data_root_pos_in_list = None    # the position in the all_argument_head_pos
    
    
    for i, curr_arg in enumerate(remaining_args):
        if curr_arg.startswith('--'):
            all_argument_head_pos.append(i)
        if curr_arg == '--annotation_path':
            annotation_path_pos_in_list = len(all_argument_head_pos) - 1
            annotation_path_pos = i
        if curr_arg == '--fast_path_mapping_path':
            fast_path_mapping_path_pos_in_list = len(all_argument_head_pos) - 1
            fast_path_mapping_path_pos = i
        if curr_arg == '--slow_path_mapping_path':
            slow_path_mapping_path_pos_in_list = len(all_argument_head_pos) - 1
            slow_path_mapping_path_pos = i            
        if curr_arg == '--data_root':
            data_root_pos_in_list = len(all_argument_head_pos) - 1
            data_root_pos = i
        if curr_arg == '--slow_path_data_root':
            slow_path_data_root_pos_in_list = len(all_argument_head_pos) - 1
            slow_path_data_root_pos = i            

    # figure out the len of the arugment input for the data_root_pos and all_argument_head_pos
    assert annotation_path_pos is not None
    assert annotation_path_pos_in_list is not None
    assert fast_path_mapping_path_pos is not None 
    assert fast_path_mapping_path_pos_in_list is not None
    assert data_root_pos is not None
    assert data_root_pos_in_list is not None
    # p.s  slow_path_mapping_path_pos, slow_path_mapping_path_pos_in_list, slow_path_data_root_pos, slow_path_data_root_pos_in_list could be None
    
    annotation_path_start = annotation_path_pos + 1
    annotation_path_end = all_argument_head_pos[annotation_path_pos_in_list + 1] \
        if annotation_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)    # if it is the last arguments
    
    fast_path_mapping_path_start = fast_path_mapping_path_pos + 1
    fast_path_mapping_path_end = all_argument_head_pos[fast_path_mapping_path_pos_in_list + 1] \
        if fast_path_mapping_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)  # if it is the last arguments
    
    data_root_start = data_root_pos + 1
    data_root_end = all_argument_head_pos[data_root_pos_in_list + 1] \
        if data_root_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)          # if it is the last arguments
    
    num_of_annotation_path = annotation_path_end - annotation_path_start
    num_of_fast_path_mapping_path = fast_path_mapping_path_end - fast_path_mapping_path_start
    num_of_data_root = data_root_end - data_root_start
    
    # assert the len of the --data_root == --fast_path_mapping_path
    # ipdb.set_trace() # check the argument parser
    assert num_of_fast_path_mapping_path == num_of_data_root == num_of_annotation_path
    assert num_of_fast_path_mapping_path > 0
    
    if slow_path_mapping_path_pos is not None: # the input has the video mapping
        slow_path_mapping_path_start = slow_path_mapping_path_pos + 1
        slow_path_mapping_path_end = all_argument_head_pos[slow_path_mapping_path_pos_in_list + 1] \
            if slow_path_mapping_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)  # if it is the last arguments        
    
        assert slow_path_data_root_pos is not None
        slow_path_data_root_start = slow_path_data_root_pos + 1
        slow_path_data_root_end = all_argument_head_pos[slow_path_data_root_pos_in_list + 1] \
            if slow_path_data_root_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)     # if it is the last arguments
        
        num_of_slow_path_mapping_path = slow_path_mapping_path_end - slow_path_mapping_path_start
        num_of_slow_path_data_root = slow_path_data_root_end - slow_path_data_root_start
        
        assert num_of_annotation_path ==  num_of_slow_path_mapping_path == num_of_slow_path_data_root
    else:
        num_of_slow_path_mapping_path = 0
        num_of_slow_path_data_root = 0
    
    # assign the result to the data_arg_class
    if num_of_fast_path_mapping_path > 1:
        annotation_path_set = remaining_args[annotation_path_start:annotation_path_end]
        fast_path_mapping_path_set = remaining_args[fast_path_mapping_path_start: fast_path_mapping_path_end]
        data_root_set = remaining_args[data_root_start: data_root_end]
        filter_set = annotation_path_set + fast_path_mapping_path_set + data_root_set + ['--annotation_path', '--fast_path_mapping_path', '--data_root']
        
        if num_of_slow_path_mapping_path > 0:
            slow_path_mapping_path_set = remaining_args[slow_path_mapping_path_start: slow_path_mapping_path_end]
            slow_path_data_root_set = remaining_args[slow_path_data_root_start: slow_path_data_root_end]
            filter_set = filter_set + slow_path_mapping_path_set + slow_path_data_root_set + ['--slow_path_mapping_path', '--slow_path_data_root']
            
    else:
        annotation_path_set = remaining_args[annotation_path_start]
        fast_path_mapping_path_set = remaining_args[fast_path_mapping_path_start]
        data_root_set = remaining_args[data_root_start]
        filter_set = ['--annotation_path', '--fast_path_mapping_path', '--data_root', remaining_args[annotation_path_start], remaining_args[fast_path_mapping_path_start], remaining_args[data_root_start]]
        
        if num_of_slow_path_mapping_path > 0:
            slow_path_mapping_path_set = remaining_args[slow_path_mapping_path_start]
            slow_path_data_root_set = remaining_args[slow_path_data_root_start]
            filter_set = filter_set + [slow_path_mapping_path_set, slow_path_data_root_set, '--slow_path_mapping_path', '--slow_path_data_root']
        
    dataset_argument_filtered = [ele for ele in remaining_args if ele not in filter_set]
    
    ######################################################## parser the rest of the auguments ##############################################
    parser = transformers.HfArgumentParser(
        (model_arg_class, data_arg_class, training_arg_class))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=dataset_argument_filtered)    
    
    # assign the dataset value back to the class
    data_args.annotation_path = annotation_path_set
    data_args.fast_path_mapping_path = fast_path_mapping_path_set
    data_args.data_root = data_root_set
    if num_of_slow_path_mapping_path > 0:
        data_args.slow_path_mapping_path = slow_path_mapping_path_set
        data_args.slow_path_data_root = slow_path_data_root_set
    
    
    return model_args, data_args, training_args
        
    
