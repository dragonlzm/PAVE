import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .ssm_temporal_aggregator import SSMTemporalAggregatorV5
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", getattr(vision_tower_cfg, "mm_image_tower", None)))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower:
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs) 
    raise ValueError(f"Unknown vision tower: {vision_tower}")




def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower == 'opensora':
        # return OpenSoraVideoTower(video_tower, args=video_tower_cfg, **kwargs)
        return None
    else:
        raise NotImplementedError
    
    
def build_temporal_aggregator(video_tower_cfg, **kwargs):
    adaptor_type = getattr(video_tower_cfg, 'temporal_aggregator_type', None)
    # add additional config
    video_tower_cfg.temporal_aggregator_config['train_addition_start_end_tokens'] = video_tower_cfg.train_addition_start_end_tokens

    if adaptor_type == 'ssmv5':
        return SSMTemporalAggregatorV5(**video_tower_cfg.temporal_aggregator_config)
    else:
        raise NotImplementedError