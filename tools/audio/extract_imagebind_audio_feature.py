from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
import random
import torchaudio
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from imagebind.data import waveform2melspec
import ipdb


import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ipdb
import argparse
import json
import os

np.random.seed(42)
# Set seed for PyTorch
torch.manual_seed(42)
# Set seed for Python's built-in random module
random.seed(42)

def my_load_and_transform_audio_data(
    audio_paths,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    # this function is modified based on the files:
    # function load_and_transform_audio_data in imagebind\data.py
    if audio_paths is None:
        return None

    assert len(audio_paths) == 1
    audio_outputs = []

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        # ipdb.set_trace()

        total_time = waveform.shape[1] / sample_rate
        # total_rounded_time_index = int(total_time // 2) # how many 2s clip we have
        # create a clip every 2s
        all_clips = []
        for i in range(0, int(total_time)-1):
            waveform_clip = waveform[
                :, int(i * sample_rate) : int((i + 2) * sample_rate),
            ] # waveform.shape: torch.Size([2, 1441120]) -> waveform_clip.shape: torch.Size([2, 32000])
            # print(int(i * sample_rate), int((i + 2) * sample_rate))
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            ) # waveform_clip.shape: torch.Size([2, 32000]) -> waveform_melspec.shape torch.Size([1, 128, 204])
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)
        # ipdb.set_trace()
    min_len = min([ele.shape[0] for ele in audio_outputs])
    # chop the len of the video
    audio_outputs = [ele[:min_len] for ele in audio_outputs]

    return torch.stack(audio_outputs, dim=0)



# Forward the model

###################################### follow code are written for extracting audio feature
# define the dataset
class AudioFeatureExtractDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 annotation_path,    # path to the mapping between the video id and the video path
                 data_root,
                 feature_saving_root,
                 reverse=False,
                 not_skip_the_first=True,
                 ):
        
        self.feature_saving_root = feature_saving_root
        self.data_root = data_root
        self.not_skip_the_first = not_skip_the_first
        
        # load the annotation and prepare the list
        if annotation_path is not None:
            all_anno = json.load(open(annotation_path))
            # get all the distinct video
            self.all_video = [ele['video'] for ele in all_anno]
            self.all_video = list(set(self.all_video))
            origin_len = len(self.all_video)
            # filter the dataset
            temp = []
            # ipdb.set_trace()
            for ele in self.all_video:
                if '/' in ele and self.not_skip_the_first: # since here is the audio file
                    saving_file_name = '/'.join(ele.split('/')[1:])
                else:
                    saving_file_name = ele
                    
                saving_file_name = saving_file_name.split('.')[0] + '.mp3'
                if os.path.exists(os.path.join(data_root, saving_file_name)):
                    temp.append(saving_file_name)
            self.all_video = temp
            
        else: # if the annotation is None then list all the videos on the folder
            self.all_video = os.listdir(data_root)
            origin_len = len(self.all_video)
        
        print('total unique video:', origin_len, 'video exist:', len(self.all_video))

        # reverse
        if reverse:
            self.all_video.reverse()

    def __len__(self):
        return len(self.all_video)

    def __getitem__(self, i):
        # determine the source path
        curr_audio_path = os.path.join(self.data_root, self.all_video[i])
        # ipdb.set_trace() # check the path
        # determine the save path
        # if '/' in self.all_video[i]: # since the first item is the the folder that contain video
        #     saving_file_name = ('/'.join(self.all_video[i].split('/')[1:])).split('.')[0] + '.pt'
        # else:
        saving_file_name = self.all_video[i].split('.')[0] + '.pt'
        curr_saving_path = os.path.join(self.feature_saving_root, saving_file_name)
        
        # if the feature file exist then skip
        if os.path.exists(curr_saving_path):
            data_dict = {}
            data_dict['audio'] = -1
            data_dict['audio_file_path'] = curr_audio_path    
            data_dict['feature_save_path'] = curr_saving_path
            data_dict['feature_saving_root'] = '/'.join(curr_saving_path.split('/')[:-1])
            # ipdb.set_trace()
            return data_dict        
        
        # load the audio and do the preprocessing    
        audio_tensor = my_load_and_transform_audio_data([curr_audio_path])

        # ipdb.set_trace() # check the 'feature_saving_root'
        data_dict = {
            'audio': audio_tensor,
            'audio_file_path': curr_audio_path,
            'feature_save_path': curr_saving_path,
            'feature_saving_root': '/'.join(curr_saving_path.split('/')[:-1]) 
        }
        # ipdb.set_trace()
        return data_dict


    
def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    # prepare a dataset
    extract_dataset = AudioFeatureExtractDataset(annotation_path=args.annotation_path,    # path to the mapping between the video id and the video path
                                                        data_root=args.data_root,
                                                        feature_saving_root=args.feature_saving_root,
                                                        reverse=args.reverse_dataset,
                                                        not_skip_the_first=args.not_skip_the_first,)        
    
    # define the dataloader
    extract_dataloader = DataLoader(
                extract_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=None,)
    
    # loop over the dataloader
    for count_i, sample in enumerate(extract_dataloader):
        # get the data
        curr_audio = sample['audio'].to(device) # expect torch.Size([1, K, 1, 128, 204])
        curr_saving_path = sample['feature_save_path'][0]
        curr_saving_root = sample['feature_saving_root'][0]        
        
        # check the feature is exist or not
        if len(curr_audio.shape) == 0: # if it is not a duplicated sample, it will be a dict
            assert curr_audio == -1
            assert os.path.exists(curr_saving_path)
            print(curr_saving_path, ' exist, skipped.')
            continue           

        #ipdb.set_trace() # check the size of the tensor
        curr_audio = curr_audio.squeeze(dim=1)

        # if the saving root is not exist make the saving root
        if not os.path.exists(curr_saving_root):
            os.makedirs(curr_saving_root)
        

        modality_value = curr_audio
        reduce_list = (
            modality_value.ndim >= 5
        )  # Audio and Video inputs consist of multiple clips
        # modality_value: torch.Size([2, 3, 1, 128, 204]) (B, Num_Clip, ?,?,?)
        if reduce_list:
            B, S = modality_value.shape[:2]
            modality_value = modality_value.reshape(
                B * S, *modality_value.shape[2:]
            )

        if modality_value is not None:
            # self.modality_preprocessors[modality_key].__class__: imagebind.models.multimodal_preprocessors.AudioPreprocessor
            # self.modality_trunks[modality_key].__class__: imagebind.models.transformer.SimpleTransformer
            # self.modality_heads[modality_key].__class__: torch.nn.modules.container.Sequential, scaling and normalization
            with torch.no_grad():
                modality_value = model.modality_preprocessors[ModalityType.AUDIO](
                    **{ModalityType.AUDIO: modality_value}
                )
                trunk_inputs = modality_value["trunk"] # modality_value["trunk"]['tokens'].shape: torch.Size([6, 229, 768])
                head_inputs = modality_value["head"] # this is empty
                modality_value = model.modality_trunks[ModalityType.AUDIO](**trunk_inputs) # the raw feature, which after the self-attention shape: torch.Size([6, 229, 768])
                modality_value = model.modality_heads[ModalityType.AUDIO](
                    modality_value, **head_inputs
                ) # layernorm -> select the first element -> linear layer: torch.Size([6, 229, 768]) -> torch.Size([6, 1024])
                modality_value = model.modality_postprocessors[ModalityType.AUDIO](
                    modality_value
                ) # normalization layer -> scaling: torch.Size([6, 1024])

            # ipdb.set_trace() # check the end feature shape
            assert B == 1
            # if reduce_list:
                # modality_value = modality_value.reshape(B*S, -1) # torch.Size([2, 3, 1024])
                # modality_value = modality_value.mean(dim=1) # torch.Size([2, 1024])
            # every 2s become a clips, in here we generate 1 tokens for 2 second
        
        # save the result
        torch.save(modality_value.detach().cpu().to(torch.bfloat16), curr_saving_path)
        print('total: ', len(extract_dataset), ' curr:', count_i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract image feature from image backebon')
    parser.add_argument('--data-root', dest='data_root', type=str, default=None)
    parser.add_argument('--feature-saving-root', dest='feature_saving_root', type=str, default=None)     
    parser.add_argument('--annotation-path', dest='annotation_path', type=str, default=None)   
    parser.add_argument('--reverse-dataset', dest='reverse_dataset', action='store_true')  
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=2) 
    parser.add_argument('--not-skip-the-first', dest='not_skip_the_first', action='store_false')      
    # parser.add_argument('--dataset-filter-key', dest='dataset_filter_key', type=str, default=None)
    args = parser.parse_args()
    main(args)
    
# Sample usage:
# CUDA_VISIBLE_DEVICES=0 python tools/Audio/extract_imagebind_audio_feature.py \
# --data-root ./MUCIS-AVQA-videos-Synthetic_audio \
# --feature-saving-root ./MUCIS-AVQA-videos-Synthetic_audio_imagebind_feat \
# --num-workers 0

# CUDA_VISIBLE_DEVICES=0 python  tools/Audio/extract_imagebind_audio_feature.py \
# --data-root ./MUSIC-AVQA-videos-Real_audio \
# --feature-saving-root ./MUSIC-AVQA-videos-Real_audio_image_feat \

# CUDA_VISIBLE_DEVICES=1 python tools/Audio/extract_imagebind_audio_feature.py \
# --data-root ./Charades_v1_audio \
# --feature-saving-root ./Charades_v1_audio_imagebind_feat \

    
# CUDA_VISIBLE_DEVICES=1 python tools/Audio/extract_imagebind_audio_feature.py \
# --data-root ./Charades_vu17_test_audio \
# --feature-saving-root ./Charades_vu17_test_audio_imagebind_feat \


