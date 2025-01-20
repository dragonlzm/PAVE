# Prepare the training and test dataset of the Music-AVQA

## Download the training / test annotations and videos
You can download the training / test annotations and videos from [Music-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/).

## Convert the training annotation 
You need to convert the raw training annotation into instruction tuning dataset. 
1. run the tools/audio/music-avqa/clean_and_check_the_annotation.py to clean the annotation.
2. run the tools/audio/music-avqa/convert_annotation_to_training_format_duplicate_audio_related.py to conver the annotation to insturction tuning dataset.


## Extract the audio feature from .mp3 files
1. Prepare the ImageBind environment following the instruction [here](https://github.com/facebookresearch/ImageBind).
2. Extract the audio from the video. Please refer to file tools\audio\extract_audio_from_video.py.
3. Extract the audio feature from the audio. Please refer to file tools\audio\extract_imagebind_audio_feature.py.

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted audio feature from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/MUCIS-AVQA-videos-Synthetic_audio_imagebind_feat.zip) and [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/MUSIC-AVQA-videos-Real_audio_imagebind_feat.zip).
