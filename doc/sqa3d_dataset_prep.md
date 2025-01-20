# Prepare the training and test dataset of the SQA3D

## Download the training / test annotations and videos
You can download the training / test annotations and videos from [SQA3D](https://sqa3d.github.io/).

## Convert the training annotation 
You need to convert the raw training annotation into instruction tuning dataset. Please refer to file tools\AVSD\convert_annotation_to_training_format.py

## Extract the 3d feature from video frames and depth files
1. Prepare the ImageBind environment following the instruction [here](https://github.com/facebookresearch/ImageBind). ??
2. Extract the audio from the video. Please refer to file tools\Audio\extract_audio_from_video.py.
3. Extract the audio feature from the audio. Please refer to file tools\Audio\extract_imagebind_audio_feature.py.

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted audio feature from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/Charades_v1_audio_imagebind_feat.zip) and [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/Charades_vu17_test_audio_imagebind_feat.zip).
