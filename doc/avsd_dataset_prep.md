# Prepare the training and test dataset of the AVSD

## Download the training annotations 
You can download the training annotation from [AVSD](https://video-dialog.com/).

## Download the training video 
You can download the training video from [Charades](https://prior.allenai.org/projects/charades).

## Download the test annotations 
You can download the test annnotation from [AVSD-test](https://drive.google.com/file/d/1iIWsG_zdWQ5i3_crO42WOGeKOAZoG-_K/view?usp=drive_link).

## Download the test video 
You can download the test video from [Charades-test](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_vu17_test.tar).

## Convert the training annotations 
You need to convert the raw training annotations into instruction tuning dataset. Please refer to file tools\audio\avsd\convert_annotation_to_training_format.py. You can also download the processed annotations and video mapping files from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/tree/main/annotations/audio).

## Convert the test annotations 
You need to convert the raw test annotations into evaluation format. Please refer to file tools\audio\avsd\convert_test_annotation_to_eval_format.py. 

## Extract the audio feature from .mp3 files
1. Prepare the ImageBind environment following the instruction [here](https://github.com/facebookresearch/ImageBind).
2. Extract the audio from the video. Please refer to file tools\audio\extract_audio_from_video.py.
3. Extract the audio feature from the audio. Please refer to file tools\audio\extract_imagebind_audio_feature.py.

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted audio feature from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/Charades_v1_audio_imagebind_feat.zip) and [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/Charades_vu17_test_audio_imagebind_feat.zip).
