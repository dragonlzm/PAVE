# Prepare the training and test dataset of the AVQA

## Download the annotations 
You can download the training annotations from [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/).

## Download the video 
Since the AVQA does not provide the method for downloading the videos. You can download the VGGSound and filter the video. The VGGSound can be found [here](https://huggingface.co/datasets/Loie/VGGSound). 
You can then filter out the videos needed by the AVQA, a sample script is provided in tools/audio/avqa/select_avqa_subset.py. You can also download the prefiltered videos created by us from [here](https://huggingface.co/datasets/zhuomingliu/PAVE_others/tree/main/avqa).

## Convert the training annotations 
You need to convert the raw training annotations into instruction tuning dataset. Please refer to file tools/audio/avqa/convert_annotation_to_instruction_dataset.py. You can also download the processed annotations and video mapping files from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/tree/main/annotations/audio).

## Extract the audio feature from .mp3 files
1. Prepare the ImageBind environment following the instruction [here](https://github.com/facebookresearch/ImageBind).
2. Extract the audio from the video. Please refer to file tools/audio/extract_audio_from_video.py.
3. Extract the audio feature from the audio. Please refer to file tools/audio/extract_imagebind_audio_feature.py.

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted audio feature from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/avqa_subset_audio_imagebind_feat.zip).