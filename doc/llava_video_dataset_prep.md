# Prepare the training and test dataset of the LLaVA-Video

## Download the training annotations and videos
You can download the training annotations and videos from [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main).

## Convert the training annotations 
You need to generate the subset of the annotation from raw annotations, please refer to tools/enhanced_video/create_llava_video_178k_anno_subset.py

## Extract the video feature from video frames and depth files
1. Prepare the LLaVA-3D environment following the instruction [here](https://github.com/ZCMax/LLaVA-3D). 
2. Run the extraction script to extract the video feature (TODO: sample scripts).

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted video feature from [here](https://huggingface.co/datasets/YiquanLi/ScanNet_for_ScanQA_SQA3D/blob/main/downsample_32_w_3d_features_refined/video_features_new.tar).

