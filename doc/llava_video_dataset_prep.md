# Prepare the training and test dataset of the LLaVA-Video

## Download the training annotations and videos
You can download the training annotations and videos from [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main).

## Convert the training annotations 
You need to generate the subset of the annotation from raw annotation files, please refer to tools/enhanced_video/create_llava_video_178k_anno_subset.py. You can also download the processed annotations and video mapping files from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/tree/main/annotations/LLaVA_Video_178K).

## Extract the video feature from video frames files
1. Prepare the Video-LLaVA environment following the instruction [here](https://github.com/PKU-YuanGroup/Video-LLaVA). We use the LanguageBind Video backbone from Video-LLaVA. 
2. Run the extraction script to extract the video feature. Please refer to a sample scripts tools/enhanced_video/extract_languagebind_feature.py.

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted video feature from:
1. [2_3_m_academic](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/LLaVA_Video_178K_2_3_m_academic_v0_1_languagebind_feat.zip)
2. [2_3_m_youtube_split1](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/LLaVA_Video_178K_2_3_m_youtube_v0_1_split1.zip), [2_3_m_youtube_split2](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/LLaVA_Video_178K_2_3_m_youtube_v0_1_split2.zip), When you unzip these two files, create a folder `languagebind_feat/videos/youtube_video_2024` and put all the .pt file under this new folder.
3. [1_2_m_academic](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/LLaVA_Video_178K_1_2_m_academic_v0_1_languagebind_feat.zip) 
4. [1_2_m_youtube](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/blob/main/LLaVA_Video_178K_1_2_m_youtube_v0_1_languagebind_feat.zip).

