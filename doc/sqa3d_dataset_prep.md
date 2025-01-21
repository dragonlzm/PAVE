# Prepare the training and test dataset of the SQA3D

## Download the training / test annotations and videos
1. You can download the training / test annotations and videos from [SQA3D](https://sqa3d.github.io/).
2. Evenly extract the 32 RGB frames and depth frames from the raw files (TODO: sample scripts).

## Convert the training annotation 
You need to convert the raw training annotation into instruction tuning dataset. Please refer to file tools/3d/sqa3d/create_sqa3d_training_annotations.py. You can also download the processed annotations and video mapping files from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/tree/main/annotations/3d).

## Convert the test annotations 
You need to convert the raw test annotations into test format. Please refer to file tools/3d/sqa3d/create_sqa3d_eval_annotations.py.

## Extract the 3d feature from video frames and depth files
1. Prepare the LLaVA-3D environment following the instruction [here](https://github.com/ZCMax/LLaVA-3D). 
2. Run the extraction script to extract the 3d feature (TODO: sample scripts).

## Download the pre-extracted audio feature
If you don't want to extract the feature by yourself, you can also download the pre-extracted 3d feature from [here](https://huggingface.co/datasets/YiquanLi/ScanNet_for_ScanQA_SQA3D/blob/main/downsample_32_w_3d_features_refined/video_features_new.tar).