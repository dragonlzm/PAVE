# Prepare the data of Ego-Exo4d

## Obtain the license
Follow the instruction [here](https://ego-exo4d-data.org/#download) to obtain the access to the Ego-Exo4d dataset.

## Download the videos and annotations 
You can download the video and annotation using following commands.
```
egoexo -o "/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin" --parts annotations --benchmarks proficiency_demonstrator
egoexo -o "/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/egoexo_origin" --parts downscaled_takes/448 --benchmarks proficiency_demonstrator
```

## Convert the training annotations 
You need to convert the raw training annotations into instruction tuning dataset. Please refer to file tools/egoexo/convert_annotation_to_instruction_dataset.py. You can also download the processed annotations and video mapping files from [here](https://huggingface.co/datasets/zhuomingliu/PAVEDataset/tree/main/annotations/egoexo).


## Extract the video feature from exo-centric videos
You can use the tools/egoexo/extract_siglip_egoexo_feature.py to extract the video feature from exo-centric videos.
