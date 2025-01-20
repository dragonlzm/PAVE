# 🛠️ PAVE: Patching and Adapting Video Large Language Models

## Contents
- [Install](#install)
- [PAVE Weights](#pave-weights)
- [Features](#features)
- [Demo](#Demo)
- [Train](#train)
- [Evaluation](#evaluation)

<!-- - [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) -->
<!-- - [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) -->



## Install

We install this environment on Linux machine:
1. Clone the repository and navigate to PAVE folder 
```
git clone https://github.com/dragonlzm/PAVE_test.git
cd PAVE_test
```

2. Install Packages
```
conda create -n pave python=3.10 -y
conda activate pave
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir
pip install peft==0.10.0
pip install rotary-embedding-torch
```


## PAVE Weights
### 3D-QA

| Dataset | Base Model | Schedule | Checkpoint | ScanQA (C) | ScanQA (B-4) | ScanQA (M) | ScanQA (R) | ScanQA (EM@1) | SQA3D (EM@1)
|----------|----------|-----------|-----------|---|---|---|---|---|---|
| ScanQA | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 1e | [pave_scanqa_v5_1_3_3d_lora](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_scanqa_v5_1_3_3d_lora.zip) |  84.2 | 13.1 | 17.0 | 42.1 | 23.1 (40.0) | - |
| ScanQA | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 1e | [pave_scanqa_v5_1_3_3d_lora_7B](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_scanqa_v5_1_3_3d_lora_7B.zip) | 103.4 | 16.0 | 19.9 | 49.0 | 29.1 (48.5) | - | 
| SQA3D  | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 2e | Coming Soon | - | - | - | - | - | ? | 
| SQA3D  | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 2e | [pave_sqa3d_v5_1_3_3d_lora_7B_2epoch](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_sqa3d_v5_1_3_3d_lora_7B_2epoch.zip) | - | - | - | - | - | 59.0 (61.4) | 

### Audio-Visual

| Dataset | Base Model | Schedule | Checkpoint | AVSD (CIDEr) | Music-AVQA (Audio Acc) | Music-AVQA (Visual Acc) | Music-AVQA (Audio-Visual Acc) | Music-AVQA (Overall Acc) |
|----------|----------|-----------|-----------|---|---|---|---|---|---|
| AVSD | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 1e | [pave_v5_1_3_lora_avsd](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_v5_1_3_lora_avsd.zip) | 134.2 | - | - | - | - |
| AVSD | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 1e | [pave_v5_1_3_lora_avsd_7B](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_v5_1_3_lora_avsd_7B.zip) | 152.9 | - | - | - | - | 
| Music-AVQA | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 2e | Coming Soon | - | ? | ? | ? | ? |
| Music-AVQA | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 2e | [pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.zip) | - | 79.7 | 93.0 | 78.0 | 82.3 |


### Enhancing Video QA


## Features
### Pre-extracted features

### Prepare the feature by yourself


## Demo

## Train

## Evaluation

## Personal Notes
1. we do minor twist in the libs\model\multimodal_encoder\blocks.py to fix the output of the function unpad_input caused by the flash-attn version 2.7.3.
2. change the "temporal_aggregator_type": "pmv5" in config.json in python, this is caused by renaming of the information aggregation module.
3. change the folder name starting with vidit with pave
3. install on the cluster
```
git clone https://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@github.com/dragonlzm/PAVE_test.git



module --force purge
module load gcc/12.3.0
module load anaconda/2020.11-py38

conda-env-mod create -n lla_repro -p /depot/schaterj/data/3d/conda-env/lla_repro -m /depot/schaterj/data/3d/modules --jupyter -y

module use /depot/schaterj/data/3d/modules
module load conda-env/lla_repro-py3.8.5

conda install python=3.10
git clone https://github.com/haotian-liu/LLaVA.git ?
cd PAVE_test ?
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install flash-attn==2.5.9.post1 --no-build-isolation --no-cache-dir
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

pip install peft==0.10.0
pip install rotary-embedding-torch
pip install icecream
pip install datasets

# The following are not needed
pip install av
pip install opencv-python
pip install triton==2.2.0 --no-cache-dir


module load cuda/12.1
module load cuda/11.2.0

pip install triton==2.2.0 --no-cache-dir
```



4. reset the huggingface home directory 
```
export HF_HOME 
HF_HOME='/video_datasets/zhuoming/huggingface' 
```

5. login huggingface directory 
```
huggingface-cli login
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

```

6. Transfer the data
```
scp -i gilbreth.pem wang4495@gilbreth.rcac.purdue.edu:/depot/schaterj/data/3d/work_dir/zhuoming_temp/checkpoint_spare/vidit_v5_1_3_lora_7B.tar /video_datasets/zhuoming/checkpoints/vidit_v5_1_3_lora_7B.tar 

scp -i gilbreth.pem wang4495@gilbreth.rcac.purdue.edu:/depot/schaterj/data/3d/work_dir/zhuoming_temp/checkpoint_spare/vidit_v5_1_2_lora_full.tar /video_datasets/zhuoming/checkpoints/vidit_v5_1_2_lora_full.tar 

scp -i gilbreth.pem wang4495@gilbreth.rcac.purdue.edu:/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/academic_source/NextQA/0023/6971535794.mp4 /video_datasets/zhuoming/checkpoints/6971535794.mp4 

scp -i gilbreth.pem wang4495@gilbreth.rcac.purdue.edu:/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/languagebind_feat/NextQA/0023/6971535794.pt /video_datasets/zhuoming/checkpoints/6971535794.pt 

cd /video_datasets/zhuoming/checkpoints
```

7. test scripts
```
ln -sf /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints /depot/schaterj/data/3d/work_dir/zhuoming_temp/PAVE_test/checkpoints
ln -sf /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data /depot/schaterj/data/3d/work_dir/zhuoming_temp/PAVE_test/data

export HF_HOME 
HF_HOME='/depot/schaterj/data/3d/work_dir/zhuoming_temp/huggingface' 

# test the inference
python demo_pave.py

# test the training
WANDB__SERVICE_WAIT=500 deepspeed --master_port 60000 train_pave_w_feat.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --annotation_path ./data/video_instruction_tuning/avsd/avsd_train_instruct.json  \
    --fast_path_mapping_path ./data/video_instruction_tuning/avsd/all_feats_mapping.json \
    --slow_path_mapping_path ./data/video_instruction_tuning/avsd/all_videos_mapping.json \
    --data_root ./data/video_instruction_tuning/avsd/Charades_v1_audio_imagebind_feat \
    --slow_path_data_root ./data/video_instruction_tuning/avsd/Charades_v1_480 \
    --use_fast_feat True \
    --use_slow True \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B \
    --output_dir ./checkpoints/testing \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --bf16 True \
    --tf32 True \
    --mm_newline_position grid \
    --mm_spatial_pool_mode bilinear \
    --feat_combine_method add \
    --fast_feat_type audio

```
