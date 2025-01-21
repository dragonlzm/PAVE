# 🛠️ PAVE: Patching and Adapting Video Large Language Models

## Contents
- [Install](#install)
- [PAVE Weights](#pave-weights)
- [Datasets](#datasets)
- [Demo](#Demo)
- [Train](#train)
- [Evaluation](#evaluation)


## Install

We install this environment on Linux machine:
1. Clone the repository and navigate to PAVE folder 
```
git clone https://github.com/dragonlzm/PAVE.git
cd PAVE
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
We includes all the PAVE weights for different tasks in this section.
### 1. 3D-QA

| Dataset | Base Model | Schedule | Checkpoint | ScanQA (C) | ScanQA (B-4) | ScanQA (M) | ScanQA (R) | ScanQA (EM@1) | SQA3D (EM@1)
|----------|----------|-----------|-----------|---|---|---|---|---|---|
| ScanQA | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 1e | [pave_scanqa](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_scanqa_v5_1_3_3d_lora.zip) |  84.2 | 13.1 | 17.0 | 42.1 | 23.1 (40.0) | - |
| ScanQA | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 1e | [pave_scanqa_7B](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_scanqa_v5_1_3_3d_lora_7B.zip) | 103.4 | 16.0 | 19.9 | 49.0 | 29.1 (48.5) | - | 
| SQA3D  | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 2e | Coming Soon | - | - | - | - | - | ? | 
| SQA3D  | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 2e | [pave_sqa3d_7B](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_sqa3d_v5_1_3_3d_lora_7B_2epoch.zip) | - | - | - | - | - | 59.0 (61.4) | 

### 2. Audio-Visual

| Dataset | Base Model | Schedule | Checkpoint | AVSD (CIDEr) | Music-AVQA (Audio Acc) | Music-AVQA (Visual Acc) | Music-AVQA (Audio-Visual Acc) | Music-AVQA (Overall Acc) |
|----------|----------|-----------|-----------|---|---|---|---|---|
| AVSD | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 1e | Coming soon | ? | - | - | - | - |
| AVSD | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 1e | [pave_avsd_7B_imagebind](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_5_1_3_lora_avsd_7B_imagebind.zip) | 152.9 | - | - | - | - | 
| Music-AVQA | [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 2e | Coming Soon | - | ? | ? | ? | ? |
| Music-AVQA | [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 2e | [pave_music_avqa_7B_imagebind](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.zip) | - | 79.7 | 93.0 | 78.0 | 82.3 |


### 3. Enhancing Video QA
| Base Model | Schedule | Checkpoint | VideoMME (Short) | VideoMME (Long) | VideoMME (Visual Acc) | MVBench |
|----------|-----------|-----------|---|---|---|---|
| [LLaVA-OneVision-0.5B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov) | 1e | Coming Soon | 57.8 | 42.7 | 37.4 | 46.0 | 46.6 |
| [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)     | 1e | [pave_v5_1_3_lora_7B](https://huggingface.co/zhuomingliu/PAVE/blob/main/pave_v5_1_3_lora_7B.zip) | 71.1 | 59.4 | 49.2 | 59.9 | 58.0 | 



## Datasets
We include the instructions for preparing different datasets for the PAVE training and evaluation in this section.

### 1. ScanQA
Please refer to [ScanQA_Prepare](./doc/scanqa_dataset_prep.md) for more information.

### 2. SQA3D
Please refer to [SQA3D_Prepare](./doc/sqa3d_dataset_prep.md) for more information.

### 3. AVSD
Please refer to [AVSD_Prepare](./doc/avsd_dataset_prep.md) for more information.

### 4. Music-AVQA
Please refer to [Music-AVQA_Prepare](./doc/music_avqa_dataset_prep.md) for more information.

### 5. LLaVA-Video
Please refer to [LLaVA-Video_Prepare](./doc/llava_video_dataset_prep.md) for more information.

## Demo
Coming Soon.

## Train
We provide the sample training scripts in this section.
### 1. ScanQA
Please refer to [ScanQA_Train](./scripts/scanqa_train.sh) for more information.

### 2. SQA3D
Please refer to [SQA3D_Train](./scripts/sqa3d_train.sh) for more information.

### 3. AVSD
Please refer to [AVSD_Train](./scripts/avsd_train.sh) for more information.

### 4. Music-AVQA
Please refer to [Music-AVQA_Train](./scripts/music_avqa_train.sh) for more information.

### 5. Enhanced Video
Please refer to [Enhanced_video_Train](./scripts/enhanced_video_train.sh) for more information.


## Evaluation
We provide the sample evaluation scripts in this section.

### 1. ScanQA
Please refer to [ScanQA_Eval](./scripts/scanqa_eval.sh) for more information.

### 2. SQA3D
Please refer to [SQA3D_Eval](./scripts/sqa3d_eval.sh) for more information.

### 3. AVSD
Please refer to [AVSD_Eval](./scripts/avsd_eval.sh) for more information.

### 4. Music-AVQA
Please refer to [Music-AVQA_Eval](./scripts/music_avqa_eval.sh) for more information.

### 5. LLaVA-Video
Please refer to [LLaVA-Video_Eval](./scripts/enhanced_video_eval.sh) for more information.
