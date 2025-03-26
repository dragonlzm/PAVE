CUDA_VISIBLE_DEVICES=0 python eval_pave_avqa.py \
    --model-path ./checkpoints/pave_v5_1_3_lora_avqa_7B_imagebind_2epoch \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B \
    --annotation-file ./data/video_instruction_tuning/avqa/val_qa.json \
    --video-folder ./data/video_instruction_tuning/avqa/avqa_subset \
    --feature-folder ./data/video_instruction_tuning/avqa/avqa_subset_audio_imagebind_feat \
    --pred-save ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avqa_7B_imagebind_2epoch.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

python tools/audio/avqa/calculate_acc.py --prediction-path ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avqa_7B_imagebind_2epoch.json