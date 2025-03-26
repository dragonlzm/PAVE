CUDA_VISIBLE_DEVICES=0 python eval_pave_egoexo.py \
    --model-path ./checkpoints/pave_egoexo_v5_1_3_3d_lora_7B_2epoch \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_egoexo_7B \
    --annotation-file ./data/video_instruction_tuning/egoexo_origin/annotations/proficiency_demonstrator_val.json \
    --video-folder ./data/video_instruction_tuning/egoexo_origin \
    --feature-folder ./data/video_instruction_tuning/egoexo_origin/sigclip_feature \
    --pred-save ./data/video_instruction_tuning/prediction/pave_egoexo_v5_1_3_3d_lora_7B_2epoch.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen


python tools/egoexo/calculate_egoexo_accuracy.py --prediction-path ./data/video_instruction_tuning/prediction/pave_egoexo_v5_1_3_3d_lora_7B_2epoch.json
