CUDA_VISIBLE_DEVICES=0 python eval_pave_sqa3d.py \
    --model-path ./checkpoints/pave_sqa3d_v5_1_3_3d_lora_7B_2epoch \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_3d_7B \
    --question-file ./data/video_instruction_tuning/scannet/sqa3q_ScanQA_format/llava3d_sqa3d_test_question.json \
    --video-folder ./data/video_instruction_tuning/scannet/posed_images_new \
    --feature-folder ./data/video_instruction_tuning/scannet/video_features_new \
    --pred-save ./data/video_instruction_tuning/prediction/pave_sqa3d_v5_1_3_3d_lora_7B_2epoch.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

python tools/3d/sqa3d/sqa3d_evaluator.py \
    --gt-file ./data/video_instruction_tuning/scannet/sqa3q_ScanQA_format/llava3d_sqa3d_test_answer.json \
    --results-file ./data/video_instruction_tuning/prediction/pave_sqa3d_v5_1_3_3d_lora_7B_2epoch.json