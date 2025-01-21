deepspeed --master_port 60000 train_pave_w_feat.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --annotation_path ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_oe_v0_1_qa_processed_2pv.json \
                      ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_oe_v0_1_qa_processed_2pv.json \
                      ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_oe_v0_1_qa_processed_2pv.json \
                      ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_oe_v0_1_qa_processed_2pv.json \
    --fast_path_mapping_path ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1_feat_mapping.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_v0_1_feat_mapping.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1_feat_mapping.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_v0_1_feat_mapping.json \
    --slow_path_mapping_path ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1_videos_mapping_updated.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_v0_1_videos_mapping_updated.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1_videos_mapping_updated.json \
                             ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_v0_1_videos_mapping_updated.json \
    --data_root ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/languagebind_feat \
                ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_v0_1/languagebind_feat \
                ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1/languagebind_feat \
                ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_v0_1/languagebind_feat \
    --slow_path_data_root ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_v0_1/academic_source_ds \
                          ./data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_v0_1/liwei_youtube_videos_ds \
                          ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_v0_1/academic_source_ds \
                          ./data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_v0_1/liwei_youtube_videos_ds \
    --use_fast_feat True \
    --use_slow True \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsV5_1_3_7B \
    --output_dir ./checkpoints/pave_v5_1_3_lora_7B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
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
    --fast_feat_type languagebind


