CUDA_VISIBLE_DEVICES=0 python eval_pave_music_avqa.py \
    --model-path ./checkpoints/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_3layers \
    --annotation-file ./data/video_instruction_tuning/music_avqa/updated_avqa-test.json \
    --video-folder ./data/video_instruction_tuning/music_avqa \
    --feature-folder ./data/video_instruction_tuning/music_avqa \
    --pred-save ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen \
    --feat-type imagebind


python tools/audio/music-avqa/calculate_acc.py --prediction-path ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.json \
--annotation-path ./data/video_instruction_tuning/music_avqa/updated_avqa-test.json