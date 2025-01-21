CUDA_VISIBLE_DEVICES=0 python eval_pave_music_avqa.py \
    --model-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/checkpoints/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B_3layers \
    --annotation-file /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/music_avqa/updated_avqa-test.json \
    --video-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/music_avqa \
    --feature-folder /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/music_avqa \
    --pred-save /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen \
    --feat-type imagebind


python tools/prepare_audio/MUSIC_AVQA/calculate_acc.py --prediction-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/pave_v5_1_3_lora_music_avqa7B_2epoch_imagebind_3layers.json \
--annotation-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/music_avqa/updated_avqa-test.json