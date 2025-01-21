CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=8 \
    lmms_eval_start.py \
    --model pave \
    --model_args model_path=./checkpoints/pave_v5_1_3_lora_7B/,model_base=lmms-lab/llava-onevision-qwen2-7b-ov,model_arg_name=VideoFeatModelArgumentsV5_1_3_7B,conv_template=qwen_1_5,fast_feat_type=languagebind,slow_feat_type=raw_video \
    --tasks videomme_feature \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vidit_videomme \
    --output_path ./logs/