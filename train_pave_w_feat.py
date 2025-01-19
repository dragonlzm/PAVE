# Adopted from https://github.com/haotian-liu/LLaVA. 
import os
import pathlib
import torch
import transformers
import ipdb
import sys

from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libs.utils.model_trainer import VideoModelTrainer
from libs import conversation_lib as conversation_lib
from libs.model import *
from libs.utils.train_utils import parse_argument_classes, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_videotrainer, get_peft_state_non_lora_maybe_zero_3_with_state_dict

from libs.dataset.base_dataset import make_video_supervised_data_module
from utils import prepare_video_model

def train_pave_func(attn_implementation=None):
    global local_rank
    # check the model_class name, data_class name and the training_class name
    # remaining_args, model_arg_class, data_arg_class, training_arg_class = parse_argument_classes(sys.argv[1:])

    # parser = transformers.HfArgumentParser(
    #     (model_arg_class, data_arg_class, training_arg_class))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_args)
    
    model_args, data_args, training_args = parse_argument_classes(sys.argv[1:])
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # prepare the model
    model, tokenizer = prepare_video_model(training_args, model_args, data_args, compute_dtype, attn_implementation)

    # make the dataset and the trainer
    data_module = make_video_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = VideoModelTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    # TODO: Handle the resume of the training. 
    # HOWEVER, since the training just ONE epoch.
    # It may not reasonable to resume the training.
    # We should restart the training at the begining.
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        torch.use_deterministic_algorithms(False) # for the 3d pooling layer
        trainer.train()
    
    # save the state dict and the model after training
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable: # this is for step2 training
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_with_state_dict(
            model.state_dict(), special_key=['temporal_aggregator', 'self_attn.v_kv_proj', 'self_attn.gate_proj']
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        # this is for step 1 training
        safe_save_model_for_hf_videotrainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train_pave_func(attn_implementation="flash_attention_2")
