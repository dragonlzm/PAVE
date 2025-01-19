from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from libs.model.vidit_arch import ViditMetaModel, ViditMetaForCausalLM


class ViditQwen2Config(Qwen2Config):
    model_type = "vidit_qwen2"


class ViditQwen2Model(ViditMetaModel, Qwen2Model):
    config_class = ViditQwen2Config

    def __init__(self, config: Qwen2Config):
        super(ViditQwen2Model, self).__init__(config)
        

class ViditQwen2ForCausalLM(Qwen2ForCausalLM, ViditMetaForCausalLM):
    config_class = ViditQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ViditQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        video_feats: Optional[torch.FloatTensor] = None,
        video_feat_fps: Optional[torch.FloatTensor] = None,
        feat_frame_nums: Optional[torch.FloatTensor] = None,
        
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["video"],
        
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        video_metas = None,
        
        question_ids = None,
        question_lens = None,
        dpo_forward = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace() # check the bug
        if inputs_embeds is None: # this is the training or the first forward at inference
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                video_feats,
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                video_metas=video_metas,
            )

        if not dpo_forward:
            loss = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            if loss.loss is not None and torch.isnan(loss.loss):
                print('the NAN Loss batch video info:', video_metas)
                import time
                torch.save(video_metas, 'error_' + str(time.time()) + '.pt')
                raise NotImplementedError
            
            return loss
        else: # dpo forward
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict            
                
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels            

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        video_feats: Optional[torch.Tensor] = None,
        video_feat_fps: Optional[torch.FloatTensor] = None,
        feat_frame_nums: Optional[torch.FloatTensor] = None,

        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["video"],        
        
        question_ids = None,
        question_lens = None,    
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # import ipdb
        # ipdb.set_trace()
        if video_feats is not None or images is not None: # if there is vision input
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                video_feats,
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.half(),
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        ## TODO: which function will call this function? What should we change here? 
        videos = kwargs.pop("videos", None)
        video_sizes = kwargs.pop("video_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if videos is not None:
            inputs['videos'] = videos
        if video_sizes is not None:
            inputs['video_sizes'] = video_sizes
        return inputs

AutoConfig.register("vidit_qwen2", ViditQwen2Config)
AutoModelForCausalLM.register(ViditQwen2Config, ViditQwen2ForCausalLM)
