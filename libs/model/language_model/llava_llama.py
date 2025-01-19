#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        ### The input of the function: 
        # input_ids:           torch.Size([BS, Seq_len]) torch.Size([4, 311]) tensor([    1,   319, 13563,  1546,   263, 12758,  1404,   322,   385, 23116, ..., 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
        # attention_mask:      torch.Size([BS, Seq_len]) torch.Size([4, 311]) tensor([True, True, True, True, True, True, True, True, True, True, True, True,..., False, False, False, False, False, False], device='cuda:0')
        # position_ids:        None
        # past_key_values:     None
        # inputs_embeds:       None
        # labels:              torch.Size([BS, Seq_len]) torch.Size([4, 311]) This is the same as the 'targets' in preprocess (libs\dataset\image_dataset.py) tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, ..., 450,  7945,   297, 278,  1967,   338,  4796,   322,  4628, 29889, ..., -100,  -100,  -100, 901, 15075,   475,   519,  970,  8608,   362,  6757, 29889,     2], device='cuda:0')   
        # use_cache:           None
        # output_attentions:   None     
        # output_hidden_states:None   
        # images:              torch.Size([BS, C, H, W]) torch.Size([4, 3, 224, 224])
        # image_sizes:         None
        # return_dict:         None
        
        ### the output of the prepare_inputs_labels_for_multimodal (concated with the video feature) 
        # New_seq_len = Seq_len + Num_of_img_token - 1 = 311 + 256 - 1 = 566
        # position_ids:       None
        # attention_mask:     torch.Size([BS, New_seq_len]) torch.Size([4, 566]) paded for the batch Tensor([True, True, True, True, True, True, True, True, True, True, True, True,..., False, False, False, False, False, False], device='cuda:0')
        # past_key_values:    None
        # inputs_embeds:      torch.Size([BS, New_seq_len, EmbedDim]) torch.Size([4, 566, 4096])
        # labels:             torch.Size([BS, New_seq_len]) torch.Size([4, 566]) This is padding 256 -100 for the image tokens tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, ..., 450,  7945,   297, 278,  1967,   338,  4796,   322,  4628, 29889, ..., -100,  -100,  -100, 901, 15075,   475,   519,  970,  8608,   362,  6757, 29889,     2], device='cuda:0')   
        # use_cache:          None
        # output_attentions:  None
        # return_dict:        None

        if inputs_embeds is None:
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
                images,
                image_sizes
            )

        return super().forward(
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

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
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
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
