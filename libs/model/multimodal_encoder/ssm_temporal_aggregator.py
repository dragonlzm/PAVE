import math
import ipdb

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import VideoCrossAttentionWith3DRope, DecoderLayer, ResNetBasicStem, ResStage
from einops import rearrange, repeat


class LayerNorm2d(nn.LayerNorm):
    # Refer to ConvNext for the layernorm: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    # assume the input should be (B, C, H, W)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm3d(nn.LayerNorm):
    # Refer to ConvNext for the layernorm: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    # assume the input should be (B, C, T, H, W)
    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1) # (B, C, T, H, W) -> (B, T, H, W, C)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3) # (B, T, H, W, C) -> (B, C, T, H, W)
        return x


# this part is copy from Qwen2-VL
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class SSMTemporalAggregatorV5(nn.Module):
    '''
        this backbone contains: 
        1. Multiple conv blocks
        2. Cross-Attn layers to aggregate the information spatially
        3. MLP layer?

    '''
    def __init__(
        self,
        input_dim,                       # input feature dimension
        output_dim,                      # the output dimion
        embed_dim=None,
   
        fast_input_mapping_type='conv',  # conv based for VideoVAE feature, the linear based for languagebind
        number_of_input_mapping_layer=5, # for the conv layers
        number_of_block_each_layer=[1, 2, 2, 2, 2],
        sptial_stride=[1, 1, 1, 1, 1],
        dim_scaling=[1, 1, 1, 1, 1],
   
        query_type='slow_feat',
        chunks_number=32,             # for using learnable tokens
        number_of_query=196,     
        query_input_dim=896,          # for using slow as query: the slow feature dimension
        cross_attn_hidden_dim=512,    # cross-attn hyper, the hidden dimension of cross-attn
        num_cross_attn_head=4,        
        num_cross_attn_layer=1,
        use_3d_rope=True,
        
        use_output_mlp=True,
        use_dropout=True,              # other configs
        dropout_rate=0.1,
        use_output_norm=False,
        train_addition_start_end_tokens=False,
        mlp_depth=2,
        use_slow_feat_before_mlp=False
        
    ):
        super().__init__()
        # assert len(backbone_arch) == 3
        # Hyper-params 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_depth = mlp_depth
        self.act = nn.SiLU(inplace=True)
        self.use_slow_feat_before_mlp = use_slow_feat_before_mlp
        
        self.fast_input_mapping_type = fast_input_mapping_type
        if self.fast_input_mapping_type == 'conv':
            # Init the convolution (This will downsample the spatial 2 by 2,
            # and increase the dimension by 4 to 16)
            self.input_mapping = nn.ModuleList()
            self.embed_dim = embed_dim if embed_dim is not None else input_dim * 4
            curr_dim = input_dim
            for i in range(number_of_input_mapping_layer):
                curr_spatial_stride = sptial_stride[i]
                curr_dim_scaling = dim_scaling[i]
                curr_stage_block_num = number_of_block_each_layer[i]
                if i == 0:
                    # init the layer
                    self.input_mapping.append(
                        ResNetBasicStem(curr_dim,
                                        self.embed_dim,
                                        kernel = [1, 1, 1],
                                        stride = [1, 1, 1],
                                        padding = [0, 0, 0],
                                        with_pooling=False,
                                        norm_module=LayerNorm3d,
                                        )
                    )
                    curr_dim = self.embed_dim
                else:
                    self.input_mapping.append(
                        ResStage(dim_in=curr_dim,
                                    dim_out=curr_dim * curr_dim_scaling,
                                    stride=curr_spatial_stride, # if i < number_of_input_mapping_layer-1 else 2
                                    temp_kernel_size=3,
                                    num_blocks=curr_stage_block_num,
                                    dim_inner=curr_dim,
                                    num_groups=1,
                                    num_block_temp_kernel=curr_stage_block_num,
                                    dilation=1 if i < number_of_input_mapping_layer-1 else 2, # use 2 for hte last layer
                                    trans_func_name="bottleneck_transform",
                                    norm_module=LayerNorm3d,
                                    )
                    )

                    curr_dim *= curr_dim_scaling    
            input_mapping_out_channels = curr_dim
        elif self.fast_input_mapping_type == 'linear':
            assert embed_dim is not None
            self.embed_dim = embed_dim
            self.input_mapping = nn.Linear(self.input_dim, self.embed_dim)
            input_mapping_out_channels = self.embed_dim
        else:
            raise NotImplementedError
        
        self.use_slow_as_query = True
        self.query_input_dim = query_input_dim
        self.cross_attn_hidden_dim = cross_attn_hidden_dim
        
        # define the input mapping of the query /
        self.query_type = query_type
        if self.query_type == 'slow_feat':
            self.query_input_mapping = nn.Linear(self.query_input_dim, self.cross_attn_hidden_dim)
        elif self.query_type == 'learnable':
            # Since we add 3d poe, therefore, we should not repeat the query tokens here
            self.learnable_query = nn.Parameter(torch.randn(chunks_number, number_of_query, cross_attn_hidden_dim))
        else:
            raise NotImplementedError
        
        # a transformer layer to reduce the token of each trunk to 196
        self.use_3d_rope = use_3d_rope
        if self.use_3d_rope:
            self.temporal_embedding = Qwen2RotaryEmbedding(self.cross_attn_hidden_dim // num_cross_attn_head)
        else:
            self.temporal_embedding = None
            
        self.cross_attn_layers = nn.ModuleList()
        for i in range(num_cross_attn_layer):
            self.cross_attn_layers.append(
                DecoderLayer(self.cross_attn_hidden_dim,
                            input_mapping_out_channels,
                            self.cross_attn_hidden_dim, 
                            num_cross_attn_head, 
                            dim_feedforward=2*self.cross_attn_hidden_dim, 
                            dropout=0.0,
                            decoder_func=VideoCrossAttentionWith3DRope)
            )        

        # Init the MLP
        self.use_output_mlp = use_output_mlp
        if self.use_output_mlp:
            mlp_input_dim = self.cross_attn_hidden_dim
            mlp_hidden_dim = mlp_input_dim * 2
            modules = [nn.Linear(mlp_input_dim, mlp_hidden_dim)]
            for _ in range(1, self.mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(mlp_hidden_dim, self.output_dim))
            self.output_mapping = nn.Sequential(*modules)
        else:
            self.output_mapping = None
            # additional assertion
            assert self.cross_attn_hidden_dim == output_dim
        
        # Init the dropout layer
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.output_dropout = nn.Dropout(p=dropout_rate)
        else:
            self.output_dropout = None
        # Init the layernorm layer 
        self.use_output_norm = use_output_norm
        if self.use_output_norm:
            self.output_norm = nn.LayerNorm(self.output_dim)
        else:
            self.output_norm = None
        
        self.module_dtype = None
        
        # init the train_addition_start_end_tokens
        self.train_addition_start_end_tokens = train_addition_start_end_tokens
        if self.train_addition_start_end_tokens:
            # train additional 4 tokens to represent the start and the end of the video and image feature
            # self.start_end_tokens[0]: video start
            # self.start_end_tokens[1]: video end 
            # self.start_end_tokens[2]: image start
            # self.start_end_tokens[3]: image end
            self.start_end_tokens = nn.Parameter(torch.randn(4, self.output_dim))
        else:
            self.start_end_tokens = None

        # initialize the weight
        self.apply(self.__init_weights__)
        
        # init the self.output_norm gamma to zero
        if self.output_norm is not None:
            nn.init.zeros_(self.output_norm.weight)
            nn.init.zeros_(self.output_norm.bias)
        

    def __init_weights__(self, module,):
        # we have check this following initialization will not affact the init in SSM
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
    # @property
    # def device(self):
    #     # a hacky way to get the device type
    #     # will throw an error if parameters are on different devices        
    #     if self.device is None:
    #         self.device = list(set(p.device for p in self.parameters()))[0]

    #     return self.device 

    @property
    def dtype(self):
        # a hacky way to get the dtype
        if self.module_dtype is None:
            self.module_dtype = list(set(p.dtype for p in self.parameters()))[0]

        return self.module_dtype
    
    def forward(self, x, frame_num, 
                q_text_embeds=None, 
                q_text_nums=None, 
                chunk_num=None,
                slow_feats=None):
        # ipdb.set_trace()
        # x: (B, T, H, W, C)
        # frame_num: (B, ) it represent the length of the mask
        # slow_feats : list of tensor [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
        
        # This version of implementation will be: 
        # (B, T, H, W, 4) torch.Size([4, 38, 28, 28, 4]) -conv-> torch.Size([4, 38, 28, 28, 256]) -res*3-> torch.Size([4, 38, 28, 28, 256]) -res*1-> torch.Size([4, 38, 14, 14, 256])
        
        # Change the datatype
        x = x.to(self.dtype)
        if slow_feats is not None:
            if isinstance(slow_feats, list):
                slow_feats = [ele.to(self.dtype) for ele in slow_feats]
            else:
                slow_feats = slow_feats.to(self.dtype)
        
        all_features = []
        # transformer to convert each chunks to fixed number of tokens
        # for each sample split the size
        for curr_feat, curr_query, curr_len in zip(x, slow_feats, frame_num):
            # prepare the query
            # ipdb.set_trace() # check the slow and fast feature, check the query
            if self.query_type == 'slow_feat':
                curr_query = self.query_input_mapping(curr_query) # (T, S, C)
            elif self.query_type == 'learnable':
                # ipdb.set_trace()
                curr_query = self.learnable_query # (T, S, C)
            else:
                raise NotImplementedError
            # update the feature
            curr_feat = curr_feat[:curr_len] # (T, H/2, W/2, C*4) torch.Size([225, 28, 28, 4]), curr_len:184
            
            if self.fast_input_mapping_type == 'conv':
                curr_feat = rearrange(curr_feat, "t h w c -> c t h w")
                curr_feat = curr_feat.unsqueeze(dim=0)
                for layer in self.input_mapping: # [1, 4, 53, 28, 28] -> [1, 256, 53, 28, 28] -> [1, 256, 53, 28, 28] -> [1, 256, 53, 28, 28] -> [1, 512, 53, 14, 14]
                    curr_feat = layer(curr_feat)
            else:
                curr_feat = self.input_mapping(curr_feat)
                curr_feat = rearrange(curr_feat, "t h w c -> c t h w")
                curr_feat = curr_feat.unsqueeze(dim=0) # keep the dimension of 1, c, t, h, w
                
            # the output here should be [B, C, T, H, W], do the upsampling if the len of the feature is smaller than chunk_num
            T = curr_feat.shape[2]
            if T < chunk_num:
                print('current feat T: ', T, ' scale to: ', chunk_num)
                temporal_scale_factor = chunk_num / T
                upsampling = nn.Upsample(scale_factor=(temporal_scale_factor, 1, 1), mode='trilinear', align_corners=True)
                curr_feat = upsampling(curr_feat)
                # curr_len = curr_feat.shape[2]
            
            # permute the dimension back
            # ipdb.set_trace() # check the dimension
            # curr_feat = curr_feat.squeeze(dim=0)   # 1, c, t, h, w
            curr_feat = rearrange(curr_feat, "b c t h w -> b t (h w) c")

            # ipdb.set_trace() # check the dimension change
            # cross-attn
            output = curr_query.unsqueeze(dim=0)  # T_q, C, D torch.Size([32, 196, 512]) -> 1, T_q, C, D 
            for layer in self.cross_attn_layers:
                output = layer(output,       # should be in 1, T_q, C, D 
                    curr_feat,               # should be in b t (h w) c
                    memory_mask=None, 
                    key_temporal_pos=self.temporal_embedding,
                    rope_axis='spatial')     
            
            # reshape output back to normal
            # ipdb.set_trace() # check the view whether in the right order
            output = output.view(-1, output.shape[-1])
            output = output.unsqueeze(dim=0)
            all_features.append(output)
        
        # ipdb.set_trace()
        output = torch.cat(all_features, dim=0)
        # a linear layer to convert the tokens to output dimension
        if self.output_mapping is not None:
            output = self.output_mapping(output)
        # add a dropout layers
        if self.output_dropout is not None:
            output = self.output_dropout(output)
        if self.output_norm is not None:
            # ipdb.set_trace() # check the norm layer init, check the dropout is Nones
            output = self.output_norm(output)

        return output