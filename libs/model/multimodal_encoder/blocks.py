# This script hold all the implementation of transformer or ResNet blocks.

import copy
import ipdb
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import math

from libs.mm_utils import split_list_lengths

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input
except ImportError:
    flash_attn_varlen_func = None


def FeedForward(dim, dim_feedforward):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim_feedforward, bias = False),
        nn.GELU(),
        nn.Linear(dim_feedforward, dim, bias = False)
    )


def create_idx_cube(T, H, W):
    # create the T index
    t_values = torch.arange(T).unsqueeze(1).unsqueeze(2)
    # Broadcast the depth values across the height and width dimensions
    t_index = t_values.expand(T, H, W)
    
    # create the H index
    h_values = torch.arange(H).unsqueeze(0).unsqueeze(2)
    h_index = h_values.expand(T, H, W)
    
    # create the W index
    w_values = torch.arange(W).unsqueeze(0).unsqueeze(0)
    w_index = w_values.expand(T, H, W)
    
    # create a T, H, W, 3 cube
    final_idx = torch.stack([t_index, h_index, w_index], dim=-1)
    return final_idx

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(input, cos, sin, position_ids, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).
        This function is base modified base on the function from the Qwen2-VL
    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.                          torch.Size([1, 4, 985, 128])
        k (`torch.Tensor`): The key tensor.                            torch.Size([1, 4, 985, 128])
        cos (`torch.Tensor`): The cosine part of the rotary embedding. torch.Size([985, 128])
        sin (`torch.Tensor`): The sine part of the rotary embedding.   torch.Size([985, 128])
        position_ids (`torch.Tensor`):                                 torch.Size([3, 1, 985])
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):                                    [16, 24, 24]
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # select poe for each dimension, since we have 3 position_ids
    
    cos = cos[position_ids] # torch.Size([3, 1, 985, 128])
    sin = sin[position_ids] # torch.Size([3, 1, 985, 128])
    mrope_section = mrope_section * 2 # [16, 24, 24] -> [16, 24, 24, 16, 24, 24]
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    ) # torch.Size([1, 1, 985, 128]) target shape in qwen vl
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    ) # torch.Size([1, 1, 985, 128]) target shape in qwen vl

    input_embed = (input * cos) + (rotate_half(input) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return input_embed


class VideoCrossAttentionWith3DRope(nn.Module):
    '''
        This version aims to handle the cross-attn between the query and the video feature
    '''
    
    def __init__(self, d_query, d_input, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_query == d_model

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_query, d_model)
        self.k_linear = nn.Linear(d_input, d_model)
        self.v_linear = nn.Linear(d_input, d_model)
        self.attn_drop = attn_drop
        self.rope_scaling = {}
        self.rope_scaling["mrope_section"] = [16, 24, 24]

    def forward(self, q, k, v, attn_mask=None, rope=None, rope_axis='time'):
        # q:    (B, T_q, S_q, C) should be the query tokens
        # k:    (B, T_k, S_k, C) should be the video tokens
        # v:    (B, T_k, S_k, C) should be the video tokens
        # attn_mask: is list of the number which indicate the length of the  or value
        # rope: rotary positional embedding
        
        # ipdb.set_trace()
        Bq, T_q, S_q, C = q.shape
        Bk, T_k, S_k, C_cond = v.shape
        Bv, T_k, S_k, C_cond = k.shape
        assert Bq == Bk == Bv == 1 # for current version, we only support bs = 1
        
        q = self.q_linear(q).view(Bq, T_q*S_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_q*S_q, H_dim)
        
        k = self.k_linear(k).view(Bk, T_k*S_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_k*S_k, H_dim)
        v = self.k_linear(v).view(Bk, T_k*S_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, T_k*S_k, H_dim)
        
        # add the rotary pos embedding
        
        if rope is not None: # expect in the format of # (batch, heads, seq len, dimension of head)
            # compute the max size 
            T_max = max(T_q, T_k)
            assert int(math.sqrt(S_q)) ** 2 == S_q # In here we assume we alway use the square
            assert int(math.sqrt(S_k)) ** 2 == S_k # In here we assume we alway use the square
            # assert S_q >= S_k
            H_max = int(math.sqrt(max(S_q, S_k)))
            W_max = H_max
            # create the poe
            cos, sin = rope(q, seq_len=max(T_max, H_max, W_max))
            
            # create a large cube for each dimension
            idx_cube = create_idx_cube(T_max, H_max, W_max).to(q.device) #  (T_max, H_max, W_max, 3) cube
            
            # select the index for slow (the query)
            # determine the temporal idx of the slow frames
            temporal_idx = np.linspace(0, T_max-1, T_q, dtype=int).tolist()
            # directly select from the cube along the temporal axis
            slow_frames_idx = idx_cube[temporal_idx]  # T_q, h_q, w_q, 3
            if S_q < S_k:
                H_q = int(math.sqrt(S_q))
                target_split = 2*H_q+1
                H_fast_index = np.linspace(0, H_max, target_split, dtype=int).tolist()
                H_fast_index = H_fast_index[1:-1]
                H_fast_index = H_fast_index[::2]     
                W_fast_index = H_fast_index   # TODO: we are assume the input shape of is squre
                slow_frames_idx = slow_frames_idx[:, H_fast_index][:,:, W_fast_index] # T_k, h_k, w_k, 3 
            # ipdb.set_trace() # check the slow frame idx
            
            # determine the spatial idx of the fast frames (the key)
            H_k = int(math.sqrt(S_k))
            target_split = 2*H_k+1
            H_fast_index = np.linspace(0, H_max, target_split, dtype=int).tolist()
            H_fast_index = H_fast_index[1:-1]
            H_fast_index = H_fast_index[::2]     
            W_fast_index = H_fast_index   # TODO: we are assume the input shape of is squre
            # directly select from the cube along the temporal axis
            fast_frames_idx = idx_cube[:, H_fast_index][:,:, W_fast_index] # T_k, h_k, w_k, 3
            
            # flatten the dimension to 1d (3, B, len)
            fast_frames_idx = fast_frames_idx.view(-1, 3).permute([1, 0]).unsqueeze(dim=1) # which is the key
            slow_frames_idx = slow_frames_idx.view(-1, 3).permute([1, 0]).unsqueeze(dim=1) # which is the query
            
            # ipdb.set_trace() # check the fast frame idx
            # apply it to q
            q = apply_multimodal_rotary_pos_emb(
                q,  # torch.Size([1, 4, 6272, 128]) (B, Head_num, sequence length, head_dim)
                cos, sin, 
                slow_frames_idx,  # torch.Size([3, 1, 6272]) #
                self.rope_scaling["mrope_section"] # [16, 24, 24]
            )
            # apply it to k
            k = apply_multimodal_rotary_pos_emb(
                k,  # torch.Size([1, 28, 32144, 128]) (B, Head_num, sequence length, head_dim)
                cos, sin, 
                fast_frames_idx,  # torch.Size([3, 1, 32144]) #
                self.rope_scaling["mrope_section"] # [16, 24, 24]
            )
        
        # if do not need to do the padding
        # ipdb.set_trace() # check the audio reshaping
        if T_k % T_q == 0:
            # reshape 
            q = rearrange(q, 'B H (T S) D -> (B T) H S D', T=T_q)
            k = rearrange(k, 'B H (T S) D -> B H T S D', T=T_k)
            k = rearrange(k, 'B H (n T) S D -> (B n) H (T S) D', n=T_q)
            v = rearrange(v, 'B H (T S) D -> B H T S D', T=T_k)
            v = rearrange(v, 'B H (n T) S D -> (B n) H (T S) D', n=T_q)            
            # additional param for mask
            attn_mask = None
            # T = max_len_of_all_chunks
            # q_len = S_q
            
        else: # do the padding for the key if need, and convert it back to batch
            q = rearrange(q, 'B H (T S) D -> (B T) H S D', T=T_q)
            k = rearrange(k, 'B H (T S) D -> B H T S D', T=T_k)  
            v = rearrange(v, 'B H (T S) D -> B H T S D', T=T_k)  
            
            # find the cutting point 
            video_feat_split_sizes = split_list_lengths(T_k, T_q)
            # split the video into multiple chunks
            splited_k = torch.split(k, video_feat_split_sizes, dim=2) # (B, Head_num, T, S, head_dim)
            splited_v = torch.split(v, video_feat_split_sizes, dim=2) # (B, Head_num, T, S, head_dim)
            
            # (T, H/2, W/2, C*4) -> (32, T/32, H/2, W/2, C*4)
            # ipdb.set_trace() # check the split, check the padding, 
            _, H_num, _, S, C = splited_k[0].shape
            num_of_chunks = len(video_feat_split_sizes)
            max_len_of_all_chunks = max(video_feat_split_sizes)
            
            k_padded_chunks = torch.zeros(num_of_chunks, H_num, max_len_of_all_chunks, S, C).to(q.device, dtype=q.dtype) # B, H, Max_len, S, C
            v_padded_chunks = torch.zeros(num_of_chunks, H_num, max_len_of_all_chunks, S, C).to(q.device, dtype=q.dtype) # B, H, Max_len, S, C
            for i, (len_of_chunk, curr_k, curr_v) in enumerate(zip(video_feat_split_sizes, splited_k, splited_v)):
                k_padded_chunks[i, :, :len_of_chunk] = curr_k 
                v_padded_chunks[i, :, :len_of_chunk] = curr_v
            k_padded_chunks = rearrange(k_padded_chunks, "B H L S C -> B H (L S) C")  # flatten the spatial with the chunk len
            v_padded_chunks = rearrange(v_padded_chunks, "B H L S C -> B H (L S) C")  # flatten the spatial with the chunk len
            
            k = k_padded_chunks
            v = v_padded_chunks
            # ipdb.set_trace() # check the split, check the padding, 
            # additional param for mask
            attn_mask = torch.tensor(video_feat_split_sizes).to(k_padded_chunks.device)
            T = max_len_of_all_chunks
            # q_len = S_q

        # The needed input shape to the dot-attn is 
        # q: (B, H_num, S_q, H_dim)
        # k: (B, H_num, (t*s), H_dim)
        # v: (B, H_num, (t*s), H_dim)

        # ipdb.set_trace() # test the video cross-attn
        if not self.training or flash_attn_varlen_func is None: # if no flash attention
            if attn_mask is not None:
                assert len(attn_mask.shape) == 1 # the input is the len
                assert T == max(attn_mask)
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone().to(q.device)
                spatial_attn_mask *= S # multiply with the spatial dimension
                max_len = max(spatial_attn_mask)
                # spatial_attn_mask = torch.tensor(spatial_attn_mask)
                #ipdb.set_trace()
                spatial_attn_mask = torch.arange(max_len).expand(len(spatial_attn_mask), max_len).to(q.device) < spatial_attn_mask.unsqueeze(1) # (B, k_l)
                spatial_attn_mask = spatial_attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                spatial_attn_mask = spatial_attn_mask.repeat(1, 1, S_q, 1) # (B, 1, q_l, k_l)
                #ipdb.set_trace()
            else:
                spatial_attn_mask = None
            
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=spatial_attn_mask, dropout_p=self.attn_drop)
            # x = x.transpose(1, 2)
            # x = x.reshape(B, -1, C)
            x = rearrange(x, 'b h s d -> b s (h d)')
        else: # use flash attention           
            if attn_mask is not None: 
                assert len(attn_mask.shape) == 1 # the input is the len
                assert T == max(attn_mask)                
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone()
                spatial_attn_mask *= S # multiply with the spatial dimension

                # caclulate len of the mask
                k_max_len = max(spatial_attn_mask)
                # create the mask
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in spatial_attn_mask]).to(k.device)
                # handle the k and v
                # ipdb.set_trace()
                # here is minor update for compatible with the flash-attn-2.7.3
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k,_ = unpad_input(k.transpose(1, 2), kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v,_ = unpad_input(v.transpose(1, 2), kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                # handle the q (B, H_num, q_len, H_dim)
                cu_seqlens_q = torch.tensor([S_q * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = S_q
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else: # if attn mask is None, then do full cross-attn
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                # ipdb.set_trace()
                cu_seqlens_q = torch.tensor([q.shape[2] * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32)
                cu_seqlens_k = torch.tensor([k.shape[2] * i for i in range(k.shape[0]+1)]).to(k.device, dtype=torch.int32) # handles for the sptial dimenstion
                max_seqlen_q = q.shape[2]
                max_seqlen_k = k.shape[2] # handles for the sptial dimenstion
            
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            x = rearrange(x, '(t s) h d -> t s (h d)', t=T_q).unsqueeze(dim=0) # (B, T_q, S_q, C) should be the query tokens
        # ipdb.set_trace() # save the result
        return x    


class DecoderVideoCrossAttention(nn.Module):
    '''
        This version aims to handle the cross-attn between the query and the video feature
    '''
    
    def __init__(self, d_query, d_input, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_query == d_model

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_query, d_model)
        self.k_linear = nn.Linear(d_input, d_model)
        self.v_linear = nn.Linear(d_input, d_model)
        self.attn_drop = attn_drop

    def forward(self, q, k, v, attn_mask=None, rope=None, rope_axis='time'):
        # q:    (B, N_query, C) should be the query tokens
        # k:    (B, T, S, C) should be the video tokens
        # v:    (B, T, S, C) should be the video tokens
        # attn_mask: is list of the number which indicate the length of the  or value
        # rope: rotary positional embedding
        
        #ipdb.set_trace()
        B, q_len, C = q.shape
        Bk, T, S, C_cond = v.shape
        Bv, T, S, C_cond = k.shape
        assert B == Bk == Bv
        
        v = self.k_linear(v).view(B, T*S, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H_num, T*S, H_dim)
        q = self.q_linear(q) # (B, q_len, Total_H_dim)
        k = self.k_linear(k) # (B, T, S, Total_H_dim)
        
        # add the rotary pos embedding
        if rope is not None: # expect in the format of # (batch, heads, seq len, dimension of head)
            # ipdb.set_trace()
            q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim) # (B, q_len, Total_H_dim) -> (B, H_num, q_len, H_dim)
            q = rope.rotate_queries_or_keys(q)
            # ipdb.set_trace()
            if rope_axis == 'time':
                k = rearrange(k, "b t s (h d) -> (b s) h t d", h=self.num_heads, d=self.head_dim) # (b*s), h, t, head_dim
                k = rope.rotate_queries_or_keys(k)
                k = rearrange(k, "(b s) h t d -> b h (t s) d", s=S) # (B, H_num, (t*s), H_dim)
            elif rope_axis == 'spatial':
                k = rearrange(k, "b t s (h d) -> (b t) h s d", h=self.num_heads, d=self.head_dim) # (b*t), h, s, head_dim
                k = rope.rotate_queries_or_keys(k)
                k = rearrange(k, "(b t) h s d -> b h (t s) d", t=T) # (B, H_num, (t*s), H_dim)
            else:
                raise NotImplementedError
        else:
            q = q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, q_len, H_dim)
            k = k.view(B, T*S, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, (t*s), H_dim)
        
        # ipdb.set_trace() # test the video cross-attn
        if not self.training or flash_attn_varlen_func is None: # if no flash attention
            if attn_mask is not None:
                assert len(attn_mask.shape) == 1 # the input is the len
                assert T == max(attn_mask)
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone().to(q.device)
                spatial_attn_mask *= S # multiply with the spatial dimension
                max_len = max(spatial_attn_mask)
                # spatial_attn_mask = torch.tensor(spatial_attn_mask)
                #ipdb.set_trace()
                spatial_attn_mask = torch.arange(max_len).expand(len(spatial_attn_mask), max_len).to(q.device) < spatial_attn_mask.unsqueeze(1) # (B, k_l)
                spatial_attn_mask = spatial_attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                spatial_attn_mask = spatial_attn_mask.repeat(1, 1, q_len, 1) # (B, 1, q_l, k_l)
                #ipdb.set_trace()
            
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=spatial_attn_mask, dropout_p=self.attn_drop)
            # x = x.transpose(1, 2)
            # x = x.reshape(B, -1, C)
            x = rearrange(x, 'b h s d -> b s (h d)')
        else: # use flash attention           
            if attn_mask is not None: 
                assert len(attn_mask.shape) == 1 # the input is the len
                assert T == max(attn_mask)                
                # update the mask with the spatial (deep copy)
                spatial_attn_mask = attn_mask.clone()
                spatial_attn_mask *= S # multiply with the spatial dimension

                # caclulate len of the mask
                k_max_len = max(spatial_attn_mask)
                # create the mask
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in spatial_attn_mask]).to(k.device)
                # handle the k and v
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k.transpose(1, 2), kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v.transpose(1, 2), kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                # handle the q (B, H_num, q_len, H_dim)
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = q_len
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else: # if attn mask is None, then do full cross-attn
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)])
                cu_seqlens_k = torch.tensor([(T*S) * i for i in range(B+1)]) # handles for the sptial dimenstion
                max_seqlen_q = q_len
                max_seqlen_k = (T*S) # handles for the sptial dimenstion
            
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        # ipdb.set_trace() # save the result
        return x    


class DecoderTextCrossAttention(nn.Module):
    '''
        This version aims to handle the cross-attn between the query and the text feature
    '''
    
    def __init__(self, d_model, num_heads, attn_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.attn_drop = attn_drop

    def forward(self, q, k, v, attn_mask=None, rope=None):
        # q:    (B, N_query, C) should be the query tokens
        # k:    (B, N_text, C) should be the text tokens
        # v:    (B, N_text, C) should be the text tokens
        # attn_mask: is list of the number which indicate the length of the key
        # rope: rotary positional embedding
        
        #ipdb.set_trace()
        B, q_len, C = q.shape
        Bk, t_len, C_cond = v.shape
        Bv, t_len, C_cond = k.shape
        assert B == Bk == Bv
        
        v = self.k_linear(v).view(B, t_len, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H_num, t_len, H_dim)
        q = self.q_linear(q) # (B, q_len, Total_H_dim)
        k = self.k_linear(k) # (B, t_len, Total_H_dim)
        
        # add the rotary pos embedding
        if rope is not None: # expect in the format of # (batch, heads, seq len, dimension of head)
            # ipdb.set_trace()
            q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim) # (B, q_len, Total_H_dim) -> (B, H_num, q_len, H_dim)
            q = rope.rotate_queries_or_keys(q)
            # ipdb.set_trace()
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads, d=self.head_dim) # (B, t_len, Total_H_dim) -> (B, H_num, t_len, H_dim)
            k = rope.rotate_queries_or_keys(k)
        else:
            q = q.view(B, q_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, q_len, H_dim)
            k = k.view(B, t_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H_num, t_len, H_dim)
        
        # ipdb.set_trace() # test the text cross-attn
        if not self.training or flash_attn_varlen_func is None: # if no flash attention
            if attn_mask is not None:
                assert len(attn_mask.shape) == 1 # the input is the len
                assert t_len == max(attn_mask)
                # update the mask with the spatial (deep copy)
                attn_mask = attn_mask.clone()
                max_len = max(attn_mask)
                # attn_mask = torch.tensor(attn_mask)
                #ipdb.set_trace()
                attn_mask = torch.arange(max_len).expand(len(attn_mask), max_len).to(q.device) < attn_mask.unsqueeze(1) # (B, k_l)
                attn_mask = attn_mask.unsqueeze(dim=1).unsqueeze(dim=1)
                attn_mask = attn_mask.repeat(1, 1, q_len, 1) # (B, 1, q_l, k_l)
                #ipdb.set_trace()
            
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop)
            # x = x.transpose(1, 2)
            # x = x.reshape(B, -1, C)
            x = rearrange(x, 'b h s d -> b s (h d)')
        
        else: # use flash attention
            if attn_mask is not None: 
                assert len(attn_mask.shape) == 1 # the input is the len
                assert t_len == max(attn_mask)                
                # deep copy
                text_attn_mask = attn_mask.clone()
                # caclulate len of the mask
                k_max_len = max(text_attn_mask)
                # create the mask
                kv_padding_mask = torch.tensor([[True]*curr_seq_len + [False]*(k_max_len-curr_seq_len) for curr_seq_len in text_attn_mask]).to(k.device)
                # handle the k and v
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k.transpose(1, 2), kv_padding_mask) # k: (batch_size, seqlen_k, nheads, d)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v.transpose(1, 2), kv_padding_mask) # v: (batch_size, seqlen_v, nheads, d)
                # handle the q (B, H_num, q_len, H_dim)
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)]).to(k.device, dtype=torch.int32)
                max_seqlen_q = q_len
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
            else: # if attn mask is None, then do full cross-attn
                q_unpad = rearrange(q, 'b h s d -> (b s) h d')
                k_unpad = rearrange(k, 'b h s d -> (b s) h d')
                v_unpad = rearrange(v, 'b h s d -> (b s) h d')
                cu_seqlens_q = torch.tensor([q_len * i for i in range(B+1)])
                cu_seqlens_k = torch.tensor([t_len * i for i in range(B+1)])
                max_seqlen_q = q_len
                max_seqlen_k = t_len
            
            x = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, 
                                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                       softmax_scale=None, causal=False,
                                       return_attn_probs=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=B)
        # ipdb.set_trace() # save the result
        return x    


class DecoderLayer(nn.Module):
    '''
        This is a version of transformer decoder which combine the DETR and Perceiver-IO
    '''
    
    def __init__(self, 
                 d_query,
                 d_input,
                 d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.0,
                 decoder_func=DecoderVideoCrossAttention):
        super().__init__()
        assert d_query == d_model
        # Init a cross-attn layer
        self.attn = decoder_func(d_query, d_input, d_model, nhead, attn_drop=dropout)
        # Implementation of Feedforward model
        self.ffn = FeedForward(d_model, dim_feedforward=dim_feedforward)
    
    def forward(self, 
                query, 
                memory,
                memory_mask: Optional[Tensor] = None,
                key_temporal_pos: Optional[Tensor] = None,
                rope_axis='time'):
        '''
            query: The query tokens (B, num_q, dim_q)
            memory: The video tokens (B, T, S, dim_video) 
            memory_mask: shoule be the mask of the video tokens
            query_pos: positional embedding for the query
        '''
        
        #ipdb.set_trace()
        query = self.attn(q=query,
                            k=memory,
                            v=memory, 
                            attn_mask=memory_mask,
                            rope=key_temporal_pos,
                            rope_axis=rope_axis) + query
        #ipdb.set_trace()
        query = self.ffn(query) + query
        return query


#### The following is from the slow-fast module
# Referemce: https://github.com/facebookresearch/SlowFast/tree/main
class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        with_pooling=True,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.with_pooling = with_pooling
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        if norm_module == nn.BatchNorm3d:
            self.norm = norm_module(num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        else:
            self.norm = norm_module(dim_out, eps=self.eps)
        self.relu = nn.ReLU(self.inplace_relu)
        if self.with_pooling:
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
            )
        else:
            self.pool_layer = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.with_pooling:
            x = self.pool_layer(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        if norm_module == nn.BatchNorm3d:
            self.a_norm = norm_module(
                num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
        else:
            self.a_norm = norm_module(dim_inner, eps=self._eps)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        if norm_module == nn.BatchNorm3d:
            self.b_norm = norm_module(
                num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
            )
        else:
            self.b_norm = norm_module(dim_inner, eps=self._eps)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c.final_conv = True
        if norm_module == nn.BatchNorm3d:
            self.c_norm = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.c_norm.transform_final_bn = True
        else:
            self.c_norm = norm_module(
                dim_out, eps=self._eps
            )

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_norm(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_norm(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_norm(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            if norm_module == nn.BatchNorm3d:
                self.branch1_norm = norm_module(
                    num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
                )
            else:
                self.branch1_norm = norm_module(
                    dim_out, eps=self._eps
                )
                
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = drop_path(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_norm(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform": BottleneckTransform,
        # "basic_transform": BasicTransform,
        # "x3d_transform": X3DTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class ResStage(nn.Module):
    """
    This version is modified base on the official implementation
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_size,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.0,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_size (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_size
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_size to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage, self).__init__()
        assert num_block_temp_kernel <= num_blocks
        self.num_blocks = num_blocks
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_size = ([temp_kernel_size] * num_blocks)[: num_block_temp_kernel] + [1] * (num_blocks - num_block_temp_kernel)
        

        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
    ):
        # ipdb.set_trace() # check init
        for i in range(self.num_blocks):
            # Retrieve the transformation function.
            trans_func = get_trans_func(trans_func_name)
            # Construct the block.
            res_block = ResBlock(
                dim_in if i == 0 else dim_out,
                dim_out,
                self.temp_kernel_size[i],
                stride if i == 0 else 1,
                trans_func,
                dim_inner,
                num_groups,
                stride_1x1=stride_1x1,
                inplace_relu=inplace_relu,
                dilation=dilation,
                norm_module=norm_module,
                block_idx=i,
                drop_connect_rate=self._drop_connect_rate,
            )
            self.add_module("res{}".format(i), res_block)

    def forward(self, inputs):

        x = inputs
        for i in range(self.num_blocks):
            m = getattr(self, "res{}".format(i))
            x = m(x)

        return x
