import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention

from triton_kernel.fast_kmeans import flash_kmeans
from triton_kernel.fast_kmeans_single import flash_kmeans_single
from triton_kernel.triton_cluster_sparse_attn import triton_cluster_sparse_attn

import time, sys
import math
import os

from torch.cuda.nvtx import range_push, range_pop






class CogVideoXAttnProcessor2_0_compare:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)


        #原始的attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )





        seq_length = query.shape[2]
        count_states = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        count_states_for_q = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        
        compression_ratio = int(os.environ.get("C_RATIO", 128))
        topk_num = int(os.environ.get("TOPK_NUM", 32))
        top_P = float(os.environ.get("TOP_P", 1))
        q_kernel_num = int(os.environ.get("Q_KERNEL_NUM", 100))
        kv_kernel_num = int(os.environ.get("KV_KERNEL_NUM", 500))

        kernel_num = int(seq_length // compression_ratio)
        if q_kernel_num == 0:
            q_kernel_num = kernel_num
        if kv_kernel_num == 0:
            kv_kernel_num = kernel_num
        
        random_generated = torch.randperm(seq_length, device=query.device)
        random_generated_for_query = torch.randperm(seq_length, device=query.device)
        
        random_indices = random_generated[: kv_kernel_num]
        random_indices = random_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)

        key_kernel = key.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        value_kernel = value.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        count_kernel_for_kv = count_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, 1))

        random_indices_for_query = random_generated_for_query[: q_kernel_num]
        random_indices_for_query = random_indices_for_query.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
        
        query_kernel = query.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        count_kernel_for_q = count_states_for_q.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, 1))

  

        # key_kernel, value_kernel, count_kernel_for_kv, cluster_indices_for_kv = kmeans_compress_kernel(
        key_kernel, value_kernel, count_kernel_for_kv, cluster_indices_for_kv = flash_kmeans(
            key_kernel,
            key,
            value_kernel,
            value,
            count_kernel_for_kv,
            count_states,
        )



        # query_kernel, _, count_kernel_for_q, cluster_indices_for_query = kmeans_compress_kernel(
        query_kernel, count_kernel_for_q, cluster_indices_for_query = flash_kmeans_single(
            query_kernel,
            query,
        )

 


        attn_weights = torch.matmul(key_kernel, query_kernel.transpose(2, 3)) / math.sqrt(head_dim) # [kernel_num, query_num]
        cluster_bias = torch.where(
            count_kernel_for_kv > 0,
            torch.log(count_kernel_for_kv),
            torch.finfo(attn_weights.dtype).min,
        )
        attn_weights = attn_weights + cluster_bias
        attn_weights = torch.softmax(
            attn_weights.transpose(2, 3),
            dim=-1,
            dtype=torch.float32
        ).to(query.dtype)

  
        sorted_attn_weights, sorted_attn_weights_indices = attn_weights.sort(dim=-1, descending=True) # [batch_size, num_heads, query_num, kernel_num]

        selected_attn_weights = sorted_attn_weights.cumsum(dim=-1) <= top_P # [batch_size, num_heads, query_num, kernel_num]

        sorted_attn_weights_indices = torch.where(selected_attn_weights, sorted_attn_weights_indices, -1) # [batch_size, num_heads, query_num, kernel_num]

        compressed_mask = torch.zeros(batch_size, attn.heads, q_kernel_num, kv_kernel_num + 1, dtype=torch.bool, device=query.device)

        valid_indices = sorted_attn_weights_indices != -1
        compressed_mask.scatter_(
            dim=-1,
            index=sorted_attn_weights_indices.where(valid_indices, kv_kernel_num),
            src=valid_indices,
        )

        compressed_mask = compressed_mask[:, :, :, :kv_kernel_num].contiguous()

        matmul_count_kernel = torch.matmul(count_kernel_for_q.to(torch.float32), count_kernel_for_kv.transpose(2, 3).to(torch.float32)).to(torch.int32)

        count_kernel_sum = compressed_mask * matmul_count_kernel



        count_kernel_for_kv = count_kernel_for_kv.squeeze(-1)
        count_kernel_for_q = count_kernel_for_q.squeeze(-1)

        count_kernel_for_kv = count_kernel_for_kv.to(torch.int32)
        count_kernel_for_q = count_kernel_for_q.to(torch.int32)

        count_kernel_for_kv = torch.cumsum(count_kernel_for_kv, dim=-1)
        count_kernel_for_q = torch.cumsum(count_kernel_for_q, dim=-1)

        count_kernel_for_kv = count_kernel_for_kv.to(torch.int32).contiguous()
        count_kernel_for_q = count_kernel_for_q.to(torch.int32).contiguous()

        sorted_q_indices = torch.argsort(cluster_indices_for_query, dim=-1)
        sorted_kv_indices = torch.argsort(cluster_indices_for_kv, dim=-1)

        recovery_q_indices = torch.argsort(sorted_q_indices, dim=-1)

        reordered_query = torch.gather(query, 2, sorted_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        reordered_key = torch.gather(key, 2, sorted_kv_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        reordered_value = torch.gather(value, 2, sorted_kv_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()

        compressed_mask = compressed_mask.contiguous()


        hidden_states_svg = triton_cluster_sparse_attn(
            query=reordered_query,
            key=reordered_key,
            value=reordered_value,
            compressed_attn_mask=compressed_mask,
            q_counts=count_kernel_for_q,
            kv_counts=count_kernel_for_kv,
            sm_scale=1.0 / math.sqrt(head_dim),
        )


        hidden_states_svg = torch.gather(hidden_states_svg, 2, recovery_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))



        topk_indices = attn_weights.topk(k=topk_num, dim=-1).indices

        compressed_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

        compressed_mask.scatter_(
            dim=-1,
            index=topk_indices,
            value=True
        )




        compressed_mask = compressed_mask.contiguous()

        hidden_states_kvclus = triton_cluster_sparse_attn(
            query=reordered_query,
            key=reordered_key,
            value=reordered_value,
            compressed_attn_mask=compressed_mask,
            q_counts=count_kernel_for_q,
            kv_counts=count_kernel_for_kv,
            sm_scale=1.0 / math.sqrt(head_dim),
        )

        hidden_states_kvclus = torch.gather(hidden_states_kvclus, 2, recovery_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        print(f"svg-full-mse : {F.mse_loss(hidden_states_svg, hidden_states)},kvclus-full-mse : {F.mse_loss(hidden_states_kvclus, hidden_states)}")






        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states