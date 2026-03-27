import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention
import math

import os

from triton_kernel.fast_kmeans import flash_kmeans
from triton_kernel.fast_kmeans_single import flash_kmeans_single
from triton_kernel.triton_cluster_sparse_attn import triton_cluster_sparse_attn

import time, sys

from torch.cuda.nvtx import range_push, range_pop

def compute_cluster_indices(kernel_k, k):
    kernel_norm = kernel_k.norm(dim=-1)
    kernel_norm = kernel_norm * kernel_norm
    
    # dis = kernel_norm.unsqueeze(2) - 2 * torch.matmul(k, kernel_k.transpose(2, 3))
    # min_indices = torch.argmin(dis, dim=-1)

    BATCH_SIZE, N_HEADS = kernel_k.shape[0], kernel_k.shape[1]
    KEY_NUM = k.shape[2]

    min_indices = torch.zeros((BATCH_SIZE, N_HEADS, KEY_NUM), dtype=torch.int64, device=kernel_k.device)

    for i in range(N_HEADS):
        dis = kernel_norm[:, i, :].unsqueeze(1) - 2 * torch.matmul(k[:, i, :], kernel_k[:, i, :].transpose(1, 2))
        min_indices[:, i, :] = torch.argmin(dis, dim=-1)

    return dis, min_indices


def compute_new_kernel(min_indices, kernel_k, k, kernel_v, v, kernel_count, count):
    # print(min_indices.shape, kernel_k.shape, k.shape)
    N_HEADS = kernel_k.shape[1]
    N_KERNEL = kernel_k.shape[2]

    for i in range(N_HEADS):

        onehot_dis = F.one_hot(min_indices[:, i, :], N_KERNEL)
        onehot_dis = onehot_dis.transpose(-1, -2).to(k)

        kernel_k[:, i, :, :] = (
            kernel_k[:, i, :, :] + torch.matmul(onehot_dis, k[:, i, :, :])
        ) / (onehot_dis.sum(dim=-1, keepdim=True) + 1)
        kernel_v[:, i, :, :] = (
            kernel_v[:, i, :, :] + torch.matmul(onehot_dis, v[:, i, :, :])
        ) / (onehot_dis.sum(dim=-1, keepdim=True) + 1)
        kernel_count[:, i, :, :] = onehot_dis.sum(dim=-1, keepdim=True)

    return kernel_k, kernel_v, kernel_count


def kmeans_compress_kernel(
    key_kernel,
    middle_key,
    value_kernel,
    middle_value,
    count_kernel,
    middle_count,
):
    # range_push("kmeans_compress_kernel")

    # check: how to handle the case of multiiple kmeans function call
    # check: how to handle middle count
    for _ in range(5):

        distance, indices = compute_cluster_indices(key_kernel, middle_key)
        key_kernel, value_kernel, count_kernel = compute_new_kernel(
            indices,
            key_kernel,
            middle_key,
            value_kernel,
            middle_value,
            count_kernel,
            middle_count,
        )

    # range_pop()
    return key_kernel, value_kernel, count_kernel, indices

class CogVideoXAttnProcessor2_0_kvclus:
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
        
        range_push("QKV proj")

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

        range_pop()
        range_push("QK norm")

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        range_pop()
        range_push("RoPE")

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        range_pop()
        range_push("Kernel Initialization")

        seq_length = query.shape[2]
        count_states = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        count_states_for_q = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        
        compression_ratio = int(os.environ.get("C_RATIO", 4))
        topk_num = int(os.environ.get("TOPK_NUM", 64))
        q_kernel_num = int(os.environ.get("Q_KERNEL_NUM", 0))
        kv_kernel_num = int(os.environ.get("KV_KERNEL_NUM", 0))

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

        range_pop()
        range_push("KV Clustering")

        # key_kernel = key_kernel.to(torch.float32)
        # value_kernel = value_kernel.to(torch.float32)
        key_kernel, value_kernel, count_kernel_for_kv, cluster_indices_for_kv = flash_kmeans(
            key_kernel,
            key,
            value_kernel,
            value,
            count_kernel_for_kv,
            count_states,
        )

        # key_kernel = key_kernel.to(query.dtype)
        # value_kernel = value_kernel.to(query.dtype)

        range_pop()
        range_push("Q Clustering")

        # query_kernel = query_kernel.to(torch.float32)
        query_kernel, count_kernel_for_q, cluster_indices_for_query = flash_kmeans_single(
            query_kernel,
            query,
        )
        # query_kernel = query_kernel.to(key_kernel.dtype)

        range_pop()
        range_push("TopK Selection")
        range_push("Attn Weights Calculation")

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

        range_pop()
        range_push("TopK Selection")

        topk_indices = attn_weights.topk(k=topk_num, dim=-1).indices

        range_pop()
        range_pop()

        # for cluster-wise sparsity calculation
        # BLOCK_N = 128
        # tmp_topk_indices = topk_indices.reshape(batch_size, attn.heads, kernel_num * topk_num)
        # selected_key_kernel_count = torch.gather(count_kernel_for_kv, 2, tmp_topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1))
        # selected_key_kernel_count = torch.reshape(selected_key_kernel_count, (batch_size, attn.heads, kernel_num, topk_num))
        
        # aligned_selected_key_kernel_count = (selected_key_kernel_count + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        # aligned_selected_key_kernel_count = torch.sum(aligned_selected_key_kernel_count, dim=-1)
        
        # selected_key_kernel_count = torch.sum(selected_key_kernel_count, dim=-1)
        
        # count_kernel_for_q_tmp = count_kernel_for_q.squeeze(-1)
        # # print(count_kernel_for_q_tmp[1, 29])
        # cluster_count_num = count_kernel_for_q_tmp * selected_key_kernel_count
        
        # aligned_count_kernel_for_q_tmp = (count_kernel_for_q_tmp + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        # aligned_cluster_count_num = aligned_count_kernel_for_q_tmp * aligned_selected_key_kernel_count

        # print("Cluster-Wise QKV theretical computation ratio: ", torch.sum(cluster_count_num) / batch_size / attn.heads / seq_length / seq_length)
        # print("Cluster-Wise QKV real computation ratio: ", torch.sum(aligned_cluster_count_num) / batch_size / attn.heads / seq_length / seq_length)
        # sys.stdout.flush()

        range_push("Compressed Mask Creation")

        compressed_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

        compressed_mask.scatter_(
            dim=-1,
            index=topk_indices,
            value=True
        )

        # matmul_count_kernel = torch.matmul(count_kernel_for_q.to(torch.float32), count_kernel_for_kv.transpose(2, 3).to(torch.float32)).to(torch.int32)

        # count_kernel_sum = compressed_mask * matmul_count_kernel

        # print("Sparsity ratio", torch.sum(count_kernel_sum) / (batch_size * attn.heads * seq_length * seq_length))
        # sys.stdout.flush()

        range_pop()

        range_push("Gathering Reordered QKV")

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

        range_pop()
        range_push("Cluster Sparse Attention")

        hidden_states = triton_cluster_sparse_attn(
            query=reordered_query,
            key=reordered_key,
            value=reordered_value,
            compressed_attn_mask=compressed_mask,
            q_counts=count_kernel_for_q,
            kv_counts=count_kernel_for_kv,
            sm_scale=1.0 / math.sqrt(head_dim),
        )

        range_pop()
        range_push("Gathering Reordered Hidden States")

        hidden_states = torch.gather(hidden_states, 2, recovery_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        range_pop()
        range_push("Post Attn")

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        range_pop()
        return hidden_states, encoder_hidden_states