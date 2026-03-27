import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention
import math

import os

def compute_cluster_indices(kernel_k, k):
    kernel_norm = kernel_k.norm(dim=-1)
    kernel_norm = kernel_norm * kernel_norm
    dis = kernel_norm.unsqueeze(2) - 2 * torch.matmul(k, kernel_k.transpose(2, 3))
    min_indices = torch.argmin(dis, dim=-1)

    return dis, min_indices


def compute_new_kernel(min_indices, kernel_k, k, kernel_v, v, kernel_count, count):
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
        kernel_count[:, i, :, :] = onehot_dis.sum(dim=-1, keepdim=True) + 1

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
        self.step_counter = 0  # 步骤计数器
        self.use_full_attn_layers = [17, 34,38,39]  # 使用完整注意力的层
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.step_counter += 1
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

        seq_length = query.shape[2]
        count_states = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=query.dtype)
        compression_ratio = int(os.environ.get("C_RATIO", 4))
        kernel_num = int(seq_length // compression_ratio)
        random_generated = torch.randperm(seq_length, device=query.device)
        random_indices = random_generated[: kernel_num]
        random_indices = random_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
        key_kernel = key.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        value_kernel = value.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        count_kernel = count_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, 1))

        key_kernel, value_kernel, count_kernel, cluster_indices = kmeans_compress_kernel(
            key_kernel,
            key,
            value_kernel,
            value,
            count_kernel,
            count_states,
        )

        mask_tensors = []
        for i in range(kernel_num):
            mask_tensors.append(cluster_indices == i) # [batch, num_head, kv_len]

        mask_stack = torch.stack(mask_tensors, dim=0)  # shape: [kernel_num, BSZ, NUM_HEAD, KV_LEN]

        chunk_query_size = 512
        iter_num = (seq_length + chunk_query_size - 1) // chunk_query_size

        final_hidden_states = torch.ones(batch_size, attn.heads, 0, head_dim, device=query.device, dtype=query.dtype)

        for i in range(iter_num):
            if i != iter_num - 1:
                cur_query = query[:, :, i * chunk_query_size: (i + 1) * chunk_query_size, :]
                cur_seq_len = chunk_query_size
            else:
                cur_query = query[:, :, i * chunk_query_size:, :]
                cur_seq_len = seq_length - (i * chunk_query_size)

            attn_weights = torch.matmul(key_kernel, cur_query.transpose(2, 3)) / math.sqrt(head_dim) # [kernel_num, query_num]
            cluster_bias = torch.where(
                count_kernel > 0,
                torch.log(count_kernel),
                torch.finfo(attn_weights.dtype).min,
            )
            attn_weights = attn_weights + cluster_bias
            attn_weights = torch.softmax(
                attn_weights.transpose(2, 3),
                dim=-1,
                dtype=torch.float32
            ).to(query.dtype)

            topk_num = int(os.environ.get("TOPK_NUM", 128))
            topk_indices = torch.topk(attn_weights, k=topk_num, dim=-1).indices # [BSZ, NUM_HEAD, Q_LEN, topk_num]

            mask_stack_exp = mask_stack.permute(1, 2, 0, 3)  # [BSZ, NUM_HEAD, kernel_num, KV_LEN]

            # Expand mask_stack to [BSZ, NUM_HEAD, Q_LEN, kernel_num, KV_LEN]
            mask_stack_exp = mask_stack_exp.unsqueeze(2).expand(-1, -1, cur_seq_len, -1, -1)  # [2, 3, 4, 16, 32]

            # topk_indices: [BSZ, NUM_HEAD, Q_LEN, topk_num]
            index_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, seq_length)  # [2, 3, 4, 5, 32]

            # Gather top-k masks from kernel_num dim (dim=3)
            topk_masks = torch.gather(mask_stack_exp, dim=3, index=index_expanded)  # [2, 3, 4, 5, 32]

            # OR across top-k dimension to get final attention_masks
            attention_masks = topk_masks.any(dim=3)  # [2, 3, 4, 32]
            # print(cur_query.shape, key.shape, value.shape, attention_masks.shape)
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                cur_query,
                key,
                value,
                attn_mask=attention_masks,
                dropout_p=0.0,
                is_causal=False,
            )

            # true_count = attention_masks.sum().item()
            # total_elements = attention_masks.numel()
            # proportion = true_count / total_elements
            # print(f"sparsity ratio: {proportion}")

            final_hidden_states = torch.cat([final_hidden_states, hidden_states], dim=-2)

        hidden_states = final_hidden_states

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states