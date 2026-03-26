from typing import Tuple
import torch

import triton
import triton.language as tl

from triton_mask_stack import triton_create_mask_stack
from fast_kmeans import flash_kmeans

import math
import time

@triton.jit
def _create_sparse_attn_mask(
    attention_mask_ptr,
    topk_indices_ptr,
    mask_stack_ptr,
    attention_mask_stride0,
    attention_mask_stride1,
    attention_mask_stride2,
    topk_indices_stride0,
    topk_indices_stride1,
    topk_indices_stride2,
    mask_stack_stride0,
    mask_stack_stride1,
    mask_stack_stride2,
    KERNEL_NUM: tl.constexpr,
    TOPK_NUM: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    head_id = tl.program_id(1)
    q_id = tl.program_id(2)
    attention_mask_offset = bid * attention_mask_stride0 + head_id * attention_mask_stride1
    topk_indices_offset = bid * topk_indices_stride0 + head_id * topk_indices_stride1 + (q_id * BLOCK_M + tl.arange(0, BLOCK_M)) * topk_indices_stride2 # [BLOCK_M]
    topk_indices_offset_max = bid * topk_indices_stride0 + head_id * topk_indices_stride1 + Q_LEN * topk_indices_stride2 # [1]
    for i in range(TOPK_NUM):
        topk_indices_val = tl.load(topk_indices_ptr + topk_indices_offset + i, mask=(topk_indices_offset + i < topk_indices_offset_max), other=0) # [BLOCK_M]
        for j in range((KV_LEN + BLOCK_N - 1) // BLOCK_N):
            mask_stack_offset = (topk_indices_val * mask_stack_stride0)[:, None] + bid * mask_stack_stride1 + head_id * mask_stack_stride2 + j * BLOCK_N + tl.arange(0, BLOCK_N) # [BLOCK_M, BLOCK_N]
            mask_stack_offset_max = (topk_indices_val * mask_stack_stride0)[:, None] + bid * mask_stack_stride1 + head_id * mask_stack_stride2 + KV_LEN # [BLOCK_M, 1]
            loaded_mask = tl.load(mask_stack_ptr + mask_stack_offset, mask=(mask_stack_offset<mask_stack_offset_max)) # [BLOCK_M, BLOCK_N]
            
            cur_attention_mask_offset = attention_mask_offset + (q_id * BLOCK_M + tl.arange(0, BLOCK_M)) * attention_mask_stride2 # [BLOCK_M]
            cur_attention_mask_offset = cur_attention_mask_offset[:, None] + j * BLOCK_N + tl.arange(0, BLOCK_N) # [BLOCK_M, BLOCK_N]
            
            cur_attention_mask_offset_q_max = attention_mask_offset + Q_LEN * attention_mask_stride2 # [1]
            cur_attention_mask_offset_q_max = cur_attention_mask_offset_q_max[:, None] + j * BLOCK_N + tl.arange(0, BLOCK_N) # [1, BLOCK_N]
            
            cur_attention_mask_offset_kv_max = attention_mask_offset + (q_id * BLOCK_M + tl.arange(0, BLOCK_M)) * attention_mask_stride2 # [BLOCK_M]
            cur_attention_mask_offset_kv_max = cur_attention_mask_offset_kv_max[:, None] + KV_LEN # [BLOCK_M, 1]
            
            mask_1 = cur_attention_mask_offset < cur_attention_mask_offset_q_max # [BLOCK_M, BLOCK_N]
            mask_2 = cur_attention_mask_offset < cur_attention_mask_offset_kv_max # [BLOCK_M, BLOCK_N]
            
            tl.atomic_add(attention_mask_ptr + cur_attention_mask_offset, loaded_mask, mask=(mask_1 & mask_2)) # [BLOCK_M, BLOCK_N]

def triton_create_sparse_attn_mask(
    attention_mask: torch.Tensor, # [BSZ, NUM_HEAD, Q_LEN, KV_LEN]
    topk_indices: torch.Tensor, # [BSZ, NUM_HEAD, Q_LEN, topk_num]
    mask_stack: torch.Tensor, # [kernel_num, BSZ, NUM_HEAD, KV_LEN]
):
    BSZ, NUM_HEAD, Q_LEN, KV_LEN = attention_mask.shape
    KERNEL_NUM = mask_stack.shape[0]
    TOPK_NUM = topk_indices.shape[-1]

    BLOCK_M = 128
    BLOCK_N = 128

    grid = lambda meta: (BSZ, NUM_HEAD, (Q_LEN+BLOCK_M-1)//BLOCK_M)
    _create_sparse_attn_mask[grid](
        attention_mask,
        topk_indices,
        mask_stack,
        attention_mask.stride(0),
        attention_mask.stride(1),
        attention_mask.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        mask_stack.stride(0),
        mask_stack.stride(1),
        mask_stack.stride(2),
        KERNEL_NUM,
        TOPK_NUM,
        Q_LEN,
        KV_LEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

if __name__ == "__main__":
    Q_LEN = 5626
    KV_LEN = 5626
    BSZ, NUM_HEAD, HEAD_DIM = 2, 30, 64
    topk_num = 64

    torch.cuda.set_device("cuda:2")

    query_states = torch.randn(BSZ, NUM_HEAD, Q_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
    key_states = torch.randn(BSZ, NUM_HEAD, KV_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
    value_states = torch.randn(BSZ, NUM_HEAD, KV_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")

    count_states = torch.ones(BSZ, NUM_HEAD, KV_LEN, 1, device=query_states.device, dtype=torch.float16)
    compression_ratio = 0.125
    print(f"{Q_LEN=}, {KV_LEN=}, {compression_ratio=}, {Q_LEN * compression_ratio=}, {KV_LEN * compression_ratio=}")
    kernel_num = int(KV_LEN * compression_ratio)
    random_generated = torch.randperm(KV_LEN, device=query_states.device)
    random_indices = random_generated[: kernel_num]
    random_indices = random_indices.unsqueeze(0).unsqueeze(0).expand(BSZ, NUM_HEAD, -1)
    key_kernel = key_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, HEAD_DIM))
    value_kernel = value_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, HEAD_DIM))
    count_kernel = count_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, 1))

    key_kernel, value_kernel, count_kernel, cluster_indices = flash_kmeans(
        key_kernel,
        key_states,
        value_kernel,
        value_states,
        count_kernel,
        count_states,
    )

    new_stack_mask = torch.zeros(kernel_num, BSZ, NUM_HEAD, KV_LEN, device=query_states.device, dtype=torch.bool)
    new_stack_mask = triton_create_mask_stack(new_stack_mask, cluster_indices)

    attn_weights = torch.matmul(key_kernel, query_states.transpose(2, 3)) / math.sqrt(HEAD_DIM) # [kernel_num, query_num]
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
    ).to(query_states.dtype) # [query_dim, kernel_num]
    topk_indices = torch.topk(attn_weights, k=topk_num, dim=-1).indices # [BSZ, NUM_HEAD, Q_LEN, topk_num]

    torch.cuda.synchronize()
    st_time = time.time()

    triton_attention_masks = torch.zeros(BSZ, NUM_HEAD, Q_LEN, KV_LEN, device=query_states.device, dtype=torch.int32)
    triton_create_sparse_attn_mask(triton_attention_masks, topk_indices, new_stack_mask)
    triton_attention_masks = triton_attention_masks.to(torch.bool)

    torch.cuda.synchronize()
    ed_time = time.time()
    print(f"triton create time: {(ed_time - st_time) * 1000:.2f} ms")