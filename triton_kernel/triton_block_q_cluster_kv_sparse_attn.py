from typing import Tuple
import torch

import triton
import triton.language as tl

@triton.jit
def _block_q_cluster_kv_sparse_attn(
    query,
    key,
    value,
    output,
    compressed_attn_mask,
    kv_counts,
    sm_scale,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    compressed_attn_mask_stride_0,
    compressed_attn_mask_stride_1,
    compressed_attn_mask_stride_2,
    kv_counts_stride_0,
    kv_counts_stride_1,
    kv_counts_stride_2,
    NUM_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_LEN: tl.constexpr,
    Q_KERNEL_NUM: tl.constexpr,
    KV_KERNEL_NUM: tl.constexpr,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_Q: tl.constexpr = 32,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q_kernel_start_id = q_block_id * BLOCK_Q

    query_offset = b_id * query_stride_0 + h_id * query_stride_1 + (q_kernel_start_id + tl.arange(0, BLOCK_Q)) * query_stride_2 # [BLOCK_Q]
    query_offset = query_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_Q, HEAD_DIM]
    query_load_mask = q_kernel_start_id + tl.arange(0, BLOCK_Q) # [BLOCK_Q]
    query_load_mask = query_load_mask < Q_LEN # [BLOCK_Q]
    load_query = tl.load(query + query_offset, mask=query_load_mask[:, None]) # [BLOCK_Q, HEAD_DIM]

    m_i = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    for j in range(KV_KERNEL_NUM):
        compressed_attn_mask_offset = b_id * compressed_attn_mask_stride_0 + h_id * compressed_attn_mask_stride_1 + q_block_id * compressed_attn_mask_stride_2 + j
        skip_flag = tl.load(compressed_attn_mask + compressed_attn_mask_offset)
        if skip_flag != False:
            KV_END_OFFSET = tl.load(
                kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + j * kv_counts_stride_2
            )
            if j == 0:
                KV_START_OFFSET = 0
            else:
                KV_START_OFFSET = tl.load(
                    kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + (j - 1) * kv_counts_stride_2
                )
            kv_len = KV_END_OFFSET - KV_START_OFFSET
            kv_iter_num = (kv_len + BLOCK_N - 1) // BLOCK_N
            for k in range(kv_iter_num):
                
                key_offset = b_id * key_stride_0 + h_id * key_stride_1 + (KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)) * key_stride_2
                key_offset = key_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
                key_max_offset = b_id * key_stride_0 + h_id * key_stride_1 + KV_END_OFFSET * key_stride_2
                key_max_offset = key_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]
                key_load_mask = key_offset < key_max_offset[None, :]
                
                load_key = tl.load(key + key_offset, mask=key_load_mask) # [BLOCK_N, HEAD_DIM]
                
                value_offset = b_id * value_stride_0 + h_id * value_stride_1 + (KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)) * value_stride_2
                value_offset = value_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
                value_max_offset = b_id * value_stride_0 + h_id * value_stride_1 + KV_END_OFFSET * value_stride_2
                value_max_offset = value_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]
                value_load_mask = value_offset < value_max_offset[None, :]
                
                load_value = tl.load(value + value_offset, mask=value_load_mask) # [BLOCK_N, HEAD_DIM]
                
                qk = tl.dot(load_query, load_key.T) # [BLOCK_N, BLOCK_N]
                qk_mask = KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)
                qk_mask = qk_mask < KV_END_OFFSET
                qk = qk * qk_scale + tl.where(qk_mask, 0, -1e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1)) # [BLOCK_N]
                qk -= m_ij[:, None]
                p = tl.math.exp2(qk) # [BLOCK_N]
                l_ij = tl.sum(p, 1) # [BLOCK_N]
                alpha = tl.math.exp2(m_i - m_ij) # [BLOCK_N]
                l_i = l_i * alpha + l_ij # [BLOCK_N]
                acc = acc * alpha[:, None]
                p = p.to(tl.float16)
                acc = tl.dot(p, load_value, acc)
                m_i = m_ij

    acc = acc / l_i[:, None]
    output_store_offset = b_id * output_stride_0 + h_id * output_stride_1 + (q_kernel_start_id + tl.arange(0, BLOCK_Q)) * output_stride_2
    output_store_offset = output_store_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_Q, HEAD_DIM]
    
    output_store_mask = q_kernel_start_id + tl.arange(0, BLOCK_Q) # [BLOCK_Q]
    output_store_mask = output_store_mask < Q_LEN # [BLOCK_Q]
    
    tl.store(output + output_store_offset, acc.to(tl.float16), mask=output_store_mask[:, None])

def triton_block_q_cluster_kv_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compressed_attn_mask: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
    BLOCK_N: int,
    BLOCK_Q: int,
):
    BATCH_SIZE, NUM_HEAD, Q_LEN, HEAD_DIM = query.shape
    _, _, KV_LEN, _ = key.shape
    assert Q_LEN == KV_LEN, "Now we only support Q_LEN == KV_LEN in diffusion model"

    _, _, Q_KERNEL_NUM, KV_KERNEL_NUM = compressed_attn_mask.shape
    
    grid = lambda args: (
        BATCH_SIZE,
        NUM_HEAD,
        Q_KERNEL_NUM
        # (Q_KERNEL_NUM + BLOCK_Q - 1) // BLOCK_Q,
    )

    # print(BATCH_SIZE, NUM_HEAD, (Q_KERNEL_NUM + KERNEL_BLOCK_N - 1) // KERNEL_BLOCK_N, BATCH_SIZE * NUM_HEAD * (Q_KERNEL_NUM + KERNEL_BLOCK_N - 1) // KERNEL_BLOCK_N)

    output = torch.zeros_like(query)

    with torch.cuda.device(query.device):
        _block_q_cluster_kv_sparse_attn[grid](
            query,
            key,
            value,
            output,
            compressed_attn_mask,
            kv_counts,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            compressed_attn_mask.stride(0),
            compressed_attn_mask.stride(1),
            compressed_attn_mask.stride(2),
            kv_counts.stride(0),
            kv_counts.stride(1),
            kv_counts.stride(2),
            NUM_HEAD,
            HEAD_DIM,
            Q_LEN,
            Q_KERNEL_NUM,
            KV_KERNEL_NUM,
            BLOCK_N=BLOCK_N,
            BLOCK_Q=BLOCK_Q,
        )

    return output