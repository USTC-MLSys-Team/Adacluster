from typing import Tuple
import torch

import triton
import triton.language as tl

@triton.jit
def _triton_fa(
    query,
    key,
    value,
    output,
    sm_scale,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    HEAD_DIM: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    qkvo_offset_start = b_id * query_stride_0 + h_id * query_stride_1

    qo_offset = qkvo_offset_start + (q_block_id * BLOCK_N + tl.arange(0, BLOCK_N)) * query_stride_2 # [BLOCK_N]
    qo_offset = qo_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
    qo_mask = q_block_id * BLOCK_N + tl.arange(0, BLOCK_N) < Q_LEN # [BLOCK_N]
    qo_mask = qo_mask[:, None] # [BLOCK_N, 1]
    load_query = tl.load(query + qo_offset, mask=qo_mask) # [BLOCK_N, HEAD_DIM]

    m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    kv_iter_num = (KV_LEN + BLOCK_N - 1) // BLOCK_N

    kv_max_offset = qkvo_offset_start + KV_LEN * key_stride_2
    kv_max_offset = kv_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]
    
    for i in range(kv_iter_num):
            
        kv_offset = qkvo_offset_start + (i * BLOCK_N + tl.arange(0, BLOCK_N)) * key_stride_2
        kv_offset = kv_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
        kv_load_mask = kv_offset < kv_max_offset[None, :]
        
        load_key = tl.load(key + kv_offset, mask=kv_load_mask) # [BLOCK_N, HEAD_DIM]
        load_value = tl.load(value + kv_offset, mask=kv_load_mask) # [BLOCK_N, HEAD_DIM]
        
        qk = tl.dot(load_query, load_key.T) # [BLOCK_N, BLOCK_N]
        qk_mask = i * BLOCK_N + tl.arange(0, BLOCK_N)
        qk_mask = qk_mask < KV_LEN
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
    
    tl.store(output + qo_offset, acc.to(tl.float16), mask=qo_mask)

def triton_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
):
    BATCH_SIZE, NUM_HEAD, Q_LEN, HEAD_DIM = query.shape
    _, _, KV_LEN, _ = key.shape
    assert Q_LEN == KV_LEN, "Now we only support Q_LEN == KV_LEN in diffusion model"

    BLOCK_N = 128
    
    grid = lambda args: (
        BATCH_SIZE,
        NUM_HEAD,
        Q_LEN // BLOCK_N,
    )

    # print(BATCH_SIZE, NUM_HEAD, (Q_KERNEL_NUM + KERNEL_BLOCK_N - 1) // KERNEL_BLOCK_N, BATCH_SIZE * NUM_HEAD * (Q_KERNEL_NUM + KERNEL_BLOCK_N - 1) // KERNEL_BLOCK_N)

    output = torch.zeros_like(query)

    with torch.cuda.device(query.device):
        _triton_fa[grid](
            query,
            key,
            value,
            output,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            HEAD_DIM,
            Q_LEN,
            KV_LEN,
            BLOCK_N=BLOCK_N,
        )

    return output