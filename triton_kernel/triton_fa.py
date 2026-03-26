from typing import Tuple
import torch

import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [64, 128]
    for s in ([2, 4])
    for w in [1]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["HEAD_DIM"])
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
    value_stride_0,
    value_stride_1,
    value_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    NUM_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q_offset = b_id * query_stride_0 + h_id * query_stride_1 + (q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)) * query_stride_2 # [BLOCK_M]
    q_offset = q_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_M, HEAD_DIM]
    q_max_offset = b_id * query_stride_0 + h_id * query_stride_1 + Q_LEN * query_stride_2
    q_max_offset = q_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]
    q_load_mask = q_offset < q_max_offset[None, :] # [BLOCK_M, HEAD_DIM]
    load_query = tl.load(query + q_offset, mask=q_load_mask) # [BLOCK_M, HEAD_DIM]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    kv_iter_num = (KV_LEN + BLOCK_N - 1) // BLOCK_N
    
    key_max_offset = b_id * key_stride_0 + h_id * key_stride_1 + KV_LEN * key_stride_2
    key_max_offset = key_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]

    value_max_offset = b_id * value_stride_0 + h_id * value_stride_1 + KV_LEN * value_stride_2
    value_max_offset = value_max_offset + tl.arange(0, HEAD_DIM) # [HEAD_DIM]

    for i in range(kv_iter_num):
            
        key_offset = b_id * key_stride_0 + h_id * key_stride_1 + (i * BLOCK_N + tl.arange(0, BLOCK_N)) * key_stride_2
        key_offset = key_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
        
        key_load_mask = key_offset < key_max_offset[None, :]
        
        load_key = tl.load(key + key_offset, mask=key_load_mask) # [BLOCK_N, HEAD_DIM]
        
        value_offset = b_id * value_stride_0 + h_id * value_stride_1 + (i * BLOCK_N + tl.arange(0, BLOCK_N)) * value_stride_2
        value_offset = value_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_N, HEAD_DIM]
        
        value_load_mask = value_offset < value_max_offset[None, :]
        
        load_value = tl.load(value + value_offset, mask=value_load_mask) # [BLOCK_N, HEAD_DIM]
        
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
    output_store_offset = b_id * output_stride_0 + h_id * output_stride_1 + (q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)) * output_stride_2
    output_store_offset = output_store_offset[:, None] + tl.arange(0, HEAD_DIM) # [BLOCK_M, HEAD_DIM]
    
    output_store_mask = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M) # [BLOCK_M]
    output_store_mask = output_store_mask < Q_LEN # [BLOCK_M]
    
    tl.store(output + output_store_offset, acc.to(tl.float16), mask=output_store_mask[:, None])

def triton_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
):
    BATCH_SIZE, NUM_HEAD, Q_LEN, HEAD_DIM = query.shape
    _, _, KV_LEN, _ = key.shape
    assert Q_LEN == KV_LEN, "Now we only support Q_LEN == KV_LEN in diffusion model"
    
    BLOCK_M = 128
    BLOCK_N = 128

    grid = lambda args: (
        BATCH_SIZE,
        NUM_HEAD,
        triton.cdiv(Q_LEN, BLOCK_M),
        # triton.cdiv(Q_LEN, args["BLOCK_M"]),
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
            value.stride(0),
            value.stride(1),
            value.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_HEAD,
            HEAD_DIM,
            Q_LEN,
            KV_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return output