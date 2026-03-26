from typing import Tuple
import torch
import triton
import triton.language as tl

@triton.jit
def _cluster_sparse_attn_1st_order(
    query,
    key,
    value,
    output,
    compressed_attn_mask,
    q_counts,
    kv_counts,
    key_center,  # 添加缺失的参数
    value_center,  # 添加缺失的参数
    k_offset,  # 添加缺失的参数
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
    q_counts_stride_0,
    q_counts_stride_1,
    q_counts_stride_2,
    kv_counts_stride_0,
    kv_counts_stride_1,
    kv_counts_stride_2,
    key_center_stride_0,  # 添加缺失的步长参数
    key_center_stride_1,  # 添加缺失的步长参数
    key_center_stride_2,  # 添加缺失的步长参数
    value_center_stride_0,  # 添加缺失的步长参数
    value_center_stride_1,  # 添加缺失的步长参数
    value_center_stride_2,  # 添加缺失的步长参数
    k_offset_stride_0,  # 添加缺失的步长参数
    k_offset_stride_1,  # 添加缺失的步长参数
    k_offset_stride_2,  # 添加缺失的步长参数
    NUM_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_LEN: tl.constexpr,
    Q_KERNEL_NUM: tl.constexpr,
    KV_KERNEL_NUM: tl.constexpr,
    BLOCK_N: tl.constexpr = 32,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    q_id = tl.program_id(2)

    qk_scale = sm_scale * 1.44269504

    # Q块起止索引
    Q_END_OFFSET = tl.load(q_counts + b_id * q_counts_stride_0 + h_id * q_counts_stride_1 + q_id * q_counts_stride_2)
    Q_START_OFFSET = 0 if q_id == 0 else tl.load(q_counts + b_id * q_counts_stride_0 + h_id * q_counts_stride_1 + (q_id - 1) * q_counts_stride_2)
    q_len = Q_END_OFFSET - Q_START_OFFSET
    if q_len == 0:
        return

    inner_block_n = min(BLOCK_N, max(32, q_len))
    q_iter_num = (q_len + inner_block_n - 1) // inner_block_n

    for i in range(q_iter_num):
        query_offset = b_id * query_stride_0 + h_id * query_stride_1 + (Q_START_OFFSET + i * BLOCK_N + tl.arange(0, BLOCK_N)) * query_stride_2
        query_offset = query_offset[:, None] + tl.arange(0, HEAD_DIM)
        query_load_mask = Q_START_OFFSET + i * BLOCK_N + tl.arange(0, BLOCK_N)
        query_load_mask = query_load_mask < Q_END_OFFSET
        load_query = tl.load(query + query_offset, mask=query_load_mask[:, None])

        m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_N], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        for j in range(KV_KERNEL_NUM):
            mask_offset = b_id * compressed_attn_mask_stride_0 + h_id * compressed_attn_mask_stride_1 + q_id * compressed_attn_mask_stride_2 + j
            skip_flag = tl.load(compressed_attn_mask + mask_offset)
            if skip_flag:
                # 跳过这个KV块的处理
                continue

            # KV块起止索引
            KV_END_OFFSET = tl.load(kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + j * kv_counts_stride_2)
            KV_START_OFFSET = 0 if j == 0 else tl.load(kv_counts + b_id * kv_counts_stride_0 + h_id * kv_counts_stride_1 + (j - 1) * kv_counts_stride_2)
            kv_len = KV_END_OFFSET - KV_START_OFFSET
            kv_iter_num = (kv_len + BLOCK_N - 1) // BLOCK_N

            for k in range(kv_iter_num):
                # 类中心
# ------------------------- Load Key -------------------------
                key_offset = b_id * key_stride_0 + h_id * key_stride_1 + (KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)) * key_stride_2
                key_offset = key_offset[:, None] + tl.arange(0, HEAD_DIM)  # [BLOCK_N, HEAD_DIM]
                key_max_offset = b_id * key_stride_0 + h_id * key_stride_1 + KV_END_OFFSET * key_stride_2
                key_max_offset = key_max_offset + tl.arange(0, HEAD_DIM)  # [HEAD_DIM]
                key_load_mask = key_offset < key_max_offset[None, :]      # mask 用于防止越界

                load_key = tl.load(key + key_offset, mask=key_load_mask)   # [BLOCK_N, HEAD_DIM] 原始 Key

                # ------------------------- Load Value -------------------------
                value_offset = b_id * value_stride_0 + h_id * value_stride_1 + (KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)) * value_stride_2
                value_offset = value_offset[:, None] + tl.arange(0, HEAD_DIM)  # [BLOCK_N, HEAD_DIM]
                value_max_offset = b_id * value_stride_0 + h_id * value_stride_1 + KV_END_OFFSET * value_stride_2
                value_max_offset = value_max_offset + tl.arange(0, HEAD_DIM)  # [HEAD_DIM]
                value_load_mask = value_offset < value_max_offset[None, :]    # mask 防止越界

                load_value = tl.load(value + value_offset, mask=value_load_mask)  # [BLOCK_N, HEAD_DIM] 原始 Value

                # ------------------------- Compute QK -------------------------
                qk = tl.dot(load_query, load_key.T)        # [BLOCK_N, BLOCK_N] query 与每个 key 的 dot-product
                qk_mask = KV_START_OFFSET + k * BLOCK_N + tl.arange(0, BLOCK_N)
                qk_mask = qk_mask < KV_END_OFFSET          # mask，用于处理块尾不满 BLOCK_N 的情况
                qk = qk * qk_scale + tl.where(qk_mask, 0, -1e6)  # 数值稳定处理
                m_ij = tl.maximum(m_i, tl.max(qk, 1))     # [BLOCK_N] 每个 query 的最大值，用于 softmax 稳定

                qk -= m_ij[:, None]                        # [BLOCK_N, BLOCK_N] 稳定化后的 qk
                p = tl.math.exp2(qk)                        # [BLOCK_N, BLOCK_N] softmax 前指数

                # ------------------------- One-step approximation (簇中心 + residual) -------------------------
                # 在一阶近似中：
                #   numerator = value_center + q · S_av   (簇中心加 residual 对 value 的一阶贡献)
                #   denominator = n_c + q · S_a           (簇元素数量 + residual 对注意力归一化的贡献)
                # 这里 load_value 可以被理解为 value_center + 一阶 residual 的近似
                l_ij = tl.sum(p, 1)                        # [BLOCK_N] 当前块每个 query 的分母累加
                alpha = tl.math.exp2(m_i - m_ij)           # [BLOCK_N] 数值稳定因子
                l_i = l_i * alpha + l_ij                    # [BLOCK_N] 总分母累加
                acc = acc * alpha[:, None]                  # [BLOCK_N, HEAD_DIM] 总分子累加
                p = p.to(load_value.dtype)                  # 转换为 value 的 dtype
                acc = tl.dot(p, load_value, acc)            # [BLOCK_N, HEAD_DIM] 注意力加权和
                m_i = m_ij                                  # 更新最大值


        # 写回输出
        acc /= l_i[:, None]
        output_offset = b_id * output_stride_0 + h_id * output_stride_1 + (Q_START_OFFSET + i * BLOCK_N + tl.arange(0, BLOCK_N)) * output_stride_2
        output_offset = output_offset[:, None] + tl.arange(0, HEAD_DIM)
        output_mask = Q_START_OFFSET + i * BLOCK_N + tl.arange(0, BLOCK_N)
        output_mask = output_mask < Q_END_OFFSET
        tl.store(output + output_offset, acc.to(tl.float16), mask=output_mask[:, None])


def triton_cluster_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_center: torch.Tensor,  # 添加缺失的参数
    value_center: torch.Tensor,  # 添加缺失的参数
    k_offset: torch.Tensor,  # 添加缺失的参数
    compressed_attn_mask: torch.Tensor,
    q_counts: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
):
    # 添加计时和统计
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    BATCH_SIZE, NUM_HEAD, Q_LEN, HEAD_DIM = query.shape
    _, _, KV_LEN, _ = key.shape
    # assert Q_LEN == KV_LEN, "Now we only support Q_LEN == KV_LEN in diffusion model"

    _, _, Q_KERNEL_NUM, KV_KERNEL_NUM = compressed_attn_mask.shape

    grid = lambda args: (
        BATCH_SIZE,
        NUM_HEAD,
        Q_KERNEL_NUM,
    )

    output = torch.zeros_like(query)

    with torch.cuda.device(query.device):
        _cluster_sparse_attn_1st_order[grid](
            query,
            key,
            value,
            output,
            compressed_attn_mask,
            q_counts,
            kv_counts,
            key_center,  # 传递缺失的参数
            value_center,  # 传递缺失的参数
            k_offset,  # 传递缺失的参数
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
            q_counts.stride(0),
            q_counts.stride(1),
            q_counts.stride(2),
            kv_counts.stride(0),
            kv_counts.stride(1),
            kv_counts.stride(2),
            key_center.stride(0),  # 传递缺失的步长参数
            key_center.stride(1),  # 传递缺失的步长参数
            key_center.stride(2),  # 传递缺失的步长参数
            value_center.stride(0),  # 传递缺失的步长参数
            value_center.stride(1),  # 传递缺失的步长参数
            value_center.stride(2),  # 传递缺失的步长参数
            k_offset.stride(0),  # 传递缺失的步长参数
            k_offset.stride(1),  # 传递缺失的步长参数
            k_offset.stride(2),  # 传递缺失的步长参数
            NUM_HEAD,
            HEAD_DIM,
            Q_LEN,
            Q_KERNEL_NUM,
            KV_KERNEL_NUM,
        )
    end_time.record()
    torch.cuda.synchronize()
    
    print(f"Execution time: {start_time.elapsed_time(end_time)}ms")
    print(f"Q counts distribution: {q_counts.cpu().numpy()}")
    return output