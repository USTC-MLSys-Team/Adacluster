from typing import Tuple
import torch

import triton
import triton.language as tl

@triton.jit
def _create_mask_stack(
    mask_stack_ptr,
    cluster_indices_ptr,
    mask_stack_stride0: tl.constexpr,
    mask_stack_stride1: tl.constexpr,
    mask_stack_stride2: tl.constexpr,
    cluster_indices_stride0: tl.constexpr,
    cluster_indices_stride1: tl.constexpr,
    KERNEL_NUM: tl.constexpr,
    KV_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    k_id = tl.program_id(0)
    b_id = tl.program_id(1)
    h_id = tl.program_id(2)
    for i in range((KV_LEN + BLOCK_N - 1) // BLOCK_N):
        cluster_indices_offset = b_id * cluster_indices_stride0 + h_id * cluster_indices_stride1 + i * BLOCK_N + tl.arange(0, BLOCK_N) # [BLOCK_N]
        cluster_indices_offset_max = b_id * cluster_indices_stride0 + h_id * cluster_indices_stride1 + KV_LEN # [1]
        indices_val = tl.load(cluster_indices_ptr + cluster_indices_offset, mask=(cluster_indices_offset < cluster_indices_offset_max), other=KV_LEN) # [BLOCK_N]
        indices_val = indices_val == k_id # [BLOCK_N]
        mask_stack_store_offset = k_id * mask_stack_stride0 + b_id * mask_stack_stride1 + h_id * mask_stack_stride2 + i * BLOCK_N + tl.arange(0, BLOCK_N) # [BLOCK_N]
        mask_stack_store_offset_max = k_id * mask_stack_stride0 + b_id * mask_stack_stride1 + h_id * mask_stack_stride2 + KV_LEN # [1]
        tl.store(mask_stack_ptr + mask_stack_store_offset, indices_val, mask=(mask_stack_store_offset < mask_stack_store_offset_max)) # [BLOCK_N]


def triton_create_mask_stack(
    mask_stack: torch.Tensor, # [kernel_num, BSZ, NUM_HEAD, KV_LEN]
    cluster_indices: torch.Tensor, # [BSZ, NUM_HEAD, KV_LEN]
):
    KERNEL_NUM, BSZ, NUM_HEAD, KV_LEN = mask_stack.shape

    BLOCK_N = 128

    grid = lambda meta: (KERNEL_NUM, BSZ, NUM_HEAD)
    with torch.cuda.device(mask_stack.device):
        _create_mask_stack[grid](
            mask_stack,
            cluster_indices,
            mask_stack.stride(0),
            mask_stack.stride(1),
            mask_stack.stride(2),
            cluster_indices.stride(0),
            cluster_indices.stride(1),
            KERNEL_NUM,
            KV_LEN,
            BLOCK_N=BLOCK_N,
        )

    return mask_stack