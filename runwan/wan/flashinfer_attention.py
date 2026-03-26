# FlashInfer Attention Wrapper for Wan Model
import sys
sys.path.insert(0, "/workspace")

import torch
import math

def flashinfer_attention_forward(q, k, v):
    """
    使用 FlashInfer 实现的 attention
    
    Args:
        q: [B, H, S, D]
        k: [B, H, S, D]  
        v: [B, H, S, D]
    
    Returns:
        output: [B, H, S, D]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # FlashInfer 期望的格式: [seq_len, batch_size * num_heads, head_dim]
    q_flashinfer = q.flatten(0, 1).permute(1, 0, 2).contiguous()
    k_flashinfer = k.flatten(0, 1).permute(1, 0, 2).contiguous()
    v_flashinfer = v.flatten(0, 1).permute(1, 0, 2).contiguous()
    
    # 导入 flashinfer (需要设置环境变量)
    import flashinfer
    
    # 使用 FlashInfer 的 single_prefill_with_kv_cache
    output, _ = flashinfer.single_prefill_with_kv_cache(
        q_flashinfer,
        k_flashinfer,
        v_flashinfer,
        causal=False,
        return_lse=True,
    )
    
    # 恢复形状: [B, H, S, D]
    output = output.permute(1, 0, 2).reshape(batch_size, num_heads, seq_len, head_dim)
    
    return output
