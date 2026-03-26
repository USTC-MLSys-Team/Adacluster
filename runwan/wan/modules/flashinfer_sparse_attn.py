import torch
import sys
sys.path.insert(0, /workspace)
import flashinfer

def flashinfer_cluster_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compressed_attn_mask: torch.Tensor,
    q_counts: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
):
    
