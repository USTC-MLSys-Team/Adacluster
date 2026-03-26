import torch
import flashinfer

def flashinfer_cluster_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compressed_attn_mask: torch.Tensor,
    q_counts: torch.Tensor,
    kv_counts: torch.Tensor,
    sm_scale: float,
    use_cache: bool = True,
    return_stats: bool = False,
):
    """
    FlashInfer Cluster Sparse Attention - 支持 float32/float16/bfloat16
    
    核心修复: FlashInfer 只支持 float16/bfloat16，输入是 float32 时需要转换
    """
    B, H, Q_LEN, D = query.shape
    device = query.device
    dtype = query.dtype
    
    # FlashInfer 只支持 float16/bfloat16
    if dtype == torch.float32:
        compute_dtype = torch.float16
    else:
        compute_dtype = dtype
    
    Q_KERNEL = compressed_attn_mask.shape[2]
    KV_KERNEL = compressed_attn_mask.shape[3]
    
    if q_counts.dim() == 4 and q_counts.shape[-1] == 1:
        q_counts = q_counts.squeeze(-1)
    if kv_counts.dim() == 4 and kv_counts.shape[-1] == 1:
        kv_counts = kv_counts.squeeze(-1)
    
    if return_stats:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    # 转换为计算 dtype
    q_compute = query.to(compute_dtype) if dtype != compute_dtype else query
    k_compute = key.to(compute_dtype) if dtype != compute_dtype else key
    v_compute = value.to(compute_dtype) if dtype != compute_dtype else value
    
    output = torch.zeros_like(q_compute)
    
    q_counts_cpu = q_counts.cpu()
    kv_counts_cpu = kv_counts.cpu()
    mask_cpu = compressed_attn_mask.cpu()
    
    for b in range(B):
        for h in range(H):
            for qk in range(Q_KERNEL):
                q_start = 0 if qk == 0 else q_counts_cpu[b, h, qk - 1].item()
                q_end = q_counts_cpu[b, h, qk].item()
                
                if q_start >= q_end:
                    continue
                
                q_slice = q_compute[b, h, q_start:q_end, :]
                
                kv_indices_list = []
                for kvk in range(KV_KERNEL):
                    if mask_cpu[b, h, qk, kvk].item():
                        kv_start = 0 if kvk == 0 else kv_counts_cpu[b, h, kvk - 1].item()
                        kv_end = kv_counts_cpu[b, h, kvk].item()
                        if kv_start < kv_end:
                            kv_indices_list.append(torch.arange(
                                kv_start, kv_end, dtype=torch.long, device=device
                            ))
                
                if not kv_indices_list:
                    continue
                
                kv_indices = torch.cat(kv_indices_list)
                k_slice = k_compute[b, h, kv_indices, :]
                v_slice = v_compute[b, h, kv_indices, :]
                
                q_expanded = q_slice.unsqueeze(1)
                k_expanded = k_slice.unsqueeze(1)
                v_expanded = v_slice.unsqueeze(1)
                
                out = flashinfer.single_prefill_with_kv_cache(
                    q_expanded,
                    k_expanded,
                    v_expanded,
                    causal=False,
                    sm_scale=sm_scale
                )
                
                output[b, h, q_start:q_end, :] = out.squeeze(1)
    
    # 转换回原始 dtype
    if output.dtype != dtype:
        output = output.to(dtype)
    
    if return_stats:
        end_event.record()
        torch.cuda.synchronize()
        stats = {'total_time_ms': start_event.elapsed_time(end_event)}
        return output, stats
    
    return output


def clear_cache():
    pass


flashinfer_cluster_sparse_attn_v2 = flashinfer_cluster_sparse_attn
flashinfer_cluster_sparse_attn_v3 = flashinfer_cluster_sparse_attn
