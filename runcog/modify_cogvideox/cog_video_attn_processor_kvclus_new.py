import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention
import math
from flash_attn import flash_attn_func
import os

from triton_kernel.fast_kmeans import flash_kmeans
from triton_kernel.fast_kmeans_single import flash_kmeans_single
from triton_kernel.triton_cluster_sparse_attn import triton_cluster_sparse_attn

import time, sys

from torch.cuda.nvtx import range_push, range_pop

    
def thresholded_kmeans_loop(
    initial_kernel: torch.Tensor,  # 初始聚类中心
    data: torch.Tensor,
    iter_time: int = 3,
    distance_threshold: float = 10,
    max_iterations: int = 20,
):
    """
    迭代执行 k-means 聚类，直到所有点都被分配到合适的簇中，或者簇的数量达到总点数的1/3。
    """
    print(f"Starting thresholded k-means with initial kernel shape {initial_kernel.shape}, distance threshold {distance_threshold}, max iterations {max_iterations}")
    print("Data shape:", data.shape)
    
    batch_size, num_head, seq_len, head_dim = data.shape
    max_clusters = seq_len // 3  # 簇数量上限
    total_points = batch_size * num_head * seq_len
    
    # 从初始聚类中心获取初始聚类数量
    kernel_num = initial_kernel.shape[2]
    print(f"Initial kernel num: {kernel_num}")
    
    # 初始化全局的分配结果（全部设为-1表示未分配）
    global_cluster_indices = torch.full((batch_size, num_head, seq_len), -1, 
                                       dtype=torch.int32, device=data.device)
    
    # 当前处理的数据（初始为全部数据）
    current_data = data.clone()
    
    # 使用正确的掩码初始化 - 记录原始索引
    current_global_indices = torch.zeros((batch_size, num_head, seq_len, 3), 
                                        dtype=torch.long, device=data.device)
    b_idx = torch.arange(batch_size, device=data.device).view(-1, 1, 1).expand(batch_size, num_head, seq_len)
    h_idx = torch.arange(num_head, device=data.device).view(1, -1, 1).expand(batch_size, num_head, seq_len)
    s_idx = torch.arange(seq_len, device=data.device).view(1, 1, -1).expand(batch_size, num_head, seq_len)
    current_global_indices = torch.stack([b_idx, h_idx, s_idx], dim=-1)  # [B,H,seq,3]

    
    kernel = initial_kernel
    print(f"Max clusters set to {max_clusters}, starting with {kernel_num} clusters")
    
    # 记录所有聚类中心
    all_kernels = []
    all_counts = []
    current_cluster_offset = 0
    it = 0
    
    while kernel_num > 0 and it < max_iterations:
        it += 1
        print(f"\n--- Iteration {it} ---")
        
        current_seq_len = current_data.shape[2]
        print(f"Processing {current_seq_len} points with {kernel_num} clusters")
        
        if current_seq_len == 0:
            print("No more points to process")
            break
            
        # 运行一次 k-means
        kernel, count_kernel, cluster_indices = flash_kmeans_single(
            kernel, current_data, iter_time=iter_time
        )
        
        # 添加详细的调试信息
        print(f"Kernel shape after k-means: {kernel.shape}")
        print(f"Cluster indices shape: {cluster_indices.shape}")
        print(f"Cluster indices range: {cluster_indices.min().item()} to {cluster_indices.max().item()}")
        
       
        if kernel.shape[2] > 0:  # 确保有聚类中心
            # 检查是否有超出范围的索引
            invalid_indices = (cluster_indices < 0) | (cluster_indices >= kernel.shape[2])
            if invalid_indices.any():
                print(f"Warning: Found {invalid_indices.sum().item()} invalid indices in cluster_indices")
                print(f"Invalid indices range: {cluster_indices[invalid_indices].min().item()} to {cluster_indices[invalid_indices].max().item()}")
            
                cluster_indices = torch.clamp(cluster_indices, 0, kernel.shape[2] - 1)# 修复索引
        
        # 计算每个点到分配中心的距离
        expanded_cluster_indices = cluster_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        # 添加更多的调试信息
        print(f"Expanded cluster indices shape: {expanded_cluster_indices.shape}")
        print(f"Expanded cluster indices range: {expanded_cluster_indices.min().item()} to {expanded_cluster_indices.max().item()}")
        
        # 使用更安全的方式执行 gather 操作
        try:
            assigned_centers = torch.gather(
                kernel, dim=2, index=expanded_cluster_indices.to(torch.int64)
            )
        except RuntimeError as e:
            print(f"Error in torch.gather: {e}")
            print("Kernel shape:", kernel.shape)
            print("Index shape:", expanded_cluster_indices.shape)
            print("Index range:", expanded_cluster_indices.min().item(), "to", expanded_cluster_indices.max().item())
            # 尝试修复问题
            expanded_cluster_indices = torch.clamp(expanded_cluster_indices, 0, kernel.shape[2] - 1)
            assigned_centers = torch.gather(
                kernel, dim=2, index=expanded_cluster_indices.to(torch.int64)
            )
        
        print("Assigned centers shape:", assigned_centers.shape)


        # 转成 float32 再计算 norm
        # center_norms = torch.norm(assigned_centers.float(), dim=-1, keepdim=True) + 1e-8
        #print("center norm stats:", center_norms.min().item(), center_norms.max().item(), center_norms.mean().item())

        #import matplotlib.pyplot as plt
        # plt.hist(center_norms.flatten().cpu().numpy(), bins=50)
        # plt.xlabel("Center Norm")
        #plt.ylabel("Count")
        # plt.title("Distribution of Key Kernel Norms")
        #plt.savefig("key_kernel_norms_hist.png")
        # plt.close()


        distances = torch.norm(current_data - assigned_centers, dim=-1) #/ center_norms.squeeze(-1)
        
        position_distances = distances.mean(dim=1, keepdim=True)  # [batch, 1, seq_len]
        batch_distance = distances.mean(dim=(0,1), keepdim=True)  # [batch, 1, 1]
        print("Distances shape:", distances.shape)
        print("position_distances shape:", position_distances.shape)
        print("batch_distance shape:", batch_distance.shape)
        position_distances = batch_distance.expand(2,num_head,position_distances.shape[2])
        print("batch_distance expanded shape:", position_distances.shape)
        #position_distances = position_distances.expand(-1, num_head, -1)
        #print("position_distances expanded shape:", position_distances.shape)
        print("Distances stats - min:", distances.min().item(), "max:", distances.max().item(), "mean:", distances.mean().item())
        print("Position distances stats - min:", position_distances.min().item(), "max:", position_distances.max().item(), "mean:", position_distances.mean().item())
        
        # 使用位置级别的距离来决定分配
        unassigned_mask_current = position_distances > distance_threshold
    # [B, 1, seq] → 每个 batch 的最大距离

        print("Unassigned mask stats - sum:", unassigned_mask_current.sum().item(), "mean:", unassigned_mask_current.float().mean().item())
        
        unassigned_count = torch.sum(unassigned_mask_current).item()
        assigned_count = torch.sum(~unassigned_mask_current).item()
        
        print(f"{unassigned_count} unassigned points (threshold={distance_threshold})")
        print(f"{assigned_count} assigned points")
        distance_threshold = max(distance_threshold * 1.1,distances.mean().item() * 1.1) 
        # 更新全局分配结果（为已分配的点）
        assigned_mask_current = ~unassigned_mask_current
        
        # 找到当前数据在全局中的位置 - 使用保存的全局索引
        if torch.any(assigned_mask_current):
            # 获取已分配点的全局索引
            assigned_global_indices = current_global_indices[assigned_mask_current]
            
            # 更新全局分配索引（加上当前的偏移量）
            assigned_cluster_indices = cluster_indices[assigned_mask_current] + current_cluster_offset
            
            # 将分配结果写回全局
            for idx, (b, h, s) in enumerate(assigned_global_indices):
                global_cluster_indices[b, h, s] = assigned_cluster_indices[idx]
                
            # 保存聚类中心和计数
            all_kernels.append(kernel)
            all_counts.append(count_kernel)
            print("all_kernels shape:", torch.cat(all_kernels, dim=2).shape if all_kernels else (batch_size, num_head, 0, head_dim))
            
            current_cluster_offset += kernel.shape[2]  # 更新聚类偏移量
        
        # 检查是否所有点都已分配
        if unassigned_count == 0:
            print("All points assigned successfully!")
            break
            
        # 准备下一轮迭代：只处理未分配的点
        if unassigned_count > 0:
            print(f"Preparing for next iteration with {unassigned_count} unassigned points")
            
            # 提取未分配点的数据和全局索引
            unassigned_data = current_data[unassigned_mask_current].view(batch_size, num_head,-1,   head_dim)
            unassigned_global_indices = current_global_indices[unassigned_mask_current].view(batch_size, num_head, -1, 3)
            
            # 更新当前数据和全局索引
            current_data = unassigned_data
            current_global_indices = unassigned_global_indices
                    
            # 检查是否达到聚类数量上限
            if current_cluster_offset + kernel_num * 2 >= max_clusters:
                print(f"Cluster count will exceed max limit ({max_clusters}), returning 'fullattn'")
                return -1
                        

            kernel_num = min(kernel_num, max_clusters - current_cluster_offset, current_data.shape[2])

            if current_data.shape[2] > kernel_num:
                idx = torch.randperm(current_data.shape[2], device=data.device)[:kernel_num]
                kernel = current_data[:, :, idx, :].clone()
            else:
                kernel = current_data.clone()
                kernel_num = current_data.shape[2]
                    
            print(f"Next iteration: {current_data.shape[2]} points, {kernel_num} clusters")
        else:
            break
    
    # 合并所有聚类中心和计数
    if len(all_kernels) > 0:
        final_kernel = torch.cat(all_kernels, dim=2)
        final_count = torch.cat(all_counts, dim=2)
        
        # 检查是否有未分配的点
        unassigned_count_final = torch.sum(global_cluster_indices == -1).item()
        if unassigned_count_final > 0:
            print(f"Warning: {unassigned_count_final} points remain unassigned after all iterations")
        
        assigned_count_final = torch.sum(global_cluster_indices != -1).item()
        print(f"Final result: {assigned_count_final}/{total_points} points assigned")
        print(f"Final cluster count: {final_kernel.shape[2]}")
        
        return final_kernel.shape[2]
    else:
        print("No clusters were created, returning 'fullattn'")
        return -1
def topk_from_qkv_minmax(query: torch.Tensor, key: torch.Tensor, topk: int):
    # query: [B, H, q_len, head_dim]
    # key:   [B, H, kv_len, head_dim]
    q_pos = torch.clamp(query, min=0.0)  # [B, H, q_len, head_dim]
    q_neg = torch.clamp(query, max=0.0)

    # 对每个 kv 向量，用 (Kmax, Kmin) 分别替代正负部分
    k_pos = torch.clamp(key, min=0.0)  # [B, H, kv_len, head_dim]
    k_neg = torch.clamp(key, max=0.0)

    # 近似打分: q_pos·k_pos + q_neg·k_neg
    score = torch.matmul(q_pos, k_pos.transpose(-2, -1)) \
          + torch.matmul(q_neg, k_neg.transpose(-2, -1))
    # shape: [B, H, q_len, kv_len]

    topk_indices = score.topk(k=topk, dim=-1).indices

    return topk_indices


class StackTimer:
    def __init__(self, device="cuda"):
        self.stack = []  # 栈，用于存储 (name, start_event, end_event)
        self.records = {}  # 保存每个步骤的耗时总和
        self.device = device

    def push(self, name: str):
        """入栈计时"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.stack.append((name, start_event, end_event))

    def pop(self):
        """出栈计时并记录耗时"""
        if not self.stack:
            raise RuntimeError("Timer stack is empty, cannot pop.")
        
        name, start_event, end_event = self.stack.pop()
        end_event.record()
        torch.cuda.synchronize()  # 等待 GPU 计时完成
        elapsed = start_event.elapsed_time(end_event)  # ms

        if name in self.records:
            self.records[name].append(elapsed)
        else:
            self.records[name] = [elapsed]
        print(f"[Timer] {name}: {elapsed:.3f} ms")  # 打印耗时
        return elapsed

    def get_stats(self):
        """返回每个步骤的平均耗时（ms）"""
        stats = {}
        for name, times in self.records.items():
            stats[name] = {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "total_ms": sum(times)
            }
        return stats
    def layer(self):
        for name, times in self.records.items():
            return len(times) % 60
    def step(self):
        for name, times in self.records.items():
            return int(len(times)/60)
    def reset(self):
        self.stack.clear()
        self.records.clear()

timer = StackTimer()
def topp_sampling(probs: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """
    Top-p sampling (nucleus sampling) implementation.
    
    Args:
        probs: Probability distribution tensor of shape [batch_size, heads, seq_len, seq_len]
        p: Cumulative probability threshold (0 < p <= 1)
        min_tokens_to_keep: Minimum number of tokens to keep
    
    Returns:
        Binary mask tensor with the same shape as probs
    """
    if p >= 1.0:
        return torch.ones_like(probs, dtype=torch.bool)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to remove (outside top-p)
    sorted_mask = cumulative_probs > p
    
    # Ensure we keep at least min_tokens_to_keep tokens
    if min_tokens_to_keep > 1:
        sorted_mask[..., :min_tokens_to_keep] = False
    
    # Scatter the mask back to original order
    original_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    original_mask.scatter_(-1, sorted_indices, sorted_mask)
    
    # Invert mask to get tokens to keep
    keep_mask = ~original_mask
    
    return keep_mask

class StackTimer:
    def __init__(self, device="cuda"):
        self.stack = []  # 栈，用于存储 (name, start_event, end_event)
        self.records = {}  # 保存每个步骤的耗时总和
        self.device = device

    def push(self, name: str):
        """入栈计时"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.stack.append((name, start_event, end_event))

    def pop(self):
        """出栈计时并记录耗时"""
        if not self.stack:
            raise RuntimeError("Timer stack is empty, cannot pop.")
        
        name, start_event, end_event = self.stack.pop()
        end_event.record()
        torch.cuda.synchronize()  # 等待 GPU 计时完成
        elapsed = start_event.elapsed_time(end_event)  # ms

        if name in self.records:
            self.records[name].append(elapsed)
        else:
            self.records[name] = [elapsed]
        print(f"[Timer] {name}: {elapsed:.3f} ms")  # 打印耗时
        return elapsed

    def get_stats(self):
        """返回每个步骤的平均耗时（ms）"""
        stats = {}
        for name, times in self.records.items():
            stats[name] = {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "total_ms": sum(times)
            }
        return stats

    def reset(self):
        self.stack.clear()
        self.records.clear()
timer = StackTimer()
def compute_cluster_indices(kernel_k, k):
    kernel_norm = kernel_k.norm(dim=-1)
    kernel_norm = kernel_norm * kernel_norm
    
    # dis = kernel_norm.unsqueeze(2) - 2 * torch.matmul(k, kernel_k.transpose(2, 3))
    # min_indices = torch.argmin(dis, dim=-1)

    BATCH_SIZE, N_HEADS = kernel_k.shape[0], kernel_k.shape[1]
    KEY_NUM = k.shape[2]

    min_indices = torch.zeros((BATCH_SIZE, N_HEADS, KEY_NUM), dtype=torch.int64, device=kernel_k.device)

    for i in range(N_HEADS):
        dis = kernel_norm[:, i, :].unsqueeze(1) - 2 * torch.matmul(k[:, i, :], kernel_k[:, i, :].transpose(1, 2))
        min_indices[:, i, :] = torch.argmin(dis, dim=-1)

    return dis, min_indices


def compute_new_kernel(min_indices, kernel_k, k, kernel_v, v, kernel_count, count):
    # print(min_indices.shape, kernel_k.shape, k.shape)
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
        kernel_count[:, i, :, :] = onehot_dis.sum(dim=-1, keepdim=True)

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

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        timer.push("QKV proj")

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape
        print("hidden_states shape:", hidden_states.shape)
        print("encoder_hidden_states shape:", encoder_hidden_states.shape)
        print("batch_size:", batch_size)
        print("sequence_length:", sequence_length)
        print("text_seq_length:", text_seq_length)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        print("attention_mask shape:", attention_mask.shape if attention_mask is not None else None)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        print("inner_dim:", inner_dim)
        print("head_dim:", head_dim)    
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        print("query shape:", query.shape)
        print("key shape:", key.shape)
        print("value shape:", value.shape)#shape: [B, H, seq_len, head_dim]

        timer.pop()
        timer.push("QK norm")

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        timer.pop()
        timer.push("RoPE")

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        timer.pop()
        timer.push("Kernel Initialization")

        seq_length = query.shape[2]
        count_states = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        count_states_for_q = torch.ones(batch_size, attn.heads, seq_length, 1, device=query.device, dtype=torch.int32)
        
        compression_ratio = int(os.environ.get("C_RATIO", 4))
        topk_num = int(os.environ.get("TOPK_NUM", 64))
        q_kernel_num = int(os.environ.get("Q_KERNEL_NUM", 100))
        kv_kernel_num = int(os.environ.get("KV_KERNEL_NUM", 500))

        kernel_num = int(seq_length // compression_ratio)
        if q_kernel_num == 0:
            q_kernel_num = kernel_num
        if kv_kernel_num == 0:
            kv_kernel_num = kernel_num
        q_length = query.shape[2]
        if attention_mask is not None:
            kv_length = torch.sum(attention_mask)
        else:
            kv_length = key.shape[2]
        initial_kkernel_num = 250  # 初始聚类中心数量
        initial_qkernel_num = 50
        random_indices_for_kv_initial = torch.randperm(kv_length, device=key.device)[:initial_kkernel_num]
        key_initial_kernel = key[:, :, random_indices_for_kv_initial, :].clone()
        
        kv_kernel_num = thresholded_kmeans_loop(
            initial_kernel=key_initial_kernel,
            data=key,
            iter_time=3,
            distance_threshold=3,
            max_iterations=10
        )
        random_indices_for_query_initial = torch.randperm(q_length, device=query.device)[:initial_qkernel_num]
        query_initial_kernel = query[:, :, random_indices_for_query_initial, :].clone()
        
        q_kernel_num = thresholded_kmeans_loop(
            initial_kernel=query_initial_kernel,
            data=query,
            iter_time=3,
            distance_threshold=5,
            max_iterations=10
        )
        random_generated = torch.randperm(seq_length, device=query.device)
        random_generated_for_query = torch.randperm(seq_length, device=query.device)
        
        random_indices = random_generated[: kv_kernel_num]
        random_indices = random_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)

        key_kernel = key.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        value_kernel = value.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        count_kernel_for_kv = count_states.gather(2, random_indices.unsqueeze(-1).expand(-1, -1, -1, 1))

        random_indices_for_query = random_generated_for_query[: q_kernel_num]
        random_indices_for_query = random_indices_for_query.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
        
        query_kernel = query.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        count_kernel_for_q = count_states_for_q.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, 1))

        timer.pop()
        timer.push("KV Clustering")

        # key_kernel = key_kernel.to(torch.float32)
        # value_kernel = value_kernel.to(torch.float32)
        key_kernel, value_kernel, count_kernel_for_kv, cluster_indices_for_kv = flash_kmeans(
            key_kernel,
            key,
            value_kernel,
            value,
            count_kernel_for_kv,
            count_states,
        )

        # key_kernel = key_kernel.to(query.dtype)
        # value_kernel = value_kernel.to(query.dtype)

        timer.pop()
        timer.push("Q Clustering")

        # query_kernel = query_kernel.to(torch.float32)
        query_kernel, count_kernel_for_q, cluster_indices_for_query = flash_kmeans_single(
            query_kernel,
            query,
        )
        # query_kernel = query_kernel.to(key_kernel.dtype)

        timer.pop()
        timer.push("TopK Selection")
        timer.push("Attn Weights Calculation")

        attn_weights = torch.matmul(key_kernel, query_kernel.transpose(2, 3)) / math.sqrt(head_dim) # [kernel_num, query_num]
        cluster_bias = torch.where(
            count_kernel_for_kv > 0,
            torch.log(count_kernel_for_kv),
            torch.finfo(attn_weights.dtype).min,
        )
        attn_weights = attn_weights + cluster_bias
        attn_weights = torch.softmax(
            attn_weights.transpose(2, 3),
            dim=-1,
            dtype=torch.float32
        ).to(query.dtype)

        timer.pop()
        timer.push("TopK Selection")
        topk_indices = topk_from_qkv_minmax(query_kernel, key_kernel, topk=topk_num)
        #topk_indices = attn_weights.topk(k=topk_num, dim=-1).indices

        timer.pop()
        timer.pop()

        # for cluster-wise sparsity calculation
        # BLOCK_N = 128
        # tmp_topk_indices = topk_indices.reshape(batch_size, attn.heads, kernel_num * topk_num)
        # selected_key_kernel_count = torch.gather(count_kernel_for_kv, 2, tmp_topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1))
        # selected_key_kernel_count = torch.reshape(selected_key_kernel_count, (batch_size, attn.heads, kernel_num, topk_num))
        
        # aligned_selected_key_kernel_count = (selected_key_kernel_count + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        # aligned_selected_key_kernel_count = torch.sum(aligned_selected_key_kernel_count, dim=-1)
        
        # selected_key_kernel_count = torch.sum(selected_key_kernel_count, dim=-1)
        
        # count_kernel_for_q_tmp = count_kernel_for_q.squeeze(-1)
        # # print(count_kernel_for_q_tmp[1, 29])
        # cluster_count_num = count_kernel_for_q_tmp * selected_key_kernel_count
        
        # aligned_count_kernel_for_q_tmp = (count_kernel_for_q_tmp + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        # aligned_cluster_count_num = aligned_count_kernel_for_q_tmp * aligned_selected_key_kernel_count

        # print("Cluster-Wise QKV theretical computation ratio: ", torch.sum(cluster_count_num) / batch_size / attn.heads / seq_length / seq_length)
        # print("Cluster-Wise QKV real computation ratio: ", torch.sum(aligned_cluster_count_num) / batch_size / attn.heads / seq_length / seq_length)
        # sys.stdout.flush()

        timer.push("Compressed Mask Creation")


        compressed_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

        compressed_mask.scatter_(
            dim=-1,
            index=topk_indices,
            value=True
        )

        # matmul_count_kernel = torch.matmul(count_kernel_for_q.to(torch.float32), count_kernel_for_kv.transpose(2, 3).to(torch.float32)).to(torch.int32)

        # count_kernel_sum = compressed_mask * matmul_count_kernel

        # print("Sparsity ratio", torch.sum(count_kernel_sum) / (batch_size * attn.heads * seq_length * seq_length))
        # sys.stdout.flush()

        timer.pop()

        timer.push("Gathering Reordered QKV")

        count_kernel_for_kv = count_kernel_for_kv.squeeze(-1)
        count_kernel_for_q = count_kernel_for_q.squeeze(-1)

        count_kernel_for_kv = count_kernel_for_kv.to(torch.int32)
        count_kernel_for_q = count_kernel_for_q.to(torch.int32)

        count_kernel_for_kv = torch.cumsum(count_kernel_for_kv, dim=-1)
        count_kernel_for_q = torch.cumsum(count_kernel_for_q, dim=-1)

        count_kernel_for_kv = count_kernel_for_kv.to(torch.int32).contiguous()
        count_kernel_for_q = count_kernel_for_q.to(torch.int32).contiguous()

        sorted_q_indices = torch.argsort(cluster_indices_for_query, dim=-1)
        sorted_kv_indices = torch.argsort(cluster_indices_for_kv, dim=-1)

        recovery_q_indices = torch.argsort(sorted_q_indices, dim=-1)

        reordered_query = torch.gather(query, 2, sorted_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        reordered_key = torch.gather(key, 2, sorted_kv_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()
        reordered_value = torch.gather(value, 2, sorted_kv_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)).contiguous()

        compressed_mask = compressed_mask.contiguous()

        timer.pop()
        timer.push("Cluster Sparse Attention")

        hidden_states = triton_cluster_sparse_attn(
            query=reordered_query,
            key=reordered_key,
            value=reordered_value,
            compressed_attn_mask=compressed_mask,
            q_counts=count_kernel_for_q,
            kv_counts=count_kernel_for_kv,
            sm_scale=1.0 / math.sqrt(head_dim),
        )

        timer.pop()
        timer.push("Gathering Reordered Hidden States")

        hidden_states = torch.gather(hidden_states, 2, recovery_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        timer.pop()
        timer.push("Post Attn")

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        timer.pop()
        print(timer.get_stats())
        #print("layer idx: ", attn.layer_idx)
        return hidden_states, encoder_hidden_states
