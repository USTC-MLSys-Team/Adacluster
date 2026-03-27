from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.models.attention_processor import Attention

import torch
import torch.nn.functional as F

import time, sys
import math
import os
from flash_attn import flash_attn_func
from torch.cuda.nvtx import range_push, range_pop

from triton_kernel.fast_kmeans import flash_kmeans
from triton_kernel.fast_kmeans_single import flash_kmeans_single
from triton_kernel.triton_cluster_sparse_attn import triton_cluster_sparse_attn
import torch
import pickle
from datetime import datetime
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
        #print(f"[Timer] {name}: {elapsed:.3f} ms")  # 打印耗时
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
def calculate_clustering_error(query_kernel, query_states, cluster_indices_for_query):
    """
    Calculates the clustering error (sum of squared distances) for k-means.

    Args:
        query_kernel (torch.Tensor): Cluster centers. Shape [batch, num_head, kernel_num, head_dim].
        query_states (torch.Tensor): Original data points. Shape [batch, num_head, seq_len, head_dim].
        cluster_indices_for_query (torch.Tensor): Assigned cluster index for each query_state.
                                                 Shape [batch, num_head, seq_len].

    Returns:
        torch.Tensor: The total clustering error (a scalar tensor).
    """

    batch_size, num_head, seq_len, head_dim = query_states.shape
    expanded_cluster_indices = cluster_indices_for_query.unsqueeze(-1)

    assigned_centers = torch.gather(
        query_kernel.to(torch.float32),
        dim=2, # Gather along the 'kernel_num' dimension (index 2)
        index=expanded_cluster_indices.expand(-1, -1, -1, head_dim).to(torch.int64) # Expand index to match head_dim
    )

    error = torch.nn.functional.mse_loss(query_states, assigned_centers)

    return error

class HunyuanVideoAttnProcessor2_0_kvclus:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
        self.use_full_attention = False
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #torch.cuda.memory._record_memory_history(max_entries=100000)
 
        
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape
        #print("Batch size:", batch_size, "Sequence length:", sequence_length)


        timer.push("QKV proj")

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
 
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        timer.pop()

        timer.push("QK norm")

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        timer.pop()
        timer.push("QK rotary")

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        timer.pop()
        timer.push("Encoder condition QKV proj")

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        timer.pop()
        timer.push("KMeans clustering")
        

        # 5. Attention
        
        q_length = query.shape[2]
        if attention_mask is not None:
            kv_length = torch.sum(attention_mask)
        else:
            kv_length = key.shape[2]
        key = key[:, :, : kv_length, :]
        value = value[:, :, : kv_length, :]


        layer_num=timer.layer()

        if (timer.step()<8 or layer_num<=17 or layer_num==17 or layer_num==34 or layer_num==38 or layer_num==39):##choose layer to use fullattn
            # print("using full attn in layer" ,timer.layer())
            # 5. Attention
            timer.push("fullattn")
            hidden_states = flash_attn_func(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                causal=False,
                softmax_scale=1.0 / math.sqrt(head_dim),
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)
            #print("Full hidden states shape", hidden_states.shape)
            timer.pop()

            range_pop()
            range_push("Output projection")

            # 6. Output projection
            if encoder_hidden_states is not None:
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : -encoder_hidden_states.shape[1]],
                    hidden_states[:, -encoder_hidden_states.shape[1] :],
                )

                if getattr(attn, "to_out", None) is not None:
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)

                if getattr(attn, "to_add_out", None) is not None:
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            range_pop()
            #print("Final hidden states shape", hidden_states.shape)

            return hidden_states, encoder_hidden_states


        # compression_ratio = int(os.environ.get("C_RATIO", 4))

        topk_num = int(os.environ.get("TOPK_NUM", 94))
        q_kernel_num = int(os.environ.get("Q_KERNEL_NUM", 100))
        kv_kernel_num = int(os.environ.get("KV_KERNEL_NUM", 500))
        q_kernel_num = int(os.environ.get("Q_KERNEL_NUM", 100))
        kv_kernel_num = int(os.environ.get("KV_KERNEL_NUM", 500))

        # if q_kernel_num == 0:
        #     q_kernel_num = int(q_length // compression_ratio)
        # if kv_kernel_num == 0:
        #     kv_kernel_num = int(kv_length // compression_ratio)
        
        if not hasattr(attn, "prev_centroid"):
            #print(f"Layer {attn.layer_idx} using kvclus with Q kernel num {q_kernel_num} and KV kernel num {kv_kernel_num}")
            #print("q_length", q_length, "kv_length", kv_length)
            


            q_kernel_num = 250
            kv_kernel_num = 1243


            # 使用确定的聚类数量初始化KV聚类
            random_indices_for_kv = torch.randperm(kv_length, device=key.device)[:kv_kernel_num]
            random_indices_for_kv = random_indices_for_kv.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
            key_kernel = key.gather(2, random_indices_for_kv.unsqueeze(-1).expand(-1, -1, -1, head_dim))            

                
            random_indices_for_query = torch.randperm(q_length, device=query.device)[:q_kernel_num]
            random_indices_for_query = random_indices_for_query.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
            query_kernel = query.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            print(f"Layer {attn.layer_idx} determined KV kernel num: {kv_kernel_num}, Q kernel num: {q_kernel_num}")
            # 使用确定的聚类数量运行flash_kmeans_single
            key_kernel, count_kernel_for_kv, cluster_indices_for_kv = flash_kmeans_single(
                key_kernel, key, 3
            )
            query_kernel, count_kernel_for_q, cluster_indices_for_query = flash_kmeans_single(
                query_kernel, query, 3
            )
            
            # 保存确定的聚类数量供后续使用
            attn.q_kernel_num = q_kernel_num
            attn.kv_kernel_num = kv_kernel_num
            
            print(f"Layer {attn.layer_idx} determined Q kernel num: {q_kernel_num}, KV kernel num: {kv_kernel_num}")
        else:
            print("q_length", q_length, "kv_length", kv_length)
            if self.use_full_attention:
                hidden_states = flash_attn_func(
                    query.transpose(1, 2).contiguous(),
                    key.transpose(1, 2).contiguous(),
                    value.transpose(1, 2).contiguous(),
                    causal=False,
                    softmax_scale=1.0 / math.sqrt(head_dim),
                )
                hidden_states = hidden_states.flatten(2, 3)
                hidden_states = hidden_states.to(query.dtype)


                # 6. Output projection
                if encoder_hidden_states is not None:
                    hidden_states, encoder_hidden_states = (
                        hidden_states[:, : -encoder_hidden_states.shape[1]],
                        hidden_states[:, -encoder_hidden_states.shape[1] :],
                    )

                    if getattr(attn, "to_out", None) is not None:
                        hidden_states = attn.to_out[0](hidden_states)
                        hidden_states = attn.to_out[1](hidden_states)

                    if getattr(attn, "to_add_out", None) is not None:
                        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    

                return hidden_states, encoder_hidden_states
                # 继续使用全注意力
            # 使用之前确定的聚类数量
            q_kernel_num = getattr(attn, 'q_kernel_num', 100)
            kv_kernel_num = getattr(attn, 'kv_kernel_num', 500)
            
            # 从之前保存的聚类中心开始
            query_kernel, key_kernel = attn.prev_centroid
            
            # 使用之前确定的聚类数量
            random_indices_for_query = torch.randperm(q_length, device=query.device)[:q_kernel_num]
            random_indices_for_query = random_indices_for_query.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
            query_kernel = query.gather(2, random_indices_for_query.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            
            random_indices_for_kv = torch.randperm(kv_length, device=key.device)[:kv_kernel_num]
            random_indices_for_kv = random_indices_for_kv.unsqueeze(0).unsqueeze(0).expand(batch_size, attn.heads, -1)
            key_kernel = key.gather(2, random_indices_for_kv.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            
            query_kernel, count_kernel_for_q, cluster_indices_for_query = flash_kmeans_single(
                query_kernel, query, 1
            )
            key_kernel, count_kernel_for_kv, cluster_indices_for_kv = flash_kmeans_single(
                key_kernel, key, 1
            )

        attn.prev_centroid = (query_kernel, key_kernel)

        timer.pop()
        timer.push("Cluster Score")

        print("Q Clustering Error", calculate_clustering_error(query_kernel, query, cluster_indices_for_query))
        print("K Clustering Error", calculate_clustering_error(key_kernel, key, cluster_indices_for_kv))
        #sys.stdout.flush()


        attn_weights = torch.matmul(key_kernel, query_kernel.transpose(2, 3)) / math.sqrt(head_dim) # [kernel_num, query_num]

        # attn_weights 现在的形状是 [B, H, q_length, kv_kernel_num]
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
        timer.push("TopK selection")

        # topk_num = int(attn.kv_kernel_num * 0.25)
        topk_indices = topk_from_qkv_minmax(query_kernel, key_kernel, topk=topk_num)
        #print("shape of topk_indices",topk_indices.shape)
        #print(topk_indices)

        #topk_indices = attn_weights.topk(k=topk_num, dim=-1).indices
        #print("shape of topk_indices",topk_indices.shape)
        #print(topk_indices)

        compressed_mask = torch.zeros_like(attn_weights, dtype=torch.bool)

        compressed_mask.scatter_(
            dim=-1,
            index=topk_indices,
            value=True
        )

        # matmul_count_kernel = torch.matmul(count_kernel_for_q.to(torch.float32), count_kernel_for_kv.transpose(2, 3).to(torch.float32)).to(torch.int32)

        # count_kernel_sum = compressed_mask * matmul_count_kernel

        # print("Sparsity ratio", torch.sum(count_kernel_sum) / (batch_size * attn.heads * q_length * kv_length))
        # sys.stdout.flush()

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
        timer.push(f"Triton cluster sparse attn {attn.layer_idx}")
        #print("Reordered Q shape", reordered_query.shape, "Reordered K shape", reordered_key.shape, "Reordered V shape", reordered_value.shape)
        #print("Compressed mask shape", compressed_mask.shape, "Q counts shape", count_kernel_for_q.shape, "KV counts shape", count_kernel_for_kv.shape) 

        hidden_states = triton_cluster_sparse_attn(
            query=reordered_query,
            key=reordered_key,
            value=reordered_value,
            compressed_attn_mask=compressed_mask,
            q_counts=count_kernel_for_q,
            kv_counts=count_kernel_for_kv,
            sm_scale=1.0 / math.sqrt(head_dim),
        )

        hidden_states = torch.gather(hidden_states, 2, recovery_q_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        timer.pop()
        timer.push("Output projection")

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states,encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
               
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        timer.pop()


        return hidden_states, encoder_hidden_states