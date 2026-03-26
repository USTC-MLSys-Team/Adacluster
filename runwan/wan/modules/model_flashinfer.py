# FlashInfer版本的Wan模型
# 这个模块复制自model.py，但使用FlashInfer实现的cluster sparse attention

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from triton_kernel.fast_kmeans_single import flash_kmeans_single
from triton_kernel.flashinfer_cluster_sparse_attn import flashinfer_cluster_sparse_attn_v2

from .attention import flash_attention

__all__ = ["WanModel""
