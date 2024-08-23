# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Paper: https://arxiv.org/html/2405.16605v1
# -----------------------------------------------------------------------

import torch
import torch.nn as nn


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def forward(self, x):
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (self.base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1).to(x.device)
        
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, h, w, c = x.shape
        n = h * w
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        if q_rope.dtype != q.dtype:
            q_rope = q_rope.half()
            k_rope = k_rope.half()

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, c1, c2, num_heads=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = c1
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.cpe1 = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.norm1 = norm_layer(self.dim)
        self.in_proj = nn.Linear(self.dim, self.dim)
        self.act_proj = nn.Linear(self.dim, self.dim)
        self.dwc = nn.Conv2d(self.dim, self.dim, 3, padding=1, groups=self.dim)
        self.attn = LinearAttention(dim=self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.dim, c2)

    def forward(self, x):
        # input size must be batch_size, channl, high, weigh
        B, C, H, W = x.shape

        x = (x + self.cpe1(x)).permute(0, 2, 3, 1)  # B H W C

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)

        # Linear Attention  replace SSM with Attention
        x = self.attn(x).reshape(B, H, W, C)
        x = self.out_proj(x * act_res).permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':

    x1 = torch.randn(8, 64, 32, 128).cpu()  # 2d
    net_n = MLLABlock(128, 128).cpu()

    # test
    y1 = net_n(x1)
    print(y1.shape)
