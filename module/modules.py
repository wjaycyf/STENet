import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from module.blocks import MultiFlowBWarp


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class DenseLayer(nn.Module):
    def __init__(self, dim: int, growth_rate: int, bias: bool) -> None:
        super().__init__()
        self.conv = nn.Conv3d(dim, growth_rate, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.conv(x))
        return torch.cat((x, out), dim=1)


class RDB(nn.Module):
    def __init__(self, dim: int, growth_rate: int, num_dense_layer: int, bias: bool) -> None:
        super().__init__()
        layers = [
            DenseLayer(dim=dim + growth_rate * index, growth_rate=growth_rate, bias=bias)
            for index in range(num_dense_layer)
        ]
        self.layer = nn.Sequential(*layers)
        self.conv = nn.Conv3d(dim + growth_rate * num_dense_layer, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        out = self.conv(out)
        return out + x


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q = self.q_dwconv(self.q(f))
        kv = self.kv_dwconv(self.kv_conv(x))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=height, w=width)
        return self.project_out(out)


class MultiAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_norm_type: str,
        ffn_expansion_factor: float,
        bias: bool,
        is_da: bool,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.co_attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.is_da = is_da

        if is_da:
            self.norm3 = LayerNorm(dim, layer_norm_type)
            self.da_attn = Attention(dim, num_heads, bias)
            self.norm4 = LayerNorm(dim, layer_norm_type)
            self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, fused_features: torch.Tensor, center_features: torch.Tensor, degradation_features: torch.Tensor | None) -> torch.Tensor:
        fused_features = fused_features + self.co_attn(self.norm1(fused_features), center_features)
        fused_features = fused_features + self.ffn1(self.norm2(fused_features))

        if self.is_da and degradation_features is not None:
            fused_features = fused_features + self.da_attn(self.norm3(fused_features), degradation_features)
            fused_features = fused_features + self.ffn2(self.norm4(fused_features))

        return fused_features


class MSA(nn.Module):
    """Single refinement block for temporal aggregation and attention fusion."""

    def __init__(
        self,
        dim: int,
        num_seq: int,
        growth_rate: int,
        num_dense_layer: int,
        num_flow: int,
        num_multi_attn: int,
        num_heads: int,
        layer_norm_type: str,
        ffn_expansion_factor: float,
        bias: bool,
        is_DA: bool = False,
        is_first_f: bool = False,
        is_first_Fw: bool = False,
    ) -> None:
        super().__init__()
        self.rdb = RDB(dim, growth_rate, num_dense_layer, bias)
        self.rdb_KD = RDB(dim, growth_rate, num_dense_layer, bias) if is_DA else None
        self.conv_KD = nn.Conv2d(dim * num_seq, dim, kernel_size=1, padding=0, stride=1, bias=bias) if is_DA else None

        self.bwarp = MultiFlowBWarp(dim, num_seq, num_flow)
        self.conv_Fw = nn.Conv2d(dim * num_seq if is_first_Fw else dim + dim * num_seq, dim, kernel_size=1, padding=0, stride=1, bias=bias)
        self.conv_f = nn.Sequential(
            nn.Conv3d(dim * 2 if is_first_f else dim * 2 + 3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.multi_attn_block = nn.ModuleList(
            [
                MultiAttentionBlock(dim, num_heads, layer_norm_type, ffn_expansion_factor, bias, is_DA)
                for _ in range(num_multi_attn)
            ]
        )

    def forward(
        self,
        F: torch.Tensor,
        Fw: torch.Tensor | None,
        f: torch.Tensor | None,
        F0_c: torch.Tensor,
        KD: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_frames, _, _ = F.shape
        refined_features = self.rdb(F)

        if f is not None:
            warped_features = self.bwarp(refined_features, f)
            f = f + self.conv_f(torch.cat([F0_c.repeat([1, 1, num_frames, 1, 1]), f, warped_features], dim=1))
        else:
            f = self.conv_f(torch.cat([F0_c.repeat([1, 1, num_frames, 1, 1]), refined_features], dim=1))

        warped_features = self.bwarp(refined_features, f)
        warped_features_2d = rearrange(warped_features, "b c t h w -> b (c t) h w")
        fused_features = self.conv_Fw(torch.cat([Fw, warped_features_2d], dim=1)) if Fw is not None else self.conv_Fw(warped_features_2d)

        refined_KD = None
        if KD is not None and self.rdb_KD is not None and self.conv_KD is not None:
            refined_KD = self.rdb_KD(KD)
            refined_KD = rearrange(refined_KD, "b c t h w -> b (c t) h w")
            refined_KD = self.conv_KD(refined_KD)

        for block in self.multi_attn_block:
            fused_features = block(fused_features, F0_c.squeeze(dim=2), refined_KD)

        return refined_features, fused_features, f
