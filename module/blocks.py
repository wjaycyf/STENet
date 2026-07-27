from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Downsampling(nn.Module):
    """Dynamic downsampling kernel applicator for stage-one reconstruction."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def kernel_normalize(self, kernel: torch.Tensor) -> torch.Tensor:
        return F.softmax(kernel, dim=-1)

    def forward(self, x: torch.Tensor, kernel: torch.Tensor, DT: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = kernel.shape
        normalized_kernel = rearrange(kernel, "b kk t h w -> b 1 h w (t kk)")
        normalized_kernel = self.kernel_normalize(normalized_kernel)

        pad_size = (self.kernel_size - self.stride) // 2
        padded_x = F.pad(x, (pad_size, pad_size, pad_size, pad_size, 0, 0), mode="replicate")
        unfolded_x = padded_x.unfold(3, self.kernel_size, self.stride)
        unfolded_x = unfolded_x.unfold(4, self.kernel_size, self.stride)
        unfolded_x = rearrange(
            unfolded_x,
            "b c t h w kh kw -> b c h w (t kh kw)",
        )
        aggregated_x = torch.sum(unfolded_x * normalized_kernel, dim=-1)

        padded_DT = F.pad(DT, (pad_size, pad_size, pad_size, pad_size, 0, 0), mode="replicate")
        unfolded_DT = padded_DT.unfold(3, self.kernel_size, self.stride)
        unfolded_DT = unfolded_DT.unfold(4, self.kernel_size, self.stride)
        unfolded_DT = rearrange(
            unfolded_DT,
            "b c t h w kh kw -> b c h w (t kh kw)",
        )
        aggregated_DT = torch.sum(unfolded_DT * normalized_kernel, dim=-1)

        return aggregated_x / (aggregated_DT + 1e-8)


class Upsampling(nn.Module):
    """Dynamic upsampling kernel applicator for stage-two restoration."""

    def __init__(self, kernel_size: int, scale: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = scale

    def kernel_normalize(self, kernel: torch.Tensor) -> torch.Tensor:
        num_weights = kernel.shape[-1]
        centered_kernel = kernel - torch.mean(kernel, dim=-1, keepdim=True)
        return centered_kernel + 1.0 / num_weights

    def forward(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape

        normalized_kernel = rearrange(
            kernel,
            "b (s1 s2 k1 k2) t h w -> b 1 h w s1 s2 (t k1 k2)",
            s1=self.scale,
            s2=self.scale,
            k1=self.kernel_size,
            k2=self.kernel_size,
        )
        normalized_kernel = self.kernel_normalize(normalized_kernel)

        pad_size = self.kernel_size // 2
        flattened_x = rearrange(x, "b c t h w -> (b t) c h w")
        padded_x = F.pad(flattened_x, (pad_size, pad_size, pad_size, pad_size), mode="replicate")
        unfolded_x = F.unfold(padded_x, [self.kernel_size, self.kernel_size], padding=0)
        unfolded_x = rearrange(
            unfolded_x,
            "(b t) (c k1 k2) (h w) -> b c h w 1 1 (t k1 k2)",
            b=batch_size,
            t=num_frames,
            c=channels,
            k1=self.kernel_size,
            k2=self.kernel_size,
            h=height,
            w=width,
        )
        upsampled_x = torch.sum(unfolded_x * normalized_kernel, dim=-1)
        upsampled_x = rearrange(upsampled_x, "b c h w s1 s2 -> b c (h s1) (w s2)")

        return upsampled_x


def backwarp(x: torch.Tensor, flow: torch.Tensor, warp_cache: Dict[str, torch.Tensor]) -> torch.Tensor:
    cache_key = f"grid{flow.dtype}{flow.device}{flow.shape[2]}{flow.shape[3]}"
    if cache_key not in warp_cache:
        horizontal_grid = torch.linspace(
            start=-1.0,
            end=1.0,
            steps=flow.shape[3],
            dtype=flow.dtype,
            device=flow.device,
        ).view(1, 1, 1, -1).repeat(1, 1, flow.shape[2], 1)
        vertical_grid = torch.linspace(
            start=-1.0,
            end=1.0,
            steps=flow.shape[2],
            dtype=flow.dtype,
            device=flow.device,
        ).view(1, 1, -1, 1).repeat(1, 1, 1, flow.shape[3])
        warp_cache[cache_key] = torch.cat([horizontal_grid, vertical_grid], 1)

    if flow.shape[3] == flow.shape[2]:
        normalized_flow = flow * (2.0 / ((flow.shape[3] and flow.shape[2]) - 1.0))
    else:
        normalized_flow = flow * torch.tensor(
            data=[2.0 / (flow.shape[3] - 1.0), 2.0 / (flow.shape[2] - 1.0)],
            dtype=flow.dtype,
            device=flow.device,
        ).view(1, 2, 1, 1)

    return nn.functional.grid_sample(
        input=x,
        grid=(warp_cache[cache_key] + normalized_flow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class ImageBWarp(nn.Module):
    def __init__(self, scale: int, num_seq: int) -> None:
        super().__init__()
        self.scale = scale
        self.num_seq = num_seq
        self.objBackwarpcache: Dict[str, torch.Tensor] = {}
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, "b c t h w -> (b t) c h w")
        f = rearrange(f, "b c t h w -> (b t) c h w")

        weight = self.sigmoid(f[:, 2:3, :, :])
        flow = f[:, :2, :, :]
        ones = torch.ones_like(x)

        if self.scale != 1:
            flow = self.scale * F.interpolate(flow, scale_factor=(self.scale, self.scale), mode="bilinear", align_corners=False)
            weight = F.interpolate(weight, scale_factor=(self.scale, self.scale), mode="bilinear", align_corners=False)

        warped_x = backwarp(x, flow, self.objBackwarpcache)
        warped_ones = backwarp(ones, flow, self.objBackwarpcache) * weight

        warped_x = rearrange(warped_x, "(b t) c h w -> b c t h w", t=self.num_seq)
        warped_ones = rearrange(warped_ones, "(b t) c h w -> b c t h w", t=self.num_seq)
        weight = rearrange(weight, "(b t) c h w -> b c t h w", t=self.num_seq)

        return warped_ones, warped_x * weight


class MultiFlowBWarp(nn.Module):
    def __init__(self, dim: int, num_seq: int, num_flow: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_seq = num_seq
        self.num_flow = num_flow
        self.objBackwarpcache: Dict[str, torch.Tensor] = {}
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, flow_with_weight: torch.Tensor) -> torch.Tensor:
        warped_features = rearrange(
            features,
            "b (n c) t h w -> (b n t) c h w",
            c=self.dim // self.num_flow,
            n=self.num_flow,
        )
        warped_flow = rearrange(
            flow_with_weight,
            "b (n c) t h w -> (b n t) c h w",
            c=3,
            n=self.num_flow,
        )

        weight = self.sigmoid(warped_flow[:, 2:3, :, :])
        flow = warped_flow[:, :2, :, :]

        warped_features = backwarp(warped_features, flow, self.objBackwarpcache) * weight
        return rearrange(warped_features, "(b n t) c h w -> b (n c) t h w", t=self.num_seq, n=self.num_flow)


class PixelShuffleBlock(nn.Module):
    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.shuffle(x)
        x = self.relu(self.conv2(x))
        x = self.shuffle(x)
        x = self.relu(self.conv3(x))
        return x
