import torch
import torch.nn as nn
from einops import rearrange

from module.blocks import Downsampling, ImageBWarp, PixelShuffleBlock, Upsampling
from module.CDMR import CDMR
from module.FFC import FFC_BN_ACT
from module.KCSSM import KCSRBlock
from module.modules import MSA

class Net_D(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.dim
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        ds_kernel_size = config.ds_kernel_size
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        layer_norm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )

        self.temporal_module = getattr(config, "temporal_module", "cdmr").lower()
        self.memory_enabled = False
        self.memory_state = None
        self.last_memory_stats = {}
        if self.temporal_module == "cdmr":
            self.cdmr = CDMR(
                dim=dim,
                num_seq=num_seq,
                memory_momentum=getattr(config, "memory_momentum", 0.9),
                motion_threshold=getattr(config, "motion_threshold", 1.0),
                confidence_threshold=getattr(config, "confidence_threshold", 0.5),
                bias=bias,
            )
        elif self.temporal_module == "none":
            self.cdmr = None
        else:
            raise ValueError(f"Unsupported temporal_module: {self.temporal_module}")

        self.msa = MSA(
            dim,
            num_seq,
            growth_rate,
            num_dense_layer,
            num_flow,
            num_transformer_block,
            num_heads,
            layer_norm_type,
            ffn_expansion_factor,
            bias,
            is_DA=False,
            is_first_f=True,
            is_first_Fw=True,
        )

        self.f_conv = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.d_conv = nn.Sequential(
            nn.Conv3d(dim // num_seq, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, ds_kernel_size * ds_kernel_size, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.a_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_frames, _, _ = x.shape
        F = self.feature_extractor(x)
        if self.cdmr is not None:
            prev_memory = self.memory_state if self.memory_enabled else None
            F, self.memory_state, self.last_memory_stats = self.cdmr(
                F,
                prev_memory=prev_memory,
                update_memory=self.memory_enabled,
            )
        F0_c = F[:, :, num_frames // 2 : num_frames // 2 + 1, :, :]

        F, Fw, f = self.msa(F, None, None, F0_c)
        Fw = rearrange(Fw, "b (c t) h w -> b c t h w", t=num_frames, c=self.dim // num_frames)
        KD = self.d_conv(Fw)

        f_Y = self.f_conv(f)
        anchor = self.a_conv(F)
        return F, KD, f_Y, f, anchor

    def reset_memory(self) -> None:
        self.memory_state = None

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_enabled = bool(enabled)

    def get_memory_stats(self) -> dict:
        return self.last_memory_stats


class Net_R(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_channels = config.in_channels
        dim = config.dim
        num_seq = config.num_seq
        ds_kernel_size = config.ds_kernel_size
        us_kernel_size = config.us_kernel_size
        growth_rate = config.growth_rate
        num_dense_layer = config.num_dense_layer
        num_flow = config.num_flow
        num_transformer_block = config.num_transformer_block
        num_heads = config.num_heads
        layer_norm_type = config.LayerNorm_type
        ffn_expansion_factor = config.ffn_expansion_factor
        bias = config.bias
        scale = config.scale

        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels + dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )

        self.temporal_module = getattr(config, "temporal_module", "cdmr").lower()
        self.memory_enabled = False
        self.memory_state = None
        self.last_memory_stats = {}
        if self.temporal_module == "cdmr":
            self.cdmr = CDMR(
                dim=dim,
                num_seq=num_seq,
                memory_momentum=getattr(config, "memory_momentum", 0.9),
                motion_threshold=getattr(config, "motion_threshold", 1.0),
                confidence_threshold=getattr(config, "confidence_threshold", 0.5),
                bias=bias,
            )
        elif self.temporal_module == "none":
            self.cdmr = None
        else:
            raise ValueError(f"Unsupported temporal_module: {self.temporal_module}")

        self.msa = MSA(
            dim,
            num_seq,
            growth_rate,
            num_dense_layer,
            num_flow,
            num_transformer_block,
            num_heads,
            layer_norm_type,
            ffn_expansion_factor,
            bias,
            is_DA=True,
            is_first_Fw=True,
        )

        self.f_conv1 = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.d_conv = nn.Sequential(
            nn.Conv3d(ds_kernel_size * ds_kernel_size, dim, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias),
        )
        self.res_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias)
        self.res_conv2 = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = PixelShuffleBlock(dim, bias=bias)

        self.pre_ffc_conv1 = nn.Conv2d(dim, 96, kernel_size=1, padding=0, stride=1, bias=bias)
        self.ffc = FFC_BN_ACT(96, 96)
        self.back_ffc_conv2 = nn.Conv2d(96, dim, kernel_size=1, padding=0, stride=1, bias=bias)

        self.r_conv = nn.Sequential(
            nn.Conv3d(dim // num_seq, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, us_kernel_size * us_kernel_size * scale * scale, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv3d(3 * num_flow, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.f_conv3 = nn.Sequential(
            nn.Conv3d(3 * num_flow + 3 + 3, 3 * num_flow, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(3 * num_flow, 3, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )
        self.kcsr_block = KCSRBlock(
            dim=dim,
            num_seq=num_seq,
            d_state=getattr(config, "kcssm_d_state", 16),
            ssm_expand=getattr(config, "kcssm_expand", 2.0),
            local_kernel_size=getattr(config, "kcssm_local_kernel", 3),
            bias=bias,
            enable_local_refine=getattr(config, "kcssm_enable_local", True),
        )

        self.bwarp = ImageBWarp(1, num_seq)
        self.duf = Upsampling(us_kernel_size, scale)
        self.a_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(dim, in_channels, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=1, bias=bias),
        )

    def forward(
        self,
        x: torch.Tensor,
        F: torch.Tensor,
        f: torch.Tensor,
        KD: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, num_frames, _, _ = x.shape

        F = self.feature_extractor(torch.cat([x, F], dim=1))
        if self.cdmr is not None:
            prev_memory = self.memory_state if self.memory_enabled else None
            F, self.memory_state, self.last_memory_stats = self.cdmr(
                F,
                prev_memory=prev_memory,
                update_memory=self.memory_enabled,
            )
        F0_c = F[:, :, num_frames // 2 : num_frames // 2 + 1, :, :]

        f = self.f_conv1(f)
        KD = self.d_conv(KD)
        F, Fw, f = self.msa(F, None, f, F0_c, KD)

        Fw = self.kcsr_block(Fw, KD)
        res = self.relu(self.res_conv1(Fw))
        res = self.relu(self.pre_ffc_conv1(res))
        res = self.relu(self.ffc(res))
        res = self.relu(self.back_ffc_conv2(res))
        res = self.upsample(res)
        res = self.res_conv2(res)

        KR = rearrange(Fw, "b (c t) h w -> b c t h w", t=num_frames)
        KR = self.r_conv(KR)

        f_X = self.f_conv2(f)
        _, warped_X = self.bwarp(x, f_X)
        f_X = self.f_conv3(torch.cat([f, warped_X, x[:, :, num_frames // 2 : num_frames // 2 + 1, :, :].repeat([1, 1, num_frames, 1, 1])], dim=1))

        _, warped_X = self.bwarp(x, f_X)
        output = self.duf(warped_X, KR) + res
        anchor = self.a_conv(F)
        return output, warped_X, anchor

    def reset_memory(self) -> None:
        self.memory_state = None

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_enabled = bool(enabled)

    def get_memory_stats(self) -> dict:
        return self.last_memory_stats

class STENet(nn.Module):
    """Stage-aware video super-resolution network with single-block attention refinement."""

    def __init__(self, config) -> None:
        super().__init__()
        self.stage = config.stage
        self.degradation_learning_network = Net_D(config)
        self.bwarp = ImageBWarp(config.scale, config.num_seq)
        self.ddf = Downsampling(config.ds_kernel_size, config.scale)

        if self.stage == 2:
            self.restoration_network = Net_R(config)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        result_dict: dict[str, torch.Tensor] = {}

        F, KD, f_Y, f, anchor_D = self.degradation_learning_network(x)

        if y is not None:
            ones, warped_Y = self.bwarp(y, f_Y)
            recon = self.ddf(warped_Y, KD, ones)
            result_dict["recon"] = recon
            result_dict["hr_warp"] = warped_Y
            result_dict["image_flow"] = f_Y[:, :2, :, :, :]
            result_dict["F_sharp_D"] = anchor_D

        if self.stage == 1:
            return result_dict

        output, warped_X, anchor_R = self.restoration_network(x, F, f, KD)
        result_dict["output"] = output
        result_dict["lr_warp"] = warped_X
        result_dict["F_sharp_R"] = anchor_R
        return result_dict

    def reset_memory(self) -> None:
        self.degradation_learning_network.reset_memory()
        if self.stage == 2:
            self.restoration_network.reset_memory()

    def set_memory_enabled(self, enabled: bool) -> None:
        self.degradation_learning_network.set_memory_enabled(enabled)
        if self.stage == 2:
            self.restoration_network.set_memory_enabled(enabled)

    def get_memory_stats(self) -> dict:
        stats = {"D": self.degradation_learning_network.get_memory_stats()}
        if self.stage == 2:
            stats["R"] = self.restoration_network.get_memory_stats()
        return stats
