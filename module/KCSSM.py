import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def _require_selective_scan():
    if selective_scan_fn is None:
        raise ImportError(
            "KCSS2D requires mamba_ssm to be installed so selective_scan_fn is available."
        )


class KernelPromptEncoder(nn.Module):
    def __init__(self, dim, bias=False, fuse_mode="center_global"):
        super().__init__()
        if fuse_mode != "center_global":
            raise ValueError(f"Unsupported fuse_mode: {fuse_mode}")

        self.dim = dim
        self.fuse_mode = fuse_mode
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
        )

    def forward(self, kd_3d):
        if kd_3d.dim() != 5:
            raise ValueError(f"KernelPromptEncoder expects [B,C,T,H,W], got shape={tuple(kd_3d.shape)}")
        if kd_3d.shape[1] != self.dim:
            raise ValueError(f"KernelPromptEncoder dim mismatch: expected {self.dim}, got {kd_3d.shape[1]}")

        _, _, t, _, _ = kd_3d.shape

        center = kd_3d[:, :, t // 2, :, :]
        global_stat = kd_3d.mean(dim=2)

        return self.fuse(torch.cat([center, global_stat], dim=1))


class LocalRefine(nn.Module):
    def __init__(self, dim, kernel_size=3, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


class KCSS2D(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        condition_on=("dt", "B", "C"),
        condition_merge="add",
        alpha_bound=1.0,
        a_log_clamp_min=-3.0,
        a_log_clamp_max=5.0,
    ):
        super().__init__()
        _require_selective_scan()

        valid_condition_targets = {"dt", "B", "C"}
        unknown_targets = set(condition_on) - valid_condition_targets
        if unknown_targets:
            raise ValueError(f"Unsupported condition targets: {sorted(unknown_targets)}")
        if condition_merge != "add":
            raise ValueError(f"Unsupported condition_merge: {condition_merge}")

        self.d_model = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.condition_on = tuple(condition_on)
        self.condition_merge = condition_merge
        self.alpha_bound = float(alpha_bound)
        self.a_log_clamp_min = float(a_log_clamp_min)
        self.a_log_clamp_max = float(a_log_clamp_max)

        self.fw_norm = nn.LayerNorm(self.d_model)
        self.kd_norm = nn.LayerNorm(self.d_model)

        self.x_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.x_conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )

        self.k_in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.k_conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        x_param_layers = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.x_param_weight = nn.Parameter(torch.stack([layer.weight for layer in x_param_layers], dim=0))
        del x_param_layers

        k_param_layers = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.k_param_weight = nn.Parameter(torch.stack([layer.weight for layer in k_param_layers], dim=0))
        del k_param_layers

        x_dt_projs = [
            self._build_dt_proj(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        ]
        self.x_dt_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in x_dt_projs], dim=0))
        self.x_dt_proj_bias = nn.Parameter(torch.stack([proj.bias for proj in x_dt_projs], dim=0))
        del x_dt_projs

        k_dt_projs = [
            self._build_dt_proj(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        ]
        self.k_dt_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in k_dt_projs], dim=0))
        self.k_dt_proj_bias = nn.Parameter(torch.stack([proj.bias for proj in k_dt_projs], dim=0))
        del k_dt_projs

        self.A_logs = self._init_a_log(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._init_d(self.d_inner, copies=4, merge=True)

        self.alpha_dt_raw = nn.Parameter(torch.zeros(1))
        self.alpha_b_raw = nn.Parameter(torch.zeros(1))
        self.alpha_c_raw = nn.Parameter(torch.zeros(1))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def _build_dt_proj(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _init_a_log(d_state, d_inner, copies=1, merge=True):
        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def _init_d(d_inner, copies=1, merge=True):
        d = torch.ones(d_inner)
        if copies > 1:
            d = d.unsqueeze(0).repeat(copies, 1)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    @staticmethod
    def _to_bhwc(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _to_bchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def _build_four_direction_sequences(self, x):
        batch, channels, height, width = x.shape
        length = height * width
        x_hwwh = torch.stack(
            [
                x.view(batch, -1, length),
                x.transpose(2, 3).contiguous().view(batch, -1, length),
            ],
            dim=1,
        ).view(batch, 2, -1, length)
        return torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

    def _project_params(self, seq, weight):
        params = torch.einsum("b k d l, k c d -> b k c l", seq, weight)
        dt_low_rank, b_param, c_param = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=2)
        return dt_low_rank, b_param, c_param

    def _project_dt(self, dt_low_rank, weight):
        return torch.einsum("b k r l, k d r -> b k d l", dt_low_rank, weight)

    def _merge_dt(self, dts_x, dts_k, bias_x, bias_k):
        if "dt" not in self.condition_on:
            return dts_x, bias_x
        alpha_dt = self._bounded_alpha(self.alpha_dt_raw)
        return dts_x + alpha_dt * dts_k, bias_x + alpha_dt * bias_k

    def _merge_state_param(self, base_param, cond_param, condition_key):
        if condition_key == "B":
            alpha = self._bounded_alpha(self.alpha_b_raw)
        elif condition_key == "C":
            alpha = self._bounded_alpha(self.alpha_c_raw)
        else:
            raise ValueError(f"Unsupported condition key: {condition_key}")

        if condition_key not in self.condition_on:
            return base_param
        return base_param + alpha * cond_param

    def _bounded_alpha(self, raw_alpha):
        return self.alpha_bound * torch.tanh(raw_alpha)

    def _merge_outputs(self, out_y, height, width):
        batch, _, _, length = out_y.shape
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch, 2, -1, length)
        wh_y = out_y[:, 1].view(batch, -1, width, height).transpose(2, 3).contiguous().view(batch, -1, length)
        invwh_y = inv_y[:, 1].view(batch, -1, width, height).transpose(2, 3).contiguous().view(batch, -1, length)
        merged = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        merged = merged.transpose(1, 2).contiguous().view(batch, height, width, -1)
        return merged

    def forward(self, fw_2d, kd_2d):
        if fw_2d.dim() != 4:
            raise ValueError(f"KCSS2D expects fw_2d as [B,C,H,W], got shape={tuple(fw_2d.shape)}")
        if kd_2d.shape != fw_2d.shape:
            raise ValueError(
                f"KCSS2D expects kd_2d shape {tuple(fw_2d.shape)}, got {tuple(kd_2d.shape)}"
            )

        batch, _, height, width = fw_2d.shape

        fw_bhwc = self.fw_norm(self._to_bhwc(fw_2d))
        kd_bhwc = self.kd_norm(self._to_bhwc(kd_2d))

        xz = self.x_in_proj(fw_bhwc)
        x_feat, z = xz.chunk(2, dim=-1)
        x_feat = self.act(self.x_conv2d(self._to_bchw(x_feat)))

        k_feat = self.k_in_proj(kd_bhwc)
        k_feat = self.act(self.k_conv2d(self._to_bchw(k_feat)))

        xs = self._build_four_direction_sequences(x_feat)
        ks = self._build_four_direction_sequences(k_feat)

        dt_x, b_x, c_x = self._project_params(xs, self.x_param_weight)
        dt_k, b_k, c_k = self._project_params(ks, self.k_param_weight)

        dts_x = self._project_dt(dt_x, self.x_dt_proj_weight)
        dts_k = self._project_dt(dt_k, self.k_dt_proj_weight)
        dts, dt_bias = self._merge_dt(dts_x, dts_k, self.x_dt_proj_bias, self.k_dt_proj_bias)

        b_param = self._merge_state_param(b_x.float(), b_k.float(), "B")
        c_param = self._merge_state_param(c_x.float(), c_k.float(), "C")

        seq_x = xs.float().view(batch, -1, height * width)
        dts = dts.contiguous().float().view(batch, -1, height * width)
        dt_bias = dt_bias.float().view(-1)
        a_logs = self.A_logs.float().clamp(min=self.a_log_clamp_min, max=self.a_log_clamp_max)
        a_param = -torch.exp(a_logs).view(-1, self.d_state)
        d_param = self.Ds.float().view(-1)

        out_y = selective_scan_fn(
            seq_x,
            dts,
            a_param,
            b_param,
            c_param,
            d_param,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, 4, -1, height * width)

        y = self._merge_outputs(out_y, height, width)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return self._to_bchw(y)


class KCSRBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_seq,
        d_state=16,
        ssm_expand=2.0,
        local_kernel_size=3,
        bias=False,
        enable_local_refine=True,
        ssm_residual_bound=1.0,
        local_residual_bound=1.0,
    ):
        super().__init__()
        if num_seq < 1:
            raise ValueError(f"KCSRBlock requires num_seq >= 1, got {num_seq}")

        self.kernel_encoder = KernelPromptEncoder(dim=dim, bias=bias)
        self.kcss2d = KCSS2D(dim=dim, d_state=d_state, expand=ssm_expand, bias=bias)
        self.enable_local_refine = enable_local_refine
        self.local_refine = LocalRefine(dim=dim, kernel_size=local_kernel_size, bias=bias) if enable_local_refine else None
        self.ssm_residual_bound = float(ssm_residual_bound)
        self.local_residual_bound = float(local_residual_bound)
        self.ssm_scale_raw = nn.Parameter(torch.zeros(1))
        self.local_scale_raw = nn.Parameter(torch.zeros(1))

    def _bounded_scale(self, raw_scale, bound):
        return bound * torch.tanh(raw_scale)

    def forward(self, fw, kd_3d):
        kd_2d = self.kernel_encoder(kd_3d)
        fw_ssm = self.kcss2d(fw, kd_2d)
        ssm_scale = self._bounded_scale(self.ssm_scale_raw, self.ssm_residual_bound)
        out = fw + ssm_scale * fw_ssm
        if self.local_refine is not None:
            local_scale = self._bounded_scale(self.local_scale_raw, self.local_residual_bound)
            out = out + local_scale * self.local_refine(fw)
        return out
