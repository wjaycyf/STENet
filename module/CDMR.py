import torch
import torch.nn as nn
import torch.nn.functional as F


class CDMR(nn.Module):
    def __init__(
        self,
        dim,
        num_seq,
        memory_momentum=0.9,
        motion_threshold=1.0,
        confidence_threshold=0.5,
        bias=False,
    ):
        super(CDMR, self).__init__()
        if num_seq < 3:
            raise ValueError(f'CDMR requires num_seq >= 3, got {num_seq}')
        if not (0.0 <= memory_momentum <= 1.0):
            raise ValueError(f'memory_momentum must be in [0,1], got {memory_momentum}')

        self.dim = dim
        self.num_seq = num_seq
        self.memory_momentum = memory_momentum
        self.motion_threshold = motion_threshold
        self.confidence_threshold = confidence_threshold

        self.delta_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.learned_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def _validate_input(self, feat):
        if feat.dim() != 5:
            raise ValueError(f'CDMR expects 5D [B,C,T,H,W], got shape={tuple(feat.shape)}')
        _, c, t, _, _ = feat.shape
        if c != self.dim:
            raise ValueError(f'CDMR dim mismatch: expected {self.dim}, got {c}')
        if t != self.num_seq:
            raise ValueError(f'CDMR num_seq mismatch: expected {self.num_seq}, got {t}')

    def _compute_motion_gate(self, delta_prev, delta_next):
        motion_score = (delta_prev.abs() + delta_next.abs()).mean(dim=1).squeeze(1)  # [B,H,W]
        motion_ref = motion_score.mean(dim=(1, 2), keepdim=True) + 1e-6
        motion_norm = motion_score / motion_ref
        motion_gate = torch.sigmoid((self.motion_threshold - motion_norm).unsqueeze(1))  # [B,1,H,W]
        return motion_gate, motion_score

    def _compute_confidence_gate(self, center_2d, memory_2d):
        confidence = F.cosine_similarity(center_2d, memory_2d, dim=1, eps=1e-6).unsqueeze(1)  # [-1,1] [B, 1, H, W]
        confidence = (confidence + 1.0) * 0.5  # [0,1]
        confidence_gate = torch.sigmoid((confidence - self.confidence_threshold) * 8.0)
        return confidence_gate, confidence

    def forward(self, feat, prev_memory=None, update_memory=True):
        self._validate_input(feat)

        b, _, t, h, w = feat.shape
        mid = t // 2

        center = feat[:, :, mid:mid + 1, :, :]   # [B, C, 1, H, W]
        prev_frame = feat[:, :, mid - 1:mid, :, :]
        next_frame = feat[:, :, mid + 1:mid + 2, :, :]

        delta_prev = center - prev_frame
        delta_next = next_frame - center
        delta_feat = self.delta_proj(torch.cat([delta_prev.squeeze(2), delta_next.squeeze(2)], dim=1))  # [B, C, H, W]

        if prev_memory is None:
            memory = center
            gate = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
            motion_score = torch.zeros((b, h, w), dtype=feat.dtype, device=feat.device)
            confidence = torch.zeros((b, 1, h, w), dtype=feat.dtype, device=feat.device)
        else:
            if prev_memory.shape != center.shape:
                raise ValueError(
                    f'prev_memory shape mismatch: expected {tuple(center.shape)}, got {tuple(prev_memory.shape)}'
                )

            memory = prev_memory
            center_2d = center.squeeze(2)  # [B, C, H, W]
            memory_2d = memory.squeeze(2)
            learned_gate = torch.sigmoid(self.learned_gate(torch.cat([center_2d, memory_2d], dim=1)))
            motion_gate, motion_score = self._compute_motion_gate(delta_prev, delta_next)
            confidence_gate, confidence = self._compute_confidence_gate(center_2d, memory_2d)
            gate = learned_gate * motion_gate * confidence_gate

        fused_center = gate * memory.squeeze(2) + (1.0 - gate) * (center.squeeze(2) + delta_feat)
        out = feat.clone()
        out[:, :, mid, :, :] = fused_center

        next_memory = None
        if update_memory:
            next_memory = (
                self.memory_momentum * memory + (1.0 - self.memory_momentum) * fused_center.unsqueeze(2)
            ).detach()

        stats = {
            'gate_mean': gate.mean().detach(),
            'motion_mean': motion_score.mean().detach(),
            'confidence_mean': confidence.mean().detach(),
        }
        return out, next_memory, stats
