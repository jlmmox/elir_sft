import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Single-level Haar DWT implemented with fixed grouped conv kernels."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = int(in_channels)

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32) / 2.0

        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,2,2]
        kernels = kernels.repeat(self.in_channels, 1, 1, 1)  # [4*C,1,2,2]
        self.register_buffer("weight", kernels, persistent=False)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"HaarDWT2D expects [B,C,H,W], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"HaarDWT2D channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )

        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
            b, c, h, w = x.shape

        y = F.conv2d(x, self.weight.to(device=x.device, dtype=x.dtype), stride=2, padding=0, groups=c)
        y = y.view(b, c, 4, h // 2, w // 2)
        ll = y[:, :, 0]
        lh = y[:, :, 1]
        hl = y[:, :, 2]
        hh = y[:, :, 3]
        return ll, lh, hl, hh


class CrossBandAttention(nn.Module):
    """跨频带全局门控：拼接 4 个子带 → 共享上下文 → 4 个独立 Sigmoid 权重。

    不是每个子带各看各的，而是让 LL 知道 HL 在发生什么（比如暴雨）。
    Sigmoid 而非 Softmax，允许各频带独立激活，不互斥。
    """

    def __init__(self, in_channels=3, hidden_ratio=1.0):
        super().__init__()
        in_ch = int(in_channels)
        num_bands = 4
        total_ch = num_bands * in_ch  # 12
        hidden_ch = max(num_bands, int(total_ch * float(hidden_ratio)))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context = nn.Sequential(
            nn.Conv2d(total_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, num_bands, kernel_size=1),
        )
        # bias=2 → sigmoid(2)≈0.88，训练初期接近恒等通过
        nn.init.constant_(self.context[2].bias, 2.0)

    def forward(self, ll, lh, hl, hh):
        b = ll.shape[0]
        pooled = [self.pool(band) for band in (ll, lh, hl, hh)]
        joint = torch.cat(pooled, dim=1)  # [B, 12, 1, 1]
        gates = torch.sigmoid(self.context(joint))  # [B, 4, 1, 1]
        # 拆分到 4 个频带
        w_ll = gates[:, 0:1, :, :]
        w_lh = gates[:, 1:2, :, :]
        w_hl = gates[:, 2:3, :, :]
        w_hh = gates[:, 3:4, :, :]
        return ll * w_ll, lh * w_lh, hl * w_hl, hh * w_hh


class WaveletStem(nn.Module):
    """Lightweight high-frequency condition extractor for decoder SFT."""

    def __init__(self, in_channels=3, base_channels=16, use_ll=True,
                 use_band_attn=False):
        super().__init__()
        in_channels = int(in_channels)
        base_channels = int(base_channels)
        self.use_ll = bool(use_ll)
        self.use_band_attn = bool(use_band_attn)

        self.dwt = HaarDWT2D(in_channels=in_channels)
        if self.use_band_attn:
            self.band_attn = CrossBandAttention(in_channels=in_channels)
        band_channels = (4 if self.use_ll else 3) * in_channels
        self.cond_channels = {
            "256": 16,
            "128": 32,
            "64": 64,
            "32": 64,
        }

        self.proj = nn.Sequential(
            nn.Conv2d(band_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.to_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.to_128 = nn.Conv2d(base_channels, 32, kernel_size=3, stride=1, padding=1)
        self.to_64 = nn.Sequential(
            nn.Conv2d(base_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.to_32 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x_lq, return_debug=False):
        if x_lq is None:
            if return_debug:
                return None, None
            return None

        ll, lh, hl, hh = self.dwt(x_lq)
        if self.use_band_attn:
            ll, lh, hl, hh = self.band_attn(ll, lh, hl, hh)
        # 可切换是否包含低频 LL；use_ll=False 时仅用高频 LH/HL/HH。
        if self.use_ll:
            full_band = torch.cat([ll, lh, hl, hh], dim=1)
        else:
            full_band = torch.cat([lh, hl, hh], dim=1)

        cond128_base = self.proj(full_band)
        cond256 = self.to_256(cond128_base)
        cond128 = self.to_128(cond128_base)
        cond64 = self.to_64(cond128_base)
        cond32 = self.to_32(cond64)

        cond_dict = {
            "256": cond256,
            "128": cond128,
            "64": cond64,
            "32": cond32,
            "cond256": cond256,
            "cond128": cond128,
            "cond64": cond64,
            "cond32": cond32,
        }

        if not return_debug:
            return cond_dict

        debug = {
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "cond128_base": cond128_base,
            "cond256": cond256,
            "cond128": cond128,
            "cond64": cond64,
            "cond32": cond32,
        }
        return cond_dict, debug
