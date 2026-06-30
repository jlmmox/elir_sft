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


class TimeBandWeight(nn.Module):
    """t_emb → MLP → 4×Sigmoid → 时间驱动的频带权重。
    流匹配 t→0 时可偏好低频(全局)，t→1 时偏好高频(细节)。"""

    def __init__(self, time_dim=160):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 4),
            nn.Sigmoid(),
        )
        # bias 偏置: 初始各频带均匀通过 (~0.5)
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.constant_(self.mlp[1].bias, 0.0)   # sigmoid(0)=0.5

    def forward(self, ll, lh, hl, hh, t_emb):
        w = self.mlp(t_emb)                # [B, 4]
        return (ll * w[:, 0:1, None, None],
                lh * w[:, 1:2, None, None],
                hl * w[:, 2:3, None, None],
                hh * w[:, 3:4, None, None])


class WaveletStem(nn.Module):
    """Lightweight high-frequency condition extractor for decoder/UNet SFT."""

    def __init__(self, in_channels=3, base_channels=16, use_ll=True,
                 use_band_attn=False, time_cond=False, time_dim=160):
        super().__init__()
        in_channels = int(in_channels)
        base_channels = int(base_channels)
        self.use_ll = bool(use_ll)
        self.use_band_attn = bool(use_band_attn)
        self.time_cond = bool(time_cond)

        self.dwt = HaarDWT2D(in_channels=in_channels)
        if self.use_band_attn:
            self.band_attn = CrossBandAttention(in_channels=in_channels)
        if self.time_cond:
            self.time_weight = TimeBandWeight(time_dim=time_dim)
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

    def forward(self, x_lq, return_debug=False, t_emb=None):
        if x_lq is None:
            if return_debug:
                return None, None
            return None

        ll, lh, hl, hh = self.dwt(x_lq)
        if self.use_band_attn:
            ll, lh, hl, hh = self.band_attn(ll, lh, hl, hh)
        if self.time_cond and t_emb is not None:
            ll, lh, hl, hh = self.time_weight(ll, lh, hl, hh, t_emb)
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


# ---------------------------------------------------------------------------
# LFG-SFG: Low-Frequency-Guided Spatial Frequency Gate
# ---------------------------------------------------------------------------

class SpatialFreqGate(nn.Module):
    """空间频率门控：以 LL 为基底，选择性激活 HF 中的可信空间位置。

    gate = sigmoid(Conv([ll_s, hf_s]))          [B,1,H,W] 空间图
    cond = ll_proj(ll_s) + gate * hf_proj(hf_s)  逐像素加权融合

    gate 零初始化 (weight=0, bias=-4) → gate≈0.018，初始接近纯 LL。
    """

    def __init__(self, ll_ch, hf_ch, out_ch, gate_bias_init=-4.0, hf_init="small"):
        super().__init__()
        self.ll_proj = nn.Sequential(
            nn.Conv2d(ll_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.hf_proj = nn.Sequential(
            nn.Conv2d(hf_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        # gate: 单通道空间注意力
        self.gate_conv = nn.Sequential(
            nn.Conv2d(ll_ch + hf_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, 1, kernel_size=3, stride=1, padding=1),
        )
        # 初始化
        nn.init.zeros_(self.gate_conv[2].weight)
        nn.init.constant_(self.gate_conv[2].bias, float(gate_bias_init))
        if hf_init == "small":
            for m in self.hf_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    with torch.no_grad():
                        m.weight *= 0.1
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, ll_s, hf_s, spatial_cond=None):
        gate = torch.sigmoid(self.gate_conv(torch.cat([ll_s, hf_s], dim=1)))  # [B,1,H,W]
        out = self.ll_proj(ll_s) + gate * self.hf_proj(hf_s)
        return out, gate


class LFGSFGWaveletStem(nn.Module):
    """Low-Frequency-Guided Spatial Frequency Gate WaveletStem。

    和旧 WaveletStem 接口完全兼容，输出 dict 包含 256/128/64/32/cond*。
    return_debug 时额外返回 gate maps。
    """

    def __init__(self, in_channels=3, base_channels=16, use_ll=True,
                 use_band_attn=False, time_cond=False, time_dim=160,
                 gate_bias_init=-4.0, hf_init="small"):
        super().__init__()
        if not use_ll:
            print("[LFG-SFG] WARNING: use_ll=false is invalid for LFG-SFG, forcing use_ll=true")
            use_ll = True

        in_ch = int(in_channels)
        base_ch = int(base_channels)
        self.use_band_attn = bool(use_band_attn)
        self.time_cond = bool(time_cond)

        self.dwt = HaarDWT2D(in_channels=in_ch)
        if self.use_band_attn:
            self.band_attn = CrossBandAttention(in_channels=in_ch)
        if self.time_cond:
            self.time_weight = TimeBandWeight(time_dim=time_dim)

        self.cond_channels = {"256": 16, "128": 32, "64": 64, "32": 64}

        # LL branch (3ch)
        self.ll_proj_in = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(),
        )
        # HF branch (LH+HL+HH = 9ch)
        self.hf_proj_in = nn.Sequential(
            nn.Conv2d(in_ch * 3, base_ch, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1), nn.SiLU(),
        )

        # 多尺度投影
        self.ll_to_256 = self._make_head(base_ch, 16, upsample=True)
        self.ll_to_128 = nn.Conv2d(base_ch, 32, 3, 1, 1)
        self.ll_to_64 = self._make_head(base_ch, 64, downsample=True)
        self.ll_to_32 = self._make_head(64, 64, downsample=True)

        self.hf_to_256 = self._make_head(base_ch, 16, upsample=True)
        self.hf_to_128 = nn.Conv2d(base_ch, 32, 3, 1, 1)
        self.hf_to_64 = self._make_head(base_ch, 64, downsample=True)
        self.hf_to_32 = self._make_head(64, 64, downsample=True)

        # 每个尺度一个 SpatialFreqGate
        self.gate_256 = SpatialFreqGate(16, 16, 16, gate_bias_init, hf_init)
        self.gate_128 = SpatialFreqGate(32, 32, 32, gate_bias_init, hf_init)
        self.gate_64  = SpatialFreqGate(64, 64, 64, gate_bias_init, hf_init)
        self.gate_32  = SpatialFreqGate(64, 64, 64, gate_bias_init, hf_init)

    @staticmethod
    def _make_head(in_ch, out_ch, upsample=False, downsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(nn.Conv2d(in_ch, out_ch, 3, 1 if not downsample else 2, 1))
        layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def forward(self, x_lq, return_debug=False, t_emb=None):
        if x_lq is None:
            return (None, None) if return_debug else None

        ll, lh, hl, hh = self.dwt(x_lq)
        if self.use_band_attn:
            ll, lh, hl, hh = self.band_attn(ll, lh, hl, hh)
        if self.time_cond and t_emb is not None:
            ll, lh, hl, hh = self.time_weight(ll, lh, hl, hh, t_emb)

        hf = torch.cat([lh, hl, hh], dim=1)  # [B, 9, H_half, W_half]

        ll_feat = self.ll_proj_in(ll)   # [B, base_ch, 128, 128]
        hf_feat = self.hf_proj_in(hf)   # [B, base_ch, 128, 128]

        ll256 = self.ll_to_256(ll_feat)
        hf256 = self.hf_to_256(hf_feat)
        cond256, gate256 = self.gate_256(ll256, hf256)

        ll128 = self.ll_to_128(ll_feat)
        hf128 = self.hf_to_128(hf_feat)
        cond128, gate128 = self.gate_128(ll128, hf128)

        ll64 = self.ll_to_64(ll_feat)
        hf64 = self.hf_to_64(hf_feat)
        cond64, gate64 = self.gate_64(ll64, hf64)

        ll32 = self.ll_to_32(ll64)
        hf32 = self.hf_to_32(hf64)
        cond32, gate32 = self.gate_32(ll32, hf32)

        cond_dict = {
            "256": cond256, "128": cond128, "64": cond64, "32": cond32,
            "cond256": cond256, "cond128": cond128, "cond64": cond64, "cond32": cond32,
        }

        if not return_debug:
            return cond_dict

        debug = {
            "ll": ll, "lh": lh, "hl": hl, "hh": hh,
            "ll_feat": ll_feat, "hf_feat": hf_feat,
            "gate256": gate256, "gate128": gate128, "gate64": gate64, "gate32": gate32,
            "cond256": cond256, "cond128": cond128, "cond64": cond64, "cond32": cond32,
        }
        return cond_dict, debug
