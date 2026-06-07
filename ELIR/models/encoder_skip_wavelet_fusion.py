"""
EncoderSkipWaveletFusion: 三流互补 decoder 条件注入。

三流信息源：
  - FMIR cond (退化感知，语义级)
  - Encoder skip (TAESD encoder 中间特征，结构保持)
  - Wavelet cond (显式频带分解，高频纹理 + 低频光照)

与 EncoderSkipFusion 的对比：
  EncoderSkipFusion:      xin + xenc + fmir_cond → DetailGate (3-way)
  EncoderSkipWaveletFusion: xin + xenc + fmir_cond + wavelet_cond → DetailGate (4-way)

消融 B 由 elir.py 中的 ablation_skip_fmir 控制（fm_cfg.ablation_skip_fmir: true）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 四路细节门控（单尺度）
# ---------------------------------------------------------------------------

class DetailGateFusion4Way(nn.Module):
    """四路细节门控：xin + xenc + fmir_cond + wavelet_cond → 残差注入。

    输入:
      xin           [B, feat_ch, H, W]  decoder 当前特征
      xenc          [B, enc_ch,  H, W]  encoder 跳跃特征（含退化噪声）
      fmir_cond     [B, fmir_ch, H, W]  FMIR 条件（退化感知）
      wavelet_cond  [B, wav_ch,  H, W]  小波条件（显式频带分解）

    流程:
      1. concat([xin, xenc, fmir_cond, wavelet_cond]) → 2×3×3 Conv + LeakyReLU
      2. → α (scale), β (shift)
      3. xenc_mod = xenc ⊙ (1 + α) + β
      4. out = xin + xenc_mod （残差注入）
    """

    def __init__(self, feat_channels: int, enc_channels: int,
                 fmir_cond_channels: int, wavelet_cond_channels: int):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.enc_channels = int(enc_channels)
        self.fmir_cond_channels = int(fmir_cond_channels)
        self.wavelet_cond_channels = int(wavelet_cond_channels)

        total_in = feat_channels + enc_channels + fmir_cond_channels + wavelet_cond_channels
        mid = feat_channels

        self.cond_net = nn.Sequential(
            nn.Conv2d(total_in, mid, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.alpha_head = nn.Conv2d(mid, enc_channels, kernel_size=3, padding=1)
        self.beta_head = nn.Conv2d(mid, enc_channels, kernel_size=3, padding=1)

        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, xin: torch.Tensor, xenc: torch.Tensor,
                fmir_cond: torch.Tensor, wavelet_cond: torch.Tensor) -> torch.Tensor:
        # 空间对齐
        if xenc.shape[-2:] != xin.shape[-2:]:
            xenc = F.interpolate(xenc, size=xin.shape[-2:], mode="bilinear", align_corners=False)
        if fmir_cond.shape[-2:] != xin.shape[-2:]:
            fmir_cond = F.interpolate(fmir_cond, size=xin.shape[-2:], mode="bilinear", align_corners=False)
        if wavelet_cond.shape[-2:] != xin.shape[-2:]:
            wavelet_cond = F.interpolate(wavelet_cond, size=xin.shape[-2:], mode="bilinear", align_corners=False)

        # 设备对齐
        if xin.device != xenc.device:
            xenc = xenc.to(xin.device)
        if xin.device != fmir_cond.device:
            fmir_cond = fmir_cond.to(xin.device)
        if xin.device != wavelet_cond.device:
            wavelet_cond = wavelet_cond.to(xin.device)

        joint = torch.cat([xin, xenc, fmir_cond, wavelet_cond], dim=1)
        cond_feat = self.cond_net(joint)

        alpha = self.alpha_head(cond_feat)
        beta = self.beta_head(cond_feat)

        xenc_mod = xenc * (1.0 + alpha) + beta
        return xin + xenc_mod


# ---------------------------------------------------------------------------
# Decoder 包装器：冻结 TAESD decoder 主干，注入 encoder skip + wavelet + fmir
# ---------------------------------------------------------------------------

class EncoderSkipWaveletFusion(nn.Module):
    """三流互补 decoder 条件注入。

    冻结 TAESD decoder 主干，在 4 个尺度注入 DetailGateFusion4Way。
    三流信息：FMIR 退化条件 + Encoder 跳跃特征 + Wavelet 频带条件。

    标记 `_use_encoder_skip_wavelet = True` 供 elir.py 路由。
    """

    def __init__(
        self,
        taesd_encoder: nn.Module,
        taesd_decoder: nn.Module,
        latent_channels: int = 16,
        fmir_cond_channels: dict | None = None,
        wavelet_cond_channels: dict | None = None,
    ):
        super().__init__()
        if fmir_cond_channels is None:
            fmir_cond_channels = {"256": 16, "128": 32, "64": 64, "32": 64}
        if wavelet_cond_channels is None:
            wavelet_cond_channels = {"256": 16, "128": 32, "64": 64, "32": 64}

        self.latent_channels = int(latent_channels)
        self.taesd_decoder = taesd_decoder

        from ELIR.models.encoder_skip_fusion import TAESDEncoderStages
        self.enc_stages = TAESDEncoderStages(taesd_encoder)

        # 冻结 backbone
        self.enc_stages.requires_grad_(False)
        self.taesd_decoder.requires_grad_(False)

        # 4 尺度 DetailGateFusion4Way（四路门控）
        self.gate_32 = DetailGateFusion4Way(
            64, 64, fmir_cond_channels["32"], wavelet_cond_channels["32"])
        self.gate_64 = DetailGateFusion4Way(
            64, 64, fmir_cond_channels["64"], wavelet_cond_channels["64"])
        self.gate_128 = DetailGateFusion4Way(
            64, 64, fmir_cond_channels["128"], wavelet_cond_channels["128"])
        self.gate_256 = DetailGateFusion4Way(
            64, 64, fmir_cond_channels["256"], wavelet_cond_channels["256"])

        self.smooth_256 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self._use_encoder_skip_wavelet = True

    # ---- 可训练参数控制 ----

    def set_trainable_only(self):
        self.enc_stages.requires_grad_(False)
        self.taesd_decoder.requires_grad_(False)
        for gate in [self.gate_32, self.gate_64, self.gate_128, self.gate_256]:
            gate.requires_grad_(True)
        self.smooth_256.requires_grad_(True)

    def trainable_parameters(self):
        yield from self.gate_32.parameters()
        yield from self.gate_64.parameters()
        yield from self.gate_128.parameters()
        yield from self.gate_256.parameters()
        yield from self.smooth_256.parameters()

    # ---- 条件字典辅助 ----

    @staticmethod
    def _fetch_cond(cond_dict: dict, name_a: str, name_b: str | None = None) -> torch.Tensor:
        if not isinstance(cond_dict, dict):
            raise TypeError(f"cond_dict must be a dict, got {type(cond_dict)}")
        if name_a in cond_dict:
            return cond_dict[name_a]
        if name_b is not None and name_b in cond_dict:
            return cond_dict[name_b]
        raise KeyError(
            f"Missing condition key '{name_a}' (alias='{name_b}'). "
            f"Available keys: {list(cond_dict.keys())}"
        )

    # ---- 前向传播 ----

    def _forward_from_layers(self, z, enc_features, fmir_cond, wavelet_cond):
        layers = self.taesd_decoder.layers
        modules = list(layers.children())

        x = modules[0](z)
        idx = 1
        if idx < len(modules) and isinstance(modules[idx], nn.ReLU):
            x = modules[idx](x)
            idx += 1

        stage_ops: list[list] = []
        curr: list = []
        for m in modules[idx:-1]:
            curr.append(m)
            if isinstance(m, nn.Upsample):
                stage_ops.append(curr)
                curr = []
        if curr:
            stage_ops.append(curr)

        if len(stage_ops) != 4:
            raise RuntimeError(
                f"EncoderSkipWaveletFusion: expected 4 decoder stages, got {len(stage_ops)}"
            )

        cond_map = [
            ("cond32", "32"),
            ("cond64", "64"),
            ("cond128", "128"),
            ("cond256", "256"),
        ]
        enc_keys = ["32", "64", "128", "256"]
        gate_list = [self.gate_32, self.gate_64, self.gate_128, self.gate_256]

        for stage_idx in range(3):
            for m in stage_ops[stage_idx]:
                if isinstance(m, nn.Upsample):
                    fc = self._fetch_cond(fmir_cond, *cond_map[stage_idx])
                    wc = self._fetch_cond(wavelet_cond, *cond_map[stage_idx])
                    x = gate_list[stage_idx](x, enc_features[enc_keys[stage_idx]], fc, wc)
                x = m(x)

        for m in stage_ops[3]:
            x = m(x)
        fc = self._fetch_cond(fmir_cond, *cond_map[3])
        wc = self._fetch_cond(wavelet_cond, *cond_map[3])
        x = gate_list[3](x, enc_features[enc_keys[3]], fc, wc)
        x = self.smooth_256(x)
        x = modules[-1](x)
        return x

    def _forward_from_up_blocks(self, z, enc_features, fmir_cond, wavelet_cond):
        x = self.taesd_decoder.conv_in(z)
        up_blocks = self.taesd_decoder.up_blocks

        gates = [self.gate_32, self.gate_64, self.gate_128, self.gate_256]
        enc_keys = ["32", "64", "128", "256"]
        cond_map = [
            ("cond32", "32"),
            ("cond64", "64"),
            ("cond128", "128"),
            ("cond256", "256"),
        ]

        for i in range(4):
            blk = up_blocks[i]
            resnets = getattr(blk, "resnets", None)
            upsamplers = getattr(blk, "upsamplers", None)

            if resnets is not None:
                for res in resnets:
                    x = res(x)

            fc = self._fetch_cond(fmir_cond, *cond_map[i])
            wc = self._fetch_cond(wavelet_cond, *cond_map[i])
            x = gates[i](x, enc_features[enc_keys[i]], fc, wc)

            if i < 3:
                if upsamplers is not None and len(upsamplers) > 0:
                    for up in upsamplers:
                        x = up(x)
            else:
                if upsamplers is not None and len(upsamplers) > 0:
                    for up in upsamplers:
                        x = up(x)
                x = self.smooth_256(x)

        x = self.taesd_decoder.conv_out(x)
        return x

    def forward(self, z: torch.Tensor, fmir_cond_dict: dict,
                wavelet_cond_dict: dict, x_lq: torch.Tensor) -> torch.Tensor:
        """主前向。

        Args:
            z:                 [B, latent_ch, H_lat, W_lat]
            fmir_cond_dict:    FMIR 条件金字塔 {"256":..., "128":..., "64":..., "32":...}
            wavelet_cond_dict: 小波条件金字塔 {"256":..., "128":..., "64":..., "32":...}
            x_lq:              [B, 3, H, W] 原始 LQ 图像（提取 encoder 跳跃特征）
        Returns:
            [B, 3, H, W]
        """
        if wavelet_cond_dict is None:
            raise ValueError(
                "EncoderSkipWaveletFusion requires a non-None wavelet_cond_dict. "
                "Ensure wavelet_cfg.enabled=true and wavelet_cfg.fuse_with_spatial=false in the config."
            )

        with torch.no_grad():
            _, enc_features = self.enc_stages(x_lq)

        if hasattr(self.taesd_decoder, "layers"):
            return self._forward_from_layers(z, enc_features, fmir_cond_dict, wavelet_cond_dict)

        if hasattr(self.taesd_decoder, "conv_in") and hasattr(self.taesd_decoder, "up_blocks"):
            return self._forward_from_up_blocks(z, enc_features, fmir_cond_dict, wavelet_cond_dict)

        raise RuntimeError(
            "EncoderSkipWaveletFusion: unsupported decoder structure."
        )
