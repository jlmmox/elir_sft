"""
EncoderSkipFusion: 用 VAE Encoder 跳跃连接 + 三路 DetailGate 替代 WaveletStem + SFT。

核心设计：
- TAESDEncoderStages: 零侵入拆分原版 TAESD encoder Sequential，暴露 256/128/64/32 中间特征
- DetailGateFusion: 拼接 [xin, xenc, fmir_cond] 提取联合条件，预测 α,β 调制 xenc（带噪）
  而非 xin（干净），实现"细节门控"
- EncoderSkipFusion: 冻结 decoder 主干，在 4 个尺度注入 DetailGateFusion，
  仅 scale 256 处加一层 3×3+SiLU 平滑，控制计算量

与 baseline SFT_TAESDFineTuner 的 A/B 对比点：
  SFT_TAESDFineTuner: FMIR cond + Wavelet → TanhSFT_NoTime 调制 decoder 特征
  EncoderSkipFusion:   FMIR cond + Encoder skip → DetailGate 门控 encoder 特征 → 残差注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 步骤一：零侵入地从 TAESD Encoder 提取多尺度特征
# ---------------------------------------------------------------------------

class TAESDEncoderStages(nn.Module):
    """将原版 TAESD encoder 的 Sequential 按分辨率边界拆分为 4 个 stage + 最终投影。

    原版 encoder.layers 结构（15 层）:
      0:  conv(3, 64)            256×256
      1:  Block(64, 64)          256×256, 64ch   ← stage256 输出
      2:  conv stride=2          256→128
      3-5: Block(64,64) ×3       128×128, 64ch   ← stage128 输出
      6:  conv stride=2          128→64
      7-9: Block(64,64) ×3       64×64, 64ch     ← stage64 输出
      10: conv stride=2          64→32
      11-13: Block(64,64) ×3     32×32, 64ch     ← stage32 输出
      14: conv(64, latent_ch)    32×32, latent_ch

    各 stage 是原 layers 的 nn.Module 对象的切片视图——参数共享，权重自动继承。
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        layers = list(encoder.layers)
        if len(layers) != 15:
            raise ValueError(
                f"TAESDEncoderStages expects 15-layer encoder, got {len(layers)} layers. "
                "Please verify the TAESD encoder structure."
            )

        self.stage256 = nn.Sequential(*layers[0:2])   # conv → Block → 256×256, 64ch
        self.stage128 = nn.Sequential(*layers[2:6])   # stride2 conv → 3×Block → 128×128, 64ch
        self.stage64  = nn.Sequential(*layers[6:10])  # stride2 conv → 3×Block → 64×64, 64ch
        self.stage32  = nn.Sequential(*layers[10:14]) # stride2 conv → 3×Block → 32×32, 64ch
        self.to_latent = layers[14]                    # conv(64, latent_ch) → 32×32, latent_ch

    def forward(self, x: torch.Tensor):
        f256 = self.stage256(x)      # [B, 64, H,   W]
        f128 = self.stage128(f256)   # [B, 64, H/2, W/2]
        f64  = self.stage64(f128)    # [B, 64, H/4, W/4]
        f32  = self.stage32(f64)     # [B, 64, H/8, W/8]
        z    = self.to_latent(f32)   # [B, C_latent, H/8, W/8]
        enc_features = {"256": f256, "128": f128, "64": f64, "32": f32}
        return z, enc_features


# ---------------------------------------------------------------------------
# 步骤二：三路联合感知的细节门控模块
# ---------------------------------------------------------------------------

class DetailGateFusion(nn.Module):
    """三路细节门控（单尺度）。

    输入:
      xin       [B, feat_ch, H, W]  decoder 当前特征（语义干净）
      xenc      [B, enc_ch,  H, W]  encoder 跳跃特征（高分辨率，含退化噪声）
      fmir_cond [B, cond_ch, H, W]  FMIR 条件金字塔特征（退化感知）

    流程:
      1. concat([xin, xenc, fmir_cond]) → 2×3×3 Conv + LeakyReLU → cond_feat
      2. cond_feat → 3×3 Conv → α (scale), β (shift)
      3. xenc_mod = xenc ⊙ (1 + α) + β   （门控调制带噪特征，非干净特征）
      4. out = xin + xenc_mod            （残差注入）

    α/β head 零初始化：训练第 0 步 xenc_mod ≈ xenc，网络逐步学会抑制退化通道。
    """

    def __init__(self, feat_channels: int, enc_channels: int, cond_channels: int):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.enc_channels = int(enc_channels)
        self.cond_channels = int(cond_channels)

        total_in = feat_channels + enc_channels + cond_channels
        mid = feat_channels  # 压缩到 feat_ch 以控制计算量

        self.cond_net = nn.Sequential(
            nn.Conv2d(total_in, mid, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.alpha_head = nn.Conv2d(mid, enc_channels, kernel_size=3, padding=1)
        self.beta_head  = nn.Conv2d(mid, enc_channels, kernel_size=3, padding=1)

        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, xin: torch.Tensor, xenc: torch.Tensor, fmir_cond: torch.Tensor) -> torch.Tensor:
        # --- 空间对齐 ---
        if xenc.shape[-2:] != xin.shape[-2:]:
            xenc = F.interpolate(xenc, size=xin.shape[-2:], mode="bilinear", align_corners=False)
        if fmir_cond.shape[-2:] != xin.shape[-2:]:
            fmir_cond = F.interpolate(fmir_cond, size=xin.shape[-2:], mode="bilinear", align_corners=False)

        # --- 运行时 shape 检查 ---
        if xin.shape[1] != self.feat_channels:
            raise ValueError(
                f"[DetailGateFusion] xin channel mismatch: "
                f"expected {self.feat_channels}, got {xin.shape[1]}"
            )
        if xenc.shape[1] != self.enc_channels:
            raise ValueError(
                f"[DetailGateFusion] xenc channel mismatch: "
                f"expected {self.enc_channels}, got {xenc.shape[1]}"
            )
        if fmir_cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"[DetailGateFusion] fmir_cond channel mismatch: "
                f"expected {self.cond_channels}, got {fmir_cond.shape[1]}"
            )
        if xin.device != xenc.device:
            xenc = xenc.to(xin.device)
        if xin.device != fmir_cond.device:
            fmir_cond = fmir_cond.to(xin.device)

        # --- 联合条件提取 ---
        # fmir_cond 始终由 FMIR condition_stem 提供，不存在 None 路径
        joint = torch.cat([xin, xenc, fmir_cond], dim=1)
        cond_feat = self.cond_net(joint)

        # --- 生成仿射参数 ---
        alpha = self.alpha_head(cond_feat)
        beta  = self.beta_head(cond_feat)

        # --- 门控调制（调制 xenc，不是 xin）---
        xenc_mod = xenc * (1.0 + alpha) + beta

        # --- 残差注入 ---
        return xin + xenc_mod


# ---------------------------------------------------------------------------
# 步骤三：Decoder 包装器——在 4 个尺度注入 DetailGateFusion
# ---------------------------------------------------------------------------

class EncoderSkipFusion(nn.Module):
    """Decoder 包装器：冻结 TAESD decoder 主干，注入 encoder skip + DetailGateFusion。

    替换 SFT_TAESDFineTuner。`_use_encoder_skip = True` 标记供 elir.py 路由。

    计算量分布:
      - DetailGateFusion ×4（每尺度 2×3×3 Conv 用于 cond_net + 2×3×3 Conv α/β head）
      - 仅 scale 256 加 smooth_256（3×3 Conv + SiLU），避免每尺度后处理
      - encoder 前向是 frozen no_grad，不计入训练 FLOPs
    """

    def __init__(
        self,
        taesd_encoder: nn.Module,
        taesd_decoder: nn.Module,
        latent_channels: int = 16,
        fmir_cond_channels: dict | None = None,
    ):
        super().__init__()
        if fmir_cond_channels is None:
            fmir_cond_channels = {"256": 16, "128": 32, "64": 64, "32": 64}

        self.latent_channels = int(latent_channels)
        self.taesd_decoder = taesd_decoder

        # 从原版 encoder 构建多尺度特征提取器（参数共享，不修改 taesd.py）
        self.enc_stages = TAESDEncoderStages(taesd_encoder)

        # ---- 冻结 backbone ----
        self.enc_stages.requires_grad_(False)
        self.taesd_decoder.requires_grad_(False)

        # ---- 4 尺度 DetailGateFusion ----
        # encoder 特征全尺度均为 64ch；decoder 特征全尺度均为 64ch
        self.gate_32  = DetailGateFusion(64, 64, fmir_cond_channels["32"])
        self.gate_64  = DetailGateFusion(64, 64, fmir_cond_channels["64"])
        self.gate_128 = DetailGateFusion(64, 64, fmir_cond_channels["128"])
        self.gate_256 = DetailGateFusion(64, 64, fmir_cond_channels["256"])

        # 仅 256 尺度做一次轻量平滑
        self.smooth_256 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # 标记：elir.py 用此属性路由到 encoder skip 路径
        self._use_encoder_skip = True

    # ---- 可训练参数控制 ----

    def set_trainable_only(self):
        """确保 backbone 冻结，仅 gate 模块 + smooth_256 可训练。"""
        self.enc_stages.requires_grad_(False)
        self.taesd_decoder.requires_grad_(False)
        for gate in [self.gate_32, self.gate_64, self.gate_128, self.gate_256]:
            gate.requires_grad_(True)
        self.smooth_256.requires_grad_(True)

    def trainable_parameters(self):
        """仅返回可训练的 gate + smooth 参数（供外部调试/统计）。"""
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

    def _forward_from_layers(
        self,
        z: torch.Tensor,
        enc_features: dict,
        fmir_cond: dict,
    ) -> torch.Tensor:
        """基于 TAESD decoder.layers (Sequential) 的逐 stage 注入。

        decoder.layers 结构:
          [conv_in, ReLU] → [Block×3, Upsample, conv]×3 → [Block×3] → conv_out
          →  32×32          →  32→64    64→128   128→256   →  256×256   → 输出
        """
        layers = self.taesd_decoder.layers
        modules = list(layers.children())

        # conv_in + ReLU → 32×32, 64ch
        x = modules[0](z)
        idx = 1
        if idx < len(modules) and isinstance(modules[idx], nn.ReLU):
            x = modules[idx](x)
            idx += 1

        # 按 Upsample 边界拆分为 4 个 stage
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
                f"EncoderSkipFusion: expected 4 decoder stages (3 Upsample boundaries "
                f"+ final 256 stage), got {len(stage_ops)}. Check TAESD decoder structure."
            )

        cond_map = {
            0: ("cond32", "32"),
            1: ("cond64", "64"),
            2: ("cond128", "128"),
            3: ("cond256", "256"),
        }
        enc_keys  = ["32", "64", "128", "256"]
        gate_list = [self.gate_32, self.gate_64, self.gate_128, self.gate_256]

        # stage 0–2: 每个 stage 以 Upsample 结束 → gate 在 Upsample 前注入
        for stage_idx in range(3):
            for m in stage_ops[stage_idx]:
                if isinstance(m, nn.Upsample):
                    fmir_c = self._fetch_cond(fmir_cond, *cond_map[stage_idx])
                    x = gate_list[stage_idx](x, enc_features[enc_keys[stage_idx]], fmir_c)
                x = m(x)

        # stage 3 (256×256): 无 Upsample → gate → smooth → conv_out
        for m in stage_ops[3]:
            x = m(x)
        fmir_c = self._fetch_cond(fmir_cond, *cond_map[3])
        x = gate_list[3](x, enc_features[enc_keys[3]], fmir_c)
        x = self.smooth_256(x)
        x = modules[-1](x)  # conv_out
        return x

    def _forward_from_up_blocks(
        self,
        z: torch.Tensor,
        enc_features: dict,
        fmir_cond: dict,
    ) -> torch.Tensor:
        """基于 diffusers 风格 up_blocks 的注入路径（兼容未来变体）。"""
        x = self.taesd_decoder.conv_in(z)
        up_blocks = self.taesd_decoder.up_blocks

        gates    = [self.gate_32, self.gate_64, self.gate_128, self.gate_256]
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

            fmir_c = self._fetch_cond(fmir_cond, *cond_map[i])
            x = gates[i](x, enc_features[enc_keys[i]], fmir_c)

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

    def forward(self, z: torch.Tensor, fmir_cond_dict: dict, x_lq: torch.Tensor) -> torch.Tensor:
        """主前向。

        Args:
            z:               [B, latent_ch, H_lat, W_lat]  潜在编码
            fmir_cond_dict:  FMIR 条件金字塔 {"256": ..., "128": ..., "64": ..., "32": ...}
            x_lq:            [B, 3, H, W]  原始 LQ 图像（用于提取 encoder 跳跃特征）

        Returns:
            [B, 3, H, W]  恢复后的图像
        """
        # 冻结前向提取 encoder 多尺度特征
        with torch.no_grad():
            _, enc_features = self.enc_stages(x_lq)

        # 根据 decoder 结构选择注入路径
        if hasattr(self.taesd_decoder, "layers"):
            return self._forward_from_layers(z, enc_features, fmir_cond_dict)

        if hasattr(self.taesd_decoder, "conv_in") and hasattr(self.taesd_decoder, "up_blocks"):
            return self._forward_from_up_blocks(z, enc_features, fmir_cond_dict)

        raise RuntimeError(
            "EncoderSkipFusion: unsupported decoder structure. "
            "Expected 'layers' (Sequential) or 'conv_in'+'up_blocks'."
        )
