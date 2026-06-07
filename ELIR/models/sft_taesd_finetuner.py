import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhSFT_NoTime(nn.Module):
    """
    无时间门控版本的 Tanh-SFT。

    设计约束：
    1) 默认执行最后一跳空间对齐，保证 feat 与 cond 1:1；
    2) gamma_head / beta_head 均为 3x3 Conv2d，且强制零初始化；
    3) 仅执行仿射调制：feat * (1 + gamma) + beta。
    """

    def __init__(self, feat_channels: int, cond_channels: int, gamma_scale: float = 0.2, beta_scale: float = 0.2):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.cond_channels = int(cond_channels)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)

        self.shared = nn.Sequential(
            nn.Conv2d(self.cond_channels, self.feat_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.gamma_head = nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
        self.beta_head = nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)

        # 强制零初始化，保证初始行为接近恒等映射，避免刚开始训练就扰动解码主干。
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond is None:
            raise ValueError("TanhSFT_NoTime requires a non-None cond tensor.")

        if feat.ndim != 4 or cond.ndim != 4:
            raise ValueError(f"Expect 4D tensors [B,C,H,W], got feat={feat.shape}, cond={cond.shape}")

        if feat.shape[-2:] != cond.shape[-2:]:
            cond = F.interpolate(cond, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        if feat.shape[1] != self.feat_channels:
            raise ValueError(
                f"Feature channel mismatch: expected {self.feat_channels}, got {feat.shape[1]}"
            )

        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"Condition channel mismatch: expected {self.cond_channels}, got {cond.shape[1]}"
            )

        h = self.shared(cond)
        gamma = torch.tanh(self.gamma_head(h)) * self.gamma_scale
        beta = torch.tanh(self.beta_head(h)) * self.beta_scale

        gamma = gamma.to(device=feat.device, dtype=feat.dtype)
        beta = beta.to(device=feat.device, dtype=feat.dtype)
        return feat * (1.0 + gamma) + beta


class SFT_TAESDFineTuner(nn.Module):
    """
    冻结 TAESD decoder 主干，只训练 4 个原生分辨率的 SFT 模块。

    目标解码路径（固定为 256 输出）:
    - z: [B, 16, 32, 32]
    - x32  -> sft_32(cond32[64ch])   -> up to x64
    - x64  -> sft_64(cond64[64ch])   -> up to x128
    - x128 -> sft_128(cond128[32ch]) -> up to x256
    - x256 -> sft_256(cond256[16ch]) -> conv_out -> [B,3,256,256]
    """

    def __init__(
        self,
        taesd_decoder: nn.Module,
        gamma_scale: float = 0.2,
        beta_scale: float = 0.2,
        latent_channels: int = 16,
    ):
        super().__init__()
        if taesd_decoder is None:
            raise ValueError("taesd_decoder must be a valid decoder module.")

        self.taesd_decoder = taesd_decoder
        self.latent_channels = int(latent_channels)

        inferred_in_ch = self._infer_decoder_in_channels(self.taesd_decoder)
        if inferred_in_ch is not None and inferred_in_ch != self.latent_channels:
            raise ValueError(
                "SFT_TAESDFineTuner latent channel mismatch: "
                f"requested latent_channels={self.latent_channels}, decoder expects {inferred_in_ch}."
            )

        # 立即冻结 TAESD decoder 全部参数，确保训练时仅更新 SFT。
        self.taesd_decoder.requires_grad_(False)

        self.sft_32 = TanhSFT_NoTime(64, 64, gamma_scale=gamma_scale, beta_scale=beta_scale)
        self.sft_64 = TanhSFT_NoTime(64, 64, gamma_scale=gamma_scale, beta_scale=beta_scale)
        self.sft_128 = TanhSFT_NoTime(64, 32, gamma_scale=gamma_scale, beta_scale=beta_scale)
        self.sft_256 = TanhSFT_NoTime(64, 16, gamma_scale=gamma_scale, beta_scale=beta_scale)

        # 显式声明 SFT 可训练，避免外部冻结逻辑误伤。
        for p in self.sft_32.parameters():
            p.requires_grad = True
        for p in self.sft_64.parameters():
            p.requires_grad = True
        for p in self.sft_128.parameters():
            p.requires_grad = True
        for p in self.sft_256.parameters():
            p.requires_grad = True

        # 标记给外部模型工厂：该模型应始终维持“冻结 decoder，仅训练 SFT”。
        self._force_sft_trainable_only = True

    def sft_parameters(self):
        """返回仅 SFT 子模块参数，用于构建优化器。"""
        yield from self.sft_32.parameters()
        yield from self.sft_64.parameters()
        yield from self.sft_128.parameters()
        yield from self.sft_256.parameters()

    def set_sft_trainable_only(self):
        """
        重新施加训练开关：decoder 全冻结，SFT 全可训练。
        在外部可能存在统一 requires_grad 控制时，调用该函数可恢复正确状态。
        """
        self.taesd_decoder.requires_grad_(False)
        self.sft_32.requires_grad_(True)
        self.sft_64.requires_grad_(True)
        self.sft_128.requires_grad_(True)
        self.sft_256.requires_grad_(True)

    @staticmethod
    def _fetch_cond(cond_dict: dict, name_a: str, name_b: str = None) -> torch.Tensor:
        if not isinstance(cond_dict, dict):
            raise ValueError("cond_dict must be a dict containing multi-scale conditions.")

        if name_a in cond_dict:
            return cond_dict[name_a]
        if name_b is not None and name_b in cond_dict:
            return cond_dict[name_b]

        keys = list(cond_dict.keys())
        raise KeyError(f"Missing condition '{name_a}' (alias='{name_b}'). Available keys: {keys}")

    @staticmethod
    def _run_module_list(module_list: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for m in module_list:
            x = m(x)
        return x

    @staticmethod
    def _infer_decoder_in_channels(decoder: nn.Module):
        conv_in = getattr(decoder, "conv_in", None)
        if isinstance(conv_in, nn.Conv2d):
            return int(conv_in.in_channels)

        layers = getattr(decoder, "layers", None)
        if isinstance(layers, nn.Sequential) and len(layers) > 0 and isinstance(layers[0], nn.Conv2d):
            return int(layers[0].in_channels)
        return None

    def _forward_from_layers(self, z: torch.Tensor, cond32: torch.Tensor, cond64: torch.Tensor,
                             cond128: torch.Tensor, cond256: torch.Tensor) -> torch.Tensor:
        """
        兼容类似 TAESD 原生实现的 decoder.layers 顺序堆叠结构：
        conv_in(+act) -> [stage1 + up] -> [stage2 + up] -> [stage3 + up] -> [stage4] -> conv_out
        """
        layers = getattr(self.taesd_decoder, "layers")
        if not isinstance(layers, nn.Sequential):
            raise TypeError("decoder.layers must be nn.Sequential for _forward_from_layers path.")

        modules = list(layers.children())
        if len(modules) < 6:
            raise RuntimeError("decoder.layers is too short to map into 4 SFT stages.")

        # 典型 TAESD: [conv_in, act, ..., conv_out]
        x = modules[0](z)
        idx = 1
        if idx < len(modules) and isinstance(modules[idx], (nn.ReLU, nn.SiLU, nn.GELU, nn.LeakyReLU)):
            x = modules[idx](x)
            idx += 1

        upsample_count = 0
        stage_ops = []
        curr_stage = []
        for m in modules[idx:-1]:
            curr_stage.append(m)
            if isinstance(m, nn.Upsample):
                stage_ops.append(curr_stage)
                curr_stage = []
                upsample_count += 1
        if curr_stage:
            stage_ops.append(curr_stage)

        # 需要 4 个逻辑 stage：前三个以 Upsample 结束，第四个不再上采样。
        if upsample_count != 3 or len(stage_ops) != 4:
            raise RuntimeError(
                "Cannot map decoder.layers to required 4 stages (3 upsample boundaries + final 256 stage)."
            )

        # stage1: 32 分辨率残差堆叠 -> SFT32 -> upsample 到 64
        for m in stage_ops[0]:
            if isinstance(m, nn.Upsample):
                x = self.sft_32(x, cond32)
            x = m(x)

        # stage2: 64 分辨率残差堆叠 -> SFT64 -> upsample 到 128
        for m in stage_ops[1]:
            if isinstance(m, nn.Upsample):
                x = self.sft_64(x, cond64)
            x = m(x)

        # stage3: 128 分辨率残差堆叠 -> SFT128 -> upsample 到 256
        for m in stage_ops[2]:
            if isinstance(m, nn.Upsample):
                x = self.sft_128(x, cond128)
            x = m(x)

        # stage4: 256 分辨率残差堆叠 -> SFT256
        for m in stage_ops[3]:
            x = m(x)
        x = self.sft_256(x, cond256)

        conv_out = modules[-1]
        x = conv_out(x)
        return x

    def _forward_from_up_blocks(self, z: torch.Tensor, cond32: torch.Tensor, cond64: torch.Tensor,
                                cond128: torch.Tensor, cond256: torch.Tensor) -> torch.Tensor:
        """
        兼容 diffusers 常见的 tiny decoder 结构：
        conv_in -> up_blocks -> conv_out。

        关键点：每个 up_block 内部分解为 resnets 与 upsamplers，
        在 upsample 前注入对应尺度 SFT，满足严格的 32/64/128/256 对齐。
        """
        if not hasattr(self.taesd_decoder, "conv_in") or not hasattr(self.taesd_decoder, "up_blocks"):
            raise AttributeError("decoder must expose conv_in and up_blocks for _forward_from_up_blocks path.")

        x = self.taesd_decoder.conv_in(z)
        up_blocks = self.taesd_decoder.up_blocks
        if len(up_blocks) < 4:
            raise RuntimeError(f"Expected >=4 up_blocks, got {len(up_blocks)}")

        cond_list = [cond32, cond64, cond128, cond256]
        sft_list = [self.sft_32, self.sft_64, self.sft_128, self.sft_256]

        for i in range(4):
            blk = up_blocks[i]
            resnets = getattr(blk, "resnets", None)
            upsamplers = getattr(blk, "upsamplers", None)

            if resnets is None:
                # 若 block 无可分解子结构，则直接执行（仅最后一个 block 允许该路径）。
                if i < 3:
                    raise RuntimeError(
                        "up_block without 'resnets' cannot satisfy pre-upsample SFT injection for scales 32/64/128."
                    )
                x = blk(x)
                x = sft_list[i](x, cond_list[i])
                continue

            for res in resnets:
                x = res(x)

            # 在每个尺度的残差块之后、上采样之前注入 SFT。
            x = sft_list[i](x, cond_list[i])

            if i < 3:
                if upsamplers is None or len(upsamplers) == 0:
                    raise RuntimeError(f"up_block[{i}] missing upsamplers; cannot reach next scale.")
                x = self._run_module_list(upsamplers, x)
            else:
                # 第四个 block 对应 256 尺度，不再做上采样。
                if upsamplers is not None and len(upsamplers) > 0:
                    x = self._run_module_list(upsamplers, x)

        if not hasattr(self.taesd_decoder, "conv_out"):
            raise AttributeError("decoder must expose conv_out.")
        x = self.taesd_decoder.conv_out(x)
        return x

    def forward(self, z: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        参数：
            z: [B, 16, H, W]（H/W 可变）
            cond_dict: 需包含 cond32/cond64/cond128/cond256（或别名 32/64/128/256）
        """
        if z.ndim != 4:
            raise ValueError(
                f"z must be [B,{self.latent_channels},H,W], got shape={z.shape}"
            )
        if z.shape[1] != self.latent_channels:
            raise ValueError(
                f"z must be [B,{self.latent_channels},H,W], got shape={z.shape}"
            )

        cond32 = self._fetch_cond(cond_dict, "cond32", "32")
        cond64 = self._fetch_cond(cond_dict, "cond64", "64")
        cond128 = self._fetch_cond(cond_dict, "cond128", "128")
        cond256 = self._fetch_cond(cond_dict, "cond256", "256")

        # 先按 diffusers 常见命名尝试；不满足时退化到 layers 顺序图。
        if hasattr(self.taesd_decoder, "conv_in") and hasattr(self.taesd_decoder, "up_blocks"):
            return self._forward_from_up_blocks(z, cond32, cond64, cond128, cond256)

        if hasattr(self.taesd_decoder, "layers"):
            return self._forward_from_layers(z, cond32, cond64, cond128, cond256)

        raise RuntimeError(
            "Unsupported TAESD decoder structure. Expected either (conv_in/up_blocks/conv_out) "
            "or (layers as nn.Sequential)."
        )
