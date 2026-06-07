"""DINOv2 表示对齐：将 MMSE 输出对齐到 DINOv2(HQ) 的 patch token 特征。

训练期辅助监督：DINOv2 冻结，仅在训练时提取 HQ 特征。
推理时不参与——DINOv2 和 Projector 不进入 forward 图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet 标准化参数
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv2Encoder(nn.Module):
    """冻结的 DINOv2 ViT-B/14，提取 HQ 的 patch token 特征。"""

    def __init__(self, device="cuda", model_name="dinov2_vitb14"):
        super().__init__()
        self.encoder = torch.hub.load("facebookresearch/dinov2", model_name)
        self.encoder.to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_hq):
        """x_hq: [B, 3, H, W] range [0, 1] → patch tokens [B, N, 768]"""
        x = x_hq / 255.0 if x_hq.max() > 1.0 else x_hq
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        x = (x - mean) / std
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        with torch.no_grad():
            out = self.encoder.forward_features(x)
        return out["x_norm_patchtokens"]  # [B, 256, 768] for 224^2 input


class DINOProjector(nn.Module):
    """将 MMSE latent [B, 16, 32, 32] 投影到 DINOv2 token 空间 [B, 256, 768]。"""

    def __init__(self, in_channels=16, dino_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, dino_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z_mmse):
        """z_mmse [B, 16, 32, 32] → 投影 → [B, 256, 768]"""
        # 下采样到 16×16 匹配 DINOv2 的 patch grid
        z = F.interpolate(z_mmse, size=(16, 16), mode="bilinear", align_corners=False)
        z = self.net(z)  # [B, 768, 16, 16]
        z = z.flatten(2).transpose(1, 2)  # [B, 256, 768]
        return z
