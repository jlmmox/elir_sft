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


def _load_dinov2_robust(model_name):
    """加载 DINOv2, 网络不通时从本地缓存加载。"""
    try:
        return torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        print(f"[dino_align] 网络不可用 ({e}), 尝试本地缓存 ...")
        import os, sys
        hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        # dinov2 源码目录
        dinov2_src = os.path.join(hub_dir, "dinov2")
        if os.path.isdir(dinov2_src):
            sys.path.insert(0, hub_dir)  # 让 import dinov2 生效
        return torch.hub.load(hub_dir, model_name, source="local")


class DINOv2Encoder(nn.Module):
    """冻结的 DINOv2 ViT-B/14，提取 HQ 的 patch token 特征。"""

    def __init__(self, device="cuda", model_name="dinov2_vitb14"):
        super().__init__()
        self.encoder = _load_dinov2_robust(model_name)
        self.encoder.to(device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def encode_no_grad(self, x):
        """HQ 侧调用，不需要梯度穿过 DINO。"""
        x = self._preprocess(x)
        with torch.no_grad():
            out = self.encoder.forward_features(x)
        return out["x_norm_patchtokens"]

    def encode_with_grad(self, x):
        """MMSE decode 侧调用，梯度必须穿过 DINO 回传。"""
        x = self._preprocess(x)
        out = self.encoder.forward_features(x)
        return out["x_norm_patchtokens"]

    def forward(self, x_hq):
        """默认无梯度（兼容旧代码）"""
        return self.encode_no_grad(x_hq)

    @staticmethod
    def _preprocess(x):
        x = x / 255.0 if x.max() > 1.0 else x
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        return x


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
        z = F.interpolate(z_mmse, size=(16, 16), mode="bilinear", align_corners=False)
        z = self.net(z)
        z = z.flatten(2).transpose(1, 2)
        return z


class DinoSpatialProjector(nn.Module):
    """将 DINOv2 patch tokens [B,256,768] 投影到空间域 [B,16,32,32]，
    与 TAESD 潜在空间对齐以做 SK Fusion。"""

    def __init__(self, dino_dim=768, out_channels=16):
        super().__init__()
        self.proj = nn.Conv2d(dino_dim, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(4, out_channels)

    def forward(self, patch_tokens):
        B, N, D = patch_tokens.shape
        H = int(N ** 0.5)
        x = patch_tokens.transpose(1, 2).reshape(B, D, H, H)
        x = self.proj(x)
        x = self.norm(x)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        return x


class SKFusion(nn.Module):
    """SK-Net 双路通道自适应融合: U_taesd + U_dino -> V_fused。
    Softmax 逐通道生成 a,b (a+b=1), 在两个冻结锚点间做最优凸组合。"""

    def __init__(self, channels=16, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 4)
        # U_taesd+U_dino 逐元素相加 -> GAP -> [B,channels]
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels * 2),
        )
        with torch.no_grad():
            self.fc[2].bias[0::2] = 3.0
            self.fc[2].bias[1::2] = -3.0

    def forward(self, u_taesd, u_dino):
        s = self.gap(u_taesd + u_dino).flatten(1)
        z = self.fc(s).view(-1, 2, u_taesd.shape[1])
        w = F.softmax(z, dim=1)
        a = w[:, 0, :].view(-1, u_taesd.shape[1], 1, 1)
        b = w[:, 1, :].view(-1, u_taesd.shape[1], 1, 1)
        return a * u_taesd + b * u_dino
