"""Perceptual loss: LPIPS with VGG fallback.

Primary: LPIPS (AlexNet or VGG backbone, pretrained) — standard for image restoration.
Fallback: torchvision VGG19 feature-matching loss — no extra pip install needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSWrapper(nn.Module):
    """Thin wrapper around lpips.LPIPS to keep a consistent interface."""

    def __init__(self, net="alex", device="cuda"):
        super().__init__()
        import lpips
        self.lpips = lpips.LPIPS(net=net).to(device)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return self.lpips(pred, target, normalize=True).mean()


class VGGPerceptualLoss(nn.Module):
    """VGG19 feature-matching perceptual loss (no LPIPS dependency).

    Uses features from layers: relu1_2, relu2_2, relu3_4, relu4_4, relu5_4.
    Each feature map is normalized by its spatial size for balanced contribution.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.slice1 = nn.Sequential(*list(vgg.features.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.features.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.features.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.features.children())[18:27])# relu4_4
        self.slice5 = nn.Sequential(*list(vgg.features.children())[27:36])# relu5_4

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, pred, target):
        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x_pred, x_target = pred, target

        for sl in [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]:
            x_pred = sl(x_pred)
            x_target = sl(x_target)
            # Weighted by 1 / (C * H * W) to balance layer contributions
            loss += F.l1_loss(x_pred, x_target) / x_pred.shape[1]

        return loss


def create_perceptual_loss(device="cuda", prefer_lpips=True):
    """Factory: returns LPIPS (if installed) or VGG fallback.

    Returns (module, name_str).
    """
    if prefer_lpips:
        try:
            import lpips
            net = LPIPSWrapper(net="alex", device=device)
            return net, "lpips_alex"
        except ImportError:
            print("[perceptual] lpips not installed, falling back to VGG19.")
    net = VGGPerceptualLoss(device=device)
    return net, "vgg19"
