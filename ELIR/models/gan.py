"""PatchGAN NLayerDiscriminator with SpectralNorm for stable GAN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with SpectralNorm on all conv layers.

    Receptive field: ~70x70 (n_layers=3) or ~34x34 (n_layers=2).
    Outputs a [B, 1, H_patch, W_patch] logits map.
    """

    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.n_layers = int(n_layers)

        kw = 4
        padw = 1

        layers = [
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for i in range(1, n_layers):
            ch_next = min(ch * 2, base_channels * 8)
            layers.append(
                spectral_norm(nn.Conv2d(ch, ch_next, kernel_size=kw, stride=2, padding=padw, bias=False))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = ch_next

        # Final two convs: stride 1, outputs single-channel logits
        layers.append(
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=kw, stride=1, padding=padw, bias=False))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            spectral_norm(nn.Conv2d(ch, 1, kernel_size=kw, stride=1, padding=padw))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_d_loss(logits_real, logits_fake):
    """Hinge discriminator loss."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake):
    """Hinge generator loss."""
    return -torch.mean(logits_fake)
