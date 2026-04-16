import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class SDVAEWrapper(nn.Module):
    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        # Keep SD-VAE in fp16; bf16 is known to produce NaNs/black outputs on some setups.
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        self.scaling_factor = float(scaling_factor)

        self.vae.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float16)
        x = x * 2.0 - 1.0
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mode() * self.scaling_factor
        return z.float()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Intentionally keep autograd enabled to backprop pixel-space loss to the U-Net.
        z = z.to(dtype=torch.float16) / self.scaling_factor
        x = self.vae.decode(z).sample
        x = (x + 1.0) / 2.0
        return torch.clamp(x.float(), 0.0, 1.0)
