import torch
import torch.nn as nn

from .kl_autoencoder import AutoencoderKL


class FMVAEWrapper(nn.Module):
	def __init__(self, pretrained_path):
		super().__init__()
		self.vae = AutoencoderKL(ckpt_path=pretrained_path)
		self.vae = self.vae.to(torch.float16)
		self.vae.eval()
		self.vae.requires_grad_(False)

	def train(self, mode=True):
		super().train(False)
		self.vae.eval()
		return self

	def encode(self, x):
		vae_dtype = next(self.vae.parameters()).dtype
		x = x.to(dtype=vae_dtype)
		x = x * 2.0 - 1.0
		posterior = self.vae.encode(x, normalize=False)
		return (posterior.sample() * 0.18215).float()

	def decode(self, z):
		vae_dtype = next(self.vae.parameters()).dtype
		z = z.to(dtype=vae_dtype)
		x = self.vae.decode(z, denorm=True)
		x = (x + 1.0) / 2.0
		return torch.clamp(x.float(), 0, 1)
