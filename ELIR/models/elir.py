import math
import torch
import torch.nn as nn
from ELIR.models.load_model import get_model



def pos_emb(t, t_dim, scale=1000):
    assert t_dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
    t = torch.tensor([t])
    half_dim = t_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim,device=t.device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb


class SFT_Refiner(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=16):
        super(SFT_Refiner, self).__init__()
        self.cond_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
        )
        self.alpha_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.beta_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Keep step-0 behavior as exact identity: alpha=0, beta=0.
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, x_base, x_lq):
        cond = self.cond_net(x_lq)
        alpha = self.alpha_head(cond)
        beta = self.beta_head(cond)
        x_final = x_base * (alpha + 1.0) + beta
        return torch.clamp(x_final, 0.0, 1.0)


class Elir(nn.Module):
    def __init__(self, fm_cfg, fmir_cfg, mmse_cfg, enc_cfg, dec_cfg, sft_cfg=None):
        super(Elir, self).__init__()
        sft_cfg = sft_cfg or {}
        self.fmir_cfg = fmir_cfg
        self.mmse_cfg = mmse_cfg
        self.enc_cfg = enc_cfg
        self.dec_cfg = dec_cfg
        self.sft_cfg = sft_cfg
        self.K = fm_cfg.get("k_steps")
        self.latent_shape = fm_cfg.get("latent_shape")
        self.sigma_s = fm_cfg.get("sigma_s",0.1)
        self.dynamic_noise = fm_cfg.get("dynamic_noise",True)
        self.force_static_noise = fm_cfg.get("force_static_noise", False)
        self.noise_seed = int(fm_cfg.get("noise_seed", 2025))
        self.detach_latent_path = fm_cfg.get("detach_latent_path", True)
        self.t_emb_dim = fmir_cfg.get("t_emb_dim",160)
        self.dt = 1/self.K
        self.fmir = get_model(fmir_cfg)
        self.mmse = get_model(mmse_cfg)
        self.enc = get_model(enc_cfg)
        self.dec = get_model(dec_cfg)
        sft_params = sft_cfg.get("params", {})
        self.sft_refiner = SFT_Refiner(
            in_channels=sft_params.get("in_channels", 3),
            out_channels=sft_params.get("out_channels", 3),
            hidden_channels=sft_params.get("hidden_channels", 16),
        )
        sft_trainable = sft_cfg.get("trainable", True)
        for p in self.sft_refiner.parameters():
            p.requires_grad = bool(sft_trainable)
        if sft_trainable:
            self.sft_refiner.train()
        else:
            self.sft_refiner.eval()
        self.noise = self.sigma_s * torch.randn((1, *self.latent_shape))
        self._static_noise_cache = {}

    def _encode_input(self, x):
        # Cached latent training passes 4- or 16-channel latents directly.
        if x.shape[1] in [4, 16]:
            return x.float()
        if hasattr(self.enc, "encode"):
            return self.enc.encode(x).float()
        if hasattr(self.enc, "encoder"):
            return self.enc.encoder(x).float()
        return self.enc(x).float()

    def _decode_latent(self, z):
        if hasattr(self.dec, "decode"):
            return self.dec.decode(z)
        if hasattr(self.dec, "decoder"):
            return self.dec.decoder(z)
        return self.dec(z)

    def _latent_path_frozen(self):
        modules = [self.enc, self.mmse, self.fmir]
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    return False
        return True

    def _sample_noise(self, z, device):
        if self.force_static_noise or (not self.dynamic_noise):
            base = self.noise.to(device=device, dtype=z.dtype)
            if tuple(base.shape[1:]) == tuple(z.shape[1:]):
                return base.expand_as(z)

            # Support variable latent resolutions with deterministic static noise per shape.
            key = (
                int(z.shape[1]),
                int(z.shape[2]),
                int(z.shape[3]),
                str(z.dtype),
                str(device),
            )
            if key not in self._static_noise_cache:
                gen = torch.Generator(device=device)
                gen.manual_seed(self.noise_seed)
                self._static_noise_cache[key] = self.sigma_s * torch.randn(
                    (1, z.shape[1], z.shape[2], z.shape[3]),
                    generator=gen,
                    device=device,
                    dtype=z.dtype,
                )
            return self._static_noise_cache[key].expand_as(z)
        return self.sigma_s * torch.randn_like(z)

    def _run_latent_ode(self, x, track_grad=True):
        if track_grad:
            z = self._encode_input(x)
            noise = self._sample_noise(z, x.device)
            z0 = self.mmse(z) + noise
            dt = 0
            for _ in range(self.K):
                z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
                dt += self.dt
            return z0

        with torch.no_grad():
            z = self._encode_input(x)
            noise = self._sample_noise(z, x.device)
            z0 = self.mmse(z) + noise
            dt = 0
            for _ in range(self.K):
                z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
                dt += self.dt
            return z0.detach()

    def _refine_output(self, x_base, x_lq):
        if x_lq is None or x_lq.shape[1] != 3:
            return x_base
        if x_lq.shape[-2:] != x_base.shape[-2:]:
            x_lq = torch.nn.functional.interpolate(
                x_lq,
                size=x_base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.sft_refiner(x_base, x_lq)

    def collapse(self):
        self.fmir.collapse()
        self.mmse.collapse()

    def load_weights(self, path):
        if path:
            state_dict = torch.load(path, map_location="cpu")
            if path.endswith(".ckpt"):
                if "state_dict" in state_dict:  # Prefer full Lightning state_dict (includes sft_refiner)
                    sd = state_dict["state_dict"]
                    cleaned = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
                    self.load_state_dict(cleaned, strict=False)
                elif "state_dict_fmir" in state_dict:  # legacy full ckpt with split submodules
                    sd_fmir = state_dict["state_dict_fmir"]
                    self.fmir.load_state_dict(sd_fmir)
                    sd_mmse = state_dict["state_dict_mmse"]
                    self.mmse.load_state_dict(sd_mmse)
                    sd_enc = state_dict["state_dict_enc"]
                    self.enc.load_state_dict(sd_enc)
                    sd_dec = state_dict["state_dict_dec"]
                    self.dec.load_state_dict(sd_dec)
                    if "state_dict_sft" in state_dict:
                        self.sft_refiner.load_state_dict(state_dict["state_dict_sft"], strict=False)
                    self.collapse()
                else:
                    # Fallback: try loading directly
                    self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        self.to(x.device)

        _, _, ori_h, ori_w = x.shape
        pad_h = (64 - ori_h % 64) % 64
        pad_w = (64 - ori_w % 64) % 64

        if pad_h > 0 or pad_w > 0:
            x_safe = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_safe = x

        x_lq = x_safe if x_safe.shape[1] == 3 else None
        track_grad = (not self.detach_latent_path) and (not self._latent_path_frozen())

        if track_grad:
            z = self._encode_input(x_safe)
            # Keep static-noise latent mapping for stable residual refinement.
            noise = self.noise.to(device=x_safe.device, dtype=z.dtype)
            if noise.shape[-2:] != z.shape[-2:]:
                noise = torch.nn.functional.interpolate(
                    noise,
                    size=z.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            noise = noise.expand_as(z)

            z0 = self.mmse(z) + noise
            dt = 0
            for _ in range(self.K):
                z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x_safe.device))
                dt += self.dt
        else:
            with torch.no_grad():
                z = self._encode_input(x_safe)
                # Keep static-noise latent mapping for stable residual refinement.
                noise = self.noise.to(device=x_safe.device, dtype=z.dtype)
                if noise.shape[-2:] != z.shape[-2:]:
                    noise = torch.nn.functional.interpolate(
                        noise,
                        size=z.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                noise = noise.expand_as(z)

                z0 = self.mmse(z) + noise
                dt = 0
                for _ in range(self.K):
                    z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x_safe.device))
                    dt += self.dt
                z0 = z0.detach()

        x_base = self._decode_latent(z0)
        y_padded = self._refine_output(x_base, x_lq)
        y_restored = y_padded[:, :, :ori_h, :ori_w]
        return y_restored

    def inference(self, x):
        y = self(x)
        out = torch.clip(y, min=0, max=1)
        return out

    def trajectories_pixel(self, x):
        self.to(x.device)
        x_lq = x if x.shape[1] == 3 else None
        z = self._encode_input(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        x_base = self._decode_latent(z0.clone())
        trajs = [self._refine_output(x_base, x_lq)]
        dt = 0
        for k in range(self.K):
            z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
            dt += self.dt
            x_base = self._decode_latent(z0).clone()
            trajs.append(self._refine_output(x_base, x_lq))
        return trajs

    def trajectories(self, x):
        self.to(x.device)
        z = self._encode_input(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        trajs = [z0.clone()]
        dt = 0
        for k in range(self.K):
            z0 = z0 + self.dt * self.fmir(z0, pos_emb(dt, self.t_emb_dim).to(x.device))
            dt += self.dt
            trajs.append(z0.clone())
        return trajs