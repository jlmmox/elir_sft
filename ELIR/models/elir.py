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


class _VAEBundle:
    def __init__(self, enc, dec):
        self.enc = enc
        self.dec = dec

    def parameters(self):
        if self.enc is None or self.dec is None:
            return
        for p in self.enc.parameters():
            yield p
        for p in self.dec.parameters():
            yield p

    def eval(self):
        if self.enc is None or self.dec is None:
            return
        self.enc.eval()
        self.dec.eval()


class Elir(nn.Module):
    def __init__(self, fm_cfg, fmir_cfg, mmse_cfg, enc_cfg=None, dec_cfg=None, sft_cfg=None):
        super(Elir, self).__init__()
        self.fmir_cfg = fmir_cfg
        self.mmse_cfg = mmse_cfg
        self.enc_cfg = enc_cfg
        self.dec_cfg = dec_cfg
        self.sft_cfg = sft_cfg or {}
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
        self.enc = get_model(enc_cfg) if enc_cfg is not None else None
        self.dec = get_model(dec_cfg) if dec_cfg is not None else None
        self.vae = _VAEBundle(self.enc, self.dec)
        if self.enc is not None and self.dec is not None:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()
        self.noise = self.sigma_s * torch.randn((1, *self.latent_shape))
        self._static_noise_cache = {}

    def train(self, mode=True):
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        return self

    def _encode_input(self, x):
        # Cached latent training passes 4- or 16-channel latents directly.
        if x.shape[1] in [4, 16]:
            return x.float()
        if self.enc is None:
            return x.float()
        if hasattr(self.enc, "encode"):
            return self.enc.encode(x).float()
        if hasattr(self.enc, "encoder"):
            return self.enc.encoder(x).float()
        return self.enc(x).float()

    def _decode_latent(self, z):
        if self.dec is None:
            return z
        if hasattr(self.dec, "decode"):
            return self.dec.decode(z)
        if hasattr(self.dec, "decoder"):
            return self.dec.decoder(z)
        return self.dec(z)

    def _build_fmir_condition(self, x_lq):
        if x_lq is None or x_lq.shape[1] != 3:
            return None
        if hasattr(self.fmir, "make_condition"):
            return self.fmir.make_condition(x_lq)
        if hasattr(self.fmir, "condition_stem"):
            cond = self.fmir.condition_stem(x_lq)
            if isinstance(cond, dict):
                return cond
            if isinstance(cond, (tuple, list)) and len(cond) == 3:
                return {"64": cond[0], "32": cond[1], "16": cond[2]}
            raise ValueError("condition_stem must return a 3-tuple/list or a dict with keys 64/32/16.")
        raise ValueError("FMIR model must provide condition_stem or make_condition for CGFM.")

    def _latent_path_frozen(self):
        modules = [module for module in [self.enc, self.mmse, self.fmir] if module is not None]
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
        x_lq = x if x.shape[1] == 3 else None
        cond = self._build_fmir_condition(x_lq)
        if track_grad:
            z = self._encode_input(x)
            noise = self._sample_noise(z, x.device)
            z0 = self.mmse(z) + noise
            dt = 0
            for _ in range(self.K):
                t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x.device, dtype=z0.dtype)
                z0 = z0 + self.dt * self.fmir(
                    z0,
                    pos_emb(dt, self.t_emb_dim).to(x.device),
                    cond=cond,
                    t=t_tensor,
                )
                dt += self.dt
            return z0

        with torch.no_grad():
            z = self._encode_input(x)
            noise = self._sample_noise(z, x.device)
            z0 = self.mmse(z) + noise
            dt = 0
            for _ in range(self.K):
                t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x.device, dtype=z0.dtype)
                z0 = z0 + self.dt * self.fmir(
                    z0,
                    pos_emb(dt, self.t_emb_dim).to(x.device),
                    cond=cond,
                    t=t_tensor,
                )
                dt += self.dt
            return z0.detach()

    def collapse(self):
        if hasattr(self.fmir, "collapse"):
            self.fmir.collapse()
        if hasattr(self.mmse, "collapse"):
            self.mmse.collapse()

    def _load_optional_state(self, module, state_dict, key):
        if module is not None:
            sub_state = state_dict.get(key)
            if sub_state is not None:
                module.load_state_dict(sub_state, strict=False)

    def load_weights(self, path):
        if path:
            state_dict = torch.load(path, map_location="cpu")
            if path.endswith(".ckpt"):
                if "state_dict" in state_dict:
                    sd = state_dict["state_dict"]
                    cleaned = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
                    missing, unexpected = self.load_state_dict(cleaned, strict=False)
                    if len(missing) > 0 or len(unexpected) > 0:
                        print(
                            f"[load_weights] model non-strict load: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}"
                        )
                elif "state_dict_fmir" in state_dict:  # legacy full ckpt with split submodules
                    sd_fmir = state_dict["state_dict_fmir"]
                    missing_fmir, unexpected_fmir = self.fmir.load_state_dict(sd_fmir, strict=False)
                    if len(missing_fmir) > 0 or len(unexpected_fmir) > 0:
                        print(
                            f"[load_weights] fmir non-strict load: "
                            f"missing={len(missing_fmir)}, unexpected={len(unexpected_fmir)}"
                        )
                    sd_mmse = state_dict["state_dict_mmse"]
                    missing_mmse, unexpected_mmse = self.mmse.load_state_dict(sd_mmse, strict=False)
                    if len(missing_mmse) > 0 or len(unexpected_mmse) > 0:
                        print(
                            f"[load_weights] mmse non-strict load: "
                            f"missing={len(missing_mmse)}, unexpected={len(unexpected_mmse)}"
                        )
                    self._load_optional_state(self.enc, state_dict, "state_dict_enc")
                    self._load_optional_state(self.dec, state_dict, "state_dict_dec")
                    self.collapse()
                else:
                    # Fallback: try loading directly
                    missing, unexpected = self.load_state_dict(state_dict, strict=False)
                    if len(missing) > 0 or len(unexpected) > 0:
                        print(
                            f"[load_weights] fallback non-strict load: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}"
                        )
            else:
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    print(
                        f"[load_weights] model non-strict load: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )

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
        cond = self._build_fmir_condition(x_lq)
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
                t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x_safe.device, dtype=z0.dtype)
                z0 = z0 + self.dt * self.fmir(
                    z0,
                    pos_emb(dt, self.t_emb_dim).to(x_safe.device),
                    cond=cond,
                    t=t_tensor,
                )
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
                    t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x_safe.device, dtype=z0.dtype)
                    z0 = z0 + self.dt * self.fmir(
                        z0,
                        pos_emb(dt, self.t_emb_dim).to(x_safe.device),
                        cond=cond,
                        t=t_tensor,
                    )
                    dt += self.dt
                z0 = z0.detach()
        y_padded = self._decode_latent(z0)
        y_restored = y_padded[:, :, :ori_h, :ori_w]
        return y_restored

    def inference(self, x):
        y = self(x)
        out = torch.clip(y, min=0, max=1)
        return out

    def trajectories_pixel(self, x):
        self.to(x.device)
        cond = self._build_fmir_condition(x if x.shape[1] == 3 else None)
        z = self._encode_input(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        trajs = [self._decode_latent(z0.clone())]
        dt = 0
        for k in range(self.K):
            t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x.device, dtype=z0.dtype)
            z0 = z0 + self.dt * self.fmir(
                z0,
                pos_emb(dt, self.t_emb_dim).to(x.device),
                cond=cond,
                t=t_tensor,
            )
            dt += self.dt
            trajs.append(self._decode_latent(z0).clone())
        return trajs

    def trajectories(self, x):
        self.to(x.device)
        x_lq = x if x.shape[1] == 3 else None
        cond = self._build_fmir_condition(x_lq)
        z = self._encode_input(x)
        z0 = self.mmse(z) + self.noise.to(x.device)
        trajs = [z0.clone()]
        dt = 0
        for k in range(self.K):
            t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x.device, dtype=z0.dtype)
            z0 = z0 + self.dt * self.fmir(
                z0,
                pos_emb(dt, self.t_emb_dim).to(x.device),
                cond=cond,
                t=t_tensor,
            )
            dt += self.dt
            trajs.append(z0.clone())
        return trajs