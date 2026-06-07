import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ELIR.models.load_model import get_model
from ELIR.models.wavelet_stem import WaveletStem
from torchvision.utils import save_image
class CondFusion1x1(nn.Module):
    """Spatial-first condition fusion with identity-biased initialization."""

    def __init__(self, spatial_ch: int, wavelet_ch: int, out_ch: int):
        super().__init__()
        self.spatial_ch = int(spatial_ch)
        self.wavelet_ch = int(wavelet_ch)
        self.out_ch = int(out_ch)
        self.proj = nn.Conv2d(self.spatial_ch + self.wavelet_ch, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            keep = min(self.spatial_ch, self.out_ch)
            for idx in range(keep):
                self.proj.weight[idx, idx, 0, 0] = 1.0

    def forward(self, spatial_cond: torch.Tensor, wavelet_cond: torch.Tensor) -> torch.Tensor:
        if spatial_cond is None:
            return wavelet_cond
        if wavelet_cond is None:
            return spatial_cond
        if spatial_cond.shape[-2:] != wavelet_cond.shape[-2:]:
            wavelet_cond = F.interpolate(wavelet_cond, size=spatial_cond.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([spatial_cond, wavelet_cond], dim=1)
        return self.proj(fused)




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
    def __init__(self, fm_cfg, fmir_cfg, mmse_cfg, enc_cfg=None, dec_cfg=None, sft_cfg=None, wavelet_cfg=None):
        super(Elir, self).__init__()
        self.fmir_cfg = fmir_cfg
        self.mmse_cfg = mmse_cfg
        self.enc_cfg = enc_cfg
        self.dec_cfg = dec_cfg
        self.sft_cfg = sft_cfg or {}
        self.wavelet_cfg = wavelet_cfg or {}
        self.K = fm_cfg.get("k_steps")
        self.latent_shape = fm_cfg.get("latent_shape")
        self.sigma_s = fm_cfg.get("sigma_s",0.1)
        self.dynamic_noise = fm_cfg.get("dynamic_noise",True)
        self.force_static_noise = fm_cfg.get("force_static_noise", False)
        self.noise_seed = int(fm_cfg.get("noise_seed", 2025))
        self.detach_latent_path = fm_cfg.get("detach_latent_path", True)
        self.ablation_skip_fmir = bool(fm_cfg.get("ablation_skip_fmir", False))
        self.ablation_skip_mmse = bool(fm_cfg.get("ablation_skip_mmse", False))
        self.t_emb_dim = fmir_cfg.get("t_emb_dim",160)
        self.dt = 1/self.K
        self.fmir = get_model(fmir_cfg)
        self.mmse = get_model(mmse_cfg)
        self.enc = get_model(enc_cfg) if enc_cfg is not None else None

        # DINOv2 表示对齐 Projector (MMSE latent → DINOv2 feature space)
        self.dino_projector = None
        if bool(fm_cfg.get("dino_align", False)):
            from ELIR.training.dino_align import DINOProjector
            self.dino_projector = DINOProjector(
                in_channels=int(fmir_cfg.get("params", {}).get("in_channels", 16)),
                dino_dim=768,
            )
        # 为 encoder_skip_fusion 注入 encoder 引用 + 自动推导 FMIR cond 通道数
        if dec_cfg is not None and dec_cfg.get("name") == "encoder_skip_fusion":
            dec_cfg = dict(dec_cfg)
            dec_params = dict(dec_cfg.get("params", {}))
            enc_ref = self.enc.encoder if hasattr(self.enc, "encoder") else self.enc
            dec_params["taesd_encoder"] = enc_ref
            if "fmir_cond_channels" not in dec_params:
                fmir_params = fmir_cfg.get("params", {})
                cond_pyramid = fmir_params.get("cond_pyramid_channels", (16, 32, 64))
                if len(cond_pyramid) == 3:
                    c256, c128, c64 = cond_pyramid
                    c32 = c64
                elif len(cond_pyramid) == 4:
                    c256, c128, c64, c32 = cond_pyramid
                else:
                    raise ValueError(f"Unexpected cond_pyramid_channels length: {len(cond_pyramid)}")
                dec_params["fmir_cond_channels"] = {"256": c256, "128": c128, "64": c64, "32": c32}
            dec_cfg["params"] = dec_params

        # 为 encoder_skip_wavelet_fusion 注入 encoder 引用 + 自动推导两路 cond 通道数
        if dec_cfg is not None and dec_cfg.get("name") == "encoder_skip_wavelet_fusion":
            dec_cfg = dict(dec_cfg)
            dec_params = dict(dec_cfg.get("params", {}))
            enc_ref = self.enc.encoder if hasattr(self.enc, "encoder") else self.enc
            dec_params["taesd_encoder"] = enc_ref
            if "fmir_cond_channels" not in dec_params:
                fmir_params = fmir_cfg.get("params", {})
                cond_pyramid = fmir_params.get("cond_pyramid_channels", (16, 32, 64))
                if len(cond_pyramid) == 3:
                    c256, c128, c64 = cond_pyramid
                    c32 = c64
                elif len(cond_pyramid) == 4:
                    c256, c128, c64, c32 = cond_pyramid
                else:
                    raise ValueError(f"Unexpected cond_pyramid_channels length: {len(cond_pyramid)}")
                dec_params["fmir_cond_channels"] = {"256": c256, "128": c128, "64": c64, "32": c32}
            if "wavelet_cond_channels" not in dec_params:
                dec_params["wavelet_cond_channels"] = {"256": 16, "128": 32, "64": 64, "32": 64}
            dec_cfg["params"] = dec_params
        self.dec = get_model(dec_cfg) if dec_cfg is not None else None
        self.use_wavelet_decoder_cond = bool(self.wavelet_cfg.get("enabled", False))
        self.use_wavelet_fmir_cond = bool(self.wavelet_cfg.get("fmir_use_wavelet_cond", False))
        self.wavelet_fuse_with_spatial = bool(self.wavelet_cfg.get("fuse_with_spatial", True))
        self.wavelet_visualize = bool(self.wavelet_cfg.get("visualize", False))
        self.wavelet_visualize_every = int(self.wavelet_cfg.get("visualize_every", 200))
        self.wavelet_visualize_dir = str(self.wavelet_cfg.get("visualize_dir", "./runs/wavelet_debug"))
        self.wavelet_visualize_max_images = int(self.wavelet_cfg.get("visualize_max_images", -1))
        self._wavelet_visualize_step = 0
        self._wavelet_visualize_saved = 0
        self.wavelet_stem = None
        self.decoder_cond_fusions = None
        self.fmir_cond_fusions = None
        if self.use_wavelet_decoder_cond:
            self.wavelet_stem = WaveletStem(
                in_channels=int(self.wavelet_cfg.get("in_channels", 3)),
                base_channels=int(self.wavelet_cfg.get("base_channels", 16)),
                use_ll=bool(self.wavelet_cfg.get("use_ll", True)),
                use_band_attn=bool(self.wavelet_cfg.get("band_attn", False)),
            )
            if not bool(self.wavelet_cfg.get("trainable", True)):
                for p in self.wavelet_stem.parameters():
                    p.requires_grad = False
                self.wavelet_stem.eval()
            if self.wavelet_fuse_with_spatial:
                spatial_cond_channels = self._resolve_spatial_cond_channels(fmir_cfg)
                wavelet_cond_channels = self.wavelet_stem.cond_channels
                self.decoder_cond_fusions = nn.ModuleDict({
                    key: CondFusion1x1(
                        spatial_ch=spatial_cond_channels[key],
                        wavelet_ch=wavelet_cond_channels[key],
                        out_ch=wavelet_cond_channels[key],
                    )
                    for key in ["256", "128", "64", "32"]
                })
            if self.wavelet_visualize:
                os.makedirs(self.wavelet_visualize_dir, exist_ok=True)

            # FMIR 端 wavelet 条件融合（将频带信息注入 FMIR 的 SFT 条件）
            if self.use_wavelet_fmir_cond:
                fmir_spatial_channels = self._resolve_spatial_cond_channels(fmir_cfg)
                fmir_wavelet_channels = self.wavelet_stem.cond_channels
                self.fmir_cond_fusions = nn.ModuleDict({
                    key: CondFusion1x1(
                        spatial_ch=fmir_spatial_channels[key],
                        wavelet_ch=fmir_wavelet_channels[key],
                        out_ch=fmir_spatial_channels[key],
                    )
                    for key in ["256", "128", "64", "32"]
                })
        self.enc_trainable = bool((enc_cfg or {}).get("trainable", False))
        self.dec_trainable = bool((dec_cfg or {}).get("trainable", False))
        self.latent_norm = None
        self.vae = _VAEBundle(self.enc, self.dec)
        if self.enc is not None:
            if not self.enc_trainable:
                for p in self.enc.parameters():
                    p.requires_grad = False
                self.enc.eval()
            else:
                self.enc.train()
                # Latent GN：REPA-E 风格，可由 use_enc_gn 开关独立控制
                if bool((enc_cfg or {}).get("use_enc_gn", True)):
                    enc_out_ch = int((enc_cfg or {}).get("params", {}).get("latent_channels", 16))
                    self.latent_norm = nn.GroupNorm(4, enc_out_ch, affine=True)

        if self.dec is not None:
            if not self.dec_trainable:
                for p in self.dec.parameters():
                    p.requires_grad = False
                self.dec.eval()
            else:
                # 对包装器保持"decoder 冻结、融合模块可训练"的安全状态。
                if hasattr(self.dec, "set_trainable_only"):
                    self.dec.set_trainable_only()
                elif hasattr(self.dec, "set_sft_trainable_only"):
                    self.dec.set_sft_trainable_only()
                self.dec.train()
        self.noise = self.sigma_s * torch.randn((1, *self.latent_shape))
        self._static_noise_cache = {}

    def train(self, mode=True):
        super().train(mode)
        if self.enc is not None:
            self.enc.eval()
        if self.dec is not None:
            if self.dec_trainable:
                if hasattr(self.dec, "set_trainable_only"):
                    self.dec.set_trainable_only()
                elif hasattr(self.dec, "set_sft_trainable_only"):
                    self.dec.set_sft_trainable_only()
                self.dec.train(mode)
            else:
                self.dec.eval()
        if self.wavelet_stem is not None:
            if any(p.requires_grad for p in self.wavelet_stem.parameters()):
                self.wavelet_stem.train(mode)
            else:
                self.wavelet_stem.eval()
        return self

    def _encode_input(self, x):
        # Cached latent training passes 4- or 16-channel latents directly.
        if x.shape[1] in [4, 16]:
            return x.float()
        if self.enc is None:
            return x.float()
        if hasattr(self.enc, "encode"):
            z = self.enc.encode(x).float()
        elif hasattr(self.enc, "encoder"):
            z = self.enc.encoder(x).float()
        else:
            return x.float()
        if self.latent_norm is not None:
            z = self.latent_norm(z)
        return z
        return self.enc(x).float()

    def _decode_latent(self, z, cond=None, x_lq=None, wavelet_cond=None):
        if self.dec is None:
            return z
        if hasattr(self.dec, "_use_encoder_skip_wavelet"):
            return self.dec(z, cond, wavelet_cond, x_lq)
        if hasattr(self.dec, "_use_encoder_skip"):
            return self.dec(z, cond, x_lq)
        if hasattr(self.dec, "_force_sft_trainable_only"):
            return self.dec(z, cond)
        if hasattr(self.dec, "decode"):
            return self.dec.decode(z)
        if hasattr(self.dec, "decoder"):
            return self.dec.decoder(z)
        return self.dec(z)

    def _build_fmir_condition(self, x_lq, wavelet_cond=None):
        if x_lq is None or x_lq.shape[1] != 3:
            return None
        if hasattr(self.fmir, "make_condition"):
            return self.fmir.make_condition(x_lq)
        if hasattr(self.fmir, "condition_stem"):
            cond = self.fmir.condition_stem(x_lq)
            if isinstance(cond, dict):
                # 兼容新旧条件键位：优先使用 256/128/64；若是旧键位则就地转换。
                if ("256" in cond) and ("128" in cond) and ("64" in cond):
                    out = {"256": cond["256"], "128": cond["128"], "64": cond["64"]}
                    if "32" in cond:
                        out["32"] = cond["32"]
                elif ("64" in cond) and ("32" in cond) and ("16" in cond):
                    out = {"256": cond["64"], "128": cond["32"], "64": cond["16"]}
                else:
                    out = cond
                # 融合 wavelet 频带条件到 FMIR 条件中
                if wavelet_cond is not None and self.fmir_cond_fusions is not None:
                    fused = {}
                    for key in ["256", "128", "64", "32"]:
                        if key in out and key in wavelet_cond:
                            fused[key] = self.fmir_cond_fusions[key](out[key], wavelet_cond[key])
                        else:
                            fused[key] = out.get(key, wavelet_cond.get(key))
                    return fused
                return out
            if isinstance(cond, (tuple, list)) and len(cond) == 3:
                return {"256": cond[0], "128": cond[1], "64": cond[2]}
            raise ValueError("condition_stem must return a 3-tuple/list or a dict with keys 256/128/64 (legacy 64/32/16 is also accepted).")
        raise ValueError("FMIR model must provide condition_stem or make_condition for CGFM.")

    @staticmethod
    def _resolve_spatial_cond_channels(fmir_cfg):
        params = (fmir_cfg or {}).get("params", {})
        cond_pyramid_channels = params.get("cond_pyramid_channels", (16, 32, 64))
        if len(cond_pyramid_channels) == 3:
            c256, c128, c64 = cond_pyramid_channels
            c32 = c64
        elif len(cond_pyramid_channels) == 4:
            c256, c128, c64, c32 = cond_pyramid_channels
        else:
            raise ValueError(
                "cond_pyramid_channels must be length 3 (256/128/64) or 4 (256/128/64/32)."
            )
        return {"256": int(c256), "128": int(c128), "64": int(c64), "32": int(c32)}

    def _build_decoder_condition(self, x_lq, spatial_cond=None):
        if self.wavelet_stem is None:
            return spatial_cond
        if x_lq is None or x_lq.shape[1] != 3:
            return spatial_cond
        if self.wavelet_visualize:
            wavelet_cond, wavelet_debug = self.wavelet_stem(x_lq, return_debug=True)
            self._maybe_save_wavelet_debug(x_lq, wavelet_debug)
        else:
            wavelet_cond = self.wavelet_stem(x_lq)
        # 兼容旧版高频-only路径：不与spatial条件融合，直接使用wavelet条件。
        if not self.wavelet_fuse_with_spatial:
            return wavelet_cond

        if spatial_cond is None:
            return wavelet_cond
        if self.decoder_cond_fusions is None:
            return spatial_cond

        fused = {}
        for key in ["256", "128", "64", "32"]:
            fused[key] = self.decoder_cond_fusions[key](spatial_cond[key], wavelet_cond[key])
            fused[f"cond{key}"] = fused[key]
        return fused

    @staticmethod
    def _norm_to_01(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if t.ndim != 4:
            raise ValueError(f"Expect [B,C,H,W], got {t.shape}")
        t_min = t.amin(dim=(2, 3), keepdim=True)
        t_max = t.amax(dim=(2, 3), keepdim=True)
        return (t - t_min) / (t_max - t_min + eps)

    def _save_vis_tensor(self, tensor: torch.Tensor, name: str, step: int, keep_rgb: bool = False):
        # 仅保存 batch 中第一张图，减少 I/O 压力。
        vis = tensor[:1]
        if keep_rgb:
            # 对输入图保留 RGB 外观与原始雾感，不做每图 min-max 拉伸。
            vis = torch.clamp(vis, 0.0, 1.0)
        else:
            vis = vis.mean(dim=1, keepdim=True)
            vis = self._norm_to_01(vis)
        out_path = os.path.join(self.wavelet_visualize_dir, f"step_{step:07d}_{name}.png")
        save_image(vis.detach().cpu(), out_path)

    def _maybe_save_wavelet_debug(self, x_lq: torch.Tensor, wavelet_debug: dict):
        if (not self.wavelet_visualize) or wavelet_debug is None:
            return
        if self.wavelet_visualize_max_images > 0 and self._wavelet_visualize_saved >= self.wavelet_visualize_max_images:
            return
        step = int(self._wavelet_visualize_step)
        self._wavelet_visualize_step += 1

        if self.wavelet_visualize_every <= 0:
            return
        if step % self.wavelet_visualize_every != 0:
            return

        with torch.no_grad():
            self._save_vis_tensor(x_lq, "input_lq", step, keep_rgb=True)
            for name in ["ll", "lh", "hl", "hh", "cond128_base", "cond256", "cond128", "cond64", "cond32"]:
                if name in wavelet_debug and wavelet_debug[name] is not None:
                    self._save_vis_tensor(wavelet_debug[name], name, step)
            self._wavelet_visualize_saved += 1

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

    def _upgrade_legacy_state_dict(self, state_dict):
        """
        兼容旧 checkpoint 到当前结构。

        典型场景：旧版 wavelet stem 仅使用 3 个高频子带（9 通道），
        当前版本使用 LL/LH/HL/HH 共 12 通道。
        """
        if not isinstance(state_dict, dict):
            return state_dict

        key = "wavelet_stem.proj.0.weight"
        if key in state_dict:
            src = state_dict[key]
            dst = self.state_dict().get(key)
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
                # 旧版: [out, 9, 3, 3] -> 新版: [out, 12, 3, 3]
                if src.ndim == 4 and dst.ndim == 4 and src.shape[1] == 9 and dst.shape[1] == 12 and src.shape[0] == dst.shape[0]:
                    upgraded = torch.zeros_like(dst)
                    # 当前 full_band 顺序为 [ll, lh, hl, hh]，每个子带 3 通道。
                    # 旧版权重按 [lh, hl, hh] 训练，映射到新张量的后 9 通道。
                    upgraded[:, 3:, :, :] = src.to(device=dst.device, dtype=dst.dtype)
                    state_dict[key] = upgraded
                    print("[load_weights] Upgraded legacy wavelet_stem.proj.0.weight from 9ch to 12ch.")

        return state_dict

    def _filter_mismatched_keys(self, state_dict):
        """过滤 shape 不匹配键，避免 strict=False 仍因 size mismatch 抛错。"""
        model_sd = self.state_dict()
        filtered = {}
        skipped = []

        for k, v in state_dict.items():
            if k not in model_sd:
                filtered[k] = v
                continue
            mv = model_sd[k]
            if isinstance(v, torch.Tensor) and isinstance(mv, torch.Tensor) and v.shape != mv.shape:
                skipped.append((k, tuple(v.shape), tuple(mv.shape)))
                continue
            filtered[k] = v

        if skipped:
            print("[load_weights] skipped mismatched keys:")
            for k, s1, s2 in skipped:
                print(f"  - {k}: ckpt{s1} vs model{s2}")

        return filtered

    def load_weights(self, path):
        if path:
            state_dict = torch.load(path, map_location="cpu")
            if path.endswith(".ckpt"):
                if "state_dict" in state_dict:
                    sd = state_dict["state_dict"]
                    cleaned = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
                    cleaned = self._upgrade_legacy_state_dict(cleaned)
                    cleaned = self._filter_mismatched_keys(cleaned)
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
                    state_dict = self._upgrade_legacy_state_dict(state_dict)
                    state_dict = self._filter_mismatched_keys(state_dict)
                    missing, unexpected = self.load_state_dict(state_dict, strict=False)
                    if len(missing) > 0 or len(unexpected) > 0:
                        print(
                            f"[load_weights] fallback non-strict load: "
                            f"missing={len(missing)}, unexpected={len(unexpected)}"
                        )
            else:
                state_dict = self._upgrade_legacy_state_dict(state_dict)
                state_dict = self._filter_mismatched_keys(state_dict)
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

        # 统一计算 wavelet_cond（复用给 FMIR 和 decoder）
        wavelet_cond = None
        if self.wavelet_stem is not None and x_lq is not None:
            if self.wavelet_visualize:
                wavelet_cond, wavelet_debug = self.wavelet_stem(x_lq, return_debug=True)
                self._maybe_save_wavelet_debug(x_lq, wavelet_debug)
            else:
                wavelet_cond = self.wavelet_stem(x_lq)

        # FMIR 条件：CNN condition_stem + 可选 wavelet 频带融合
        spatial_cond = self._build_fmir_condition(
            x_lq,
            wavelet_cond=wavelet_cond if self.use_wavelet_fmir_cond else None,
        )

        # 三流互补 decoder：FMIR cond 与 wavelet cond 分别传入，不融合
        use_wavelet_skip = self.dec is not None and hasattr(self.dec, "_use_encoder_skip_wavelet")
        if use_wavelet_skip:
            decoder_cond = spatial_cond
        else:
            decoder_cond = self._build_decoder_condition(x_lq, spatial_cond=spatial_cond)

        # ---- 消融：跳过 FMIR ODE ----
        if self.ablation_skip_fmir:
            with torch.no_grad():
                z0 = self._encode_input(x_safe)
            y_padded = self._decode_latent(z0, cond=decoder_cond, x_lq=x_lq, wavelet_cond=wavelet_cond)
            y_restored = y_padded[:, :, :ori_h, :ori_w]
            return y_restored

        track_grad = (not self.detach_latent_path) and (not self._latent_path_frozen())

        # ---- 消融：跳过 MMSE ----
        if self.ablation_skip_mmse:
            z = self._encode_input(x_safe)
            noise = self.noise.to(device=x_safe.device, dtype=z.dtype)
            if noise.shape[-2:] != z.shape[-2:]:
                noise = torch.nn.functional.interpolate(
                    noise, size=z.shape[-2:], mode="bilinear", align_corners=False)
            z0 = z + noise
            dt = 0
            with torch.no_grad() if not track_grad else torch.enable_grad():
                for _ in range(self.K):
                    t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x_safe.device, dtype=z0.dtype)
                    z0 = z0 + self.dt * self.fmir(
                        z0, pos_emb(dt, self.t_emb_dim).to(x_safe.device),
                        cond=spatial_cond, t=t_tensor)
                    dt += self.dt
            y_padded = self._decode_latent(z0, cond=decoder_cond, x_lq=x_lq, wavelet_cond=wavelet_cond)
            y_restored = y_padded[:, :, :ori_h, :ori_w]
            return y_restored

        # track_grad already set above (after fmir ablation, before mmse ablation)

        if track_grad:
            z = self._encode_input(x_safe)
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
                    cond=spatial_cond,
                    t=t_tensor,
                )
                dt += self.dt
        else:
            with torch.no_grad():
                z = self._encode_input(x_safe)
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
                        cond=spatial_cond,
                        t=t_tensor,
                    )
                    dt += self.dt
                z0 = z0.detach()
        y_padded = self._decode_latent(z0, cond=decoder_cond, x_lq=x_lq, wavelet_cond=wavelet_cond)
        y_restored = y_padded[:, :, :ori_h, :ori_w]
        return y_restored

    def inference(self, x, use_tta=False):
        y = self(x)
        if use_tta and x.ndim == 4:
            y = self._tta_inference(x)
        out = torch.clip(y, min=0, max=1)
        return out

    def _tta_inference(self, x):
        """8 种几何变换 TTA：原图 + flip + transpose 组合取平均。"""
        # 8 种变换：(h_flip, v_flip, transpose)
        transforms = [
            (False, False, False),  # 原图
            (True,  False, False),  # 水平翻转
            (False, True,  False),  # 垂直翻转
            (True,  True,  False),  # 180°旋转
            (False, False, True),   # 转置
            (True,  False, True),   # 转置+水平翻转
            (False, True,  True),   # 转置+垂直翻转
            (True,  True,  True),   # 转置+180°旋转
        ]

        outputs = []
        for hf, vf, tr in transforms:
            x_t = x.clone()
            if tr:
                x_t = x_t.transpose(2, 3)
            if hf:
                x_t = torch.flip(x_t, [2])
            if vf:
                x_t = torch.flip(x_t, [3])

            y_t = self.forward(x_t)

            # 逆变换：先翻转再转置，顺序与正变换相反
            if vf:
                y_t = torch.flip(y_t, [3])
            if hf:
                y_t = torch.flip(y_t, [2])
            if tr:
                y_t = y_t.transpose(2, 3)

            outputs.append(y_t)

        return torch.stack(outputs, dim=0).mean(dim=0)

    def trajectories_pixel(self, x):
        self.to(x.device)

        _, _, ori_h, ori_w = x.shape
        pad_h = (64 - ori_h % 64) % 64
        pad_w = (64 - ori_w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x_lq = x if x.shape[1] == 3 else None

        # 统一 wavelet 条件计算（对齐 forward() 的逻辑）
        wavelet_cond = None
        if self.wavelet_stem is not None and x_lq is not None:
            wavelet_cond = self.wavelet_stem(x_lq)

        cond = self._build_fmir_condition(
            x_lq,
            wavelet_cond=wavelet_cond if self.use_wavelet_fmir_cond else None,
        )
        dec_cond = self._build_decoder_condition(x_lq, spatial_cond=cond)
        z = self._encode_input(x)
        noise = self.noise.to(x.device)
        if noise.shape[-2:] != z.shape[-2:]:
            noise = torch.nn.functional.interpolate(
                noise, size=z.shape[-2:], mode="bilinear", align_corners=False,
            )
        z0 = self.mmse(z) + noise
        trajs = [self._decode_latent(z0.clone(), cond=dec_cond, x_lq=x_lq)[:, :, :ori_h, :ori_w]]
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
            trajs.append(self._decode_latent(z0, cond=dec_cond, x_lq=x_lq)[:, :, :ori_h, :ori_w].clone())
        return trajs

    def trajectories(self, x):
        self.to(x.device)
        x_lq = x if x.shape[1] == 3 else None
        wavelet_cond = None
        if self.wavelet_stem is not None and x_lq is not None:
            wavelet_cond = self.wavelet_stem(x_lq)
        cond = self._build_fmir_condition(
            x_lq,
            wavelet_cond=wavelet_cond if self.use_wavelet_fmir_cond else None,
        )
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