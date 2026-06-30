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
    if not torch.is_tensor(t):
        t = torch.tensor([t] if not isinstance(t, (list, tuple)) else t, dtype=torch.float32)
    if t.ndim < 1:
        t = t.unsqueeze(0)
    elif t.ndim > 1:
        t = t.squeeze()
    device = t.device
    half_dim = t_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb.detach()


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
        self.fm_cfg = fm_cfg or {}  # 存储供 load_weights 读取 ckpt_skip_prefixes
        self.K = fm_cfg.get("k_steps")
        self.latent_shape = fm_cfg.get("latent_shape")
        self.sigma_s = fm_cfg.get("sigma_s",0.1)  # 保留兼容
        self.train_noise_scale = float(fm_cfg.get("train_noise_scale", fm_cfg.get("sigma_s", 0.1)))
        self.inference_noise_scale = float(fm_cfg.get("inference_noise_scale", fm_cfg.get("sigma_s", 0.1)))
        self.dynamic_noise = fm_cfg.get("dynamic_noise",True)
        self.force_static_noise = fm_cfg.get("force_static_noise", False)
        self.noise_seed = int(fm_cfg.get("noise_seed", 2025))
        # 推理噪声模式：legacy_fixed(当前默认) / dynamic / force_static
        self.noise_mode = str(fm_cfg.get("noise_mode", "legacy_fixed"))
        self.detach_latent_path = fm_cfg.get("detach_latent_path", True)
        self.ablation_skip_fmir = bool(fm_cfg.get("ablation_skip_fmir", False))
        self.ablation_skip_mmse = bool(fm_cfg.get("ablation_skip_mmse", False))
        self.inference_mode = str(fm_cfg.get("inference_mode", "ode"))
        self.flow_infer_scale = float(fm_cfg.get("flow_infer_scale", 1.0))
        self.inference_t_val = float(fm_cfg.get("inference_t", 0.0))
        self._one_step_logged = False  # one-time debug flag
        self.t_emb_dim = fmir_cfg.get("t_emb_dim",160)
        self.dt = 1/self.K
        ode_ts = [k * self.dt for k in range(self.K)]
        print(f"[Elir init] K={self.K}, dt={self.dt:.4f}, inference_mode={self.inference_mode}, "
              f"flow_infer_scale={self.flow_infer_scale}, "
              f"inference_t={self.inference_t_val}, noise_mode={self.noise_mode}  "
              f"(from arch fm_cfg)")
        print(f"[Elir ODE] interval=[0,1], K={self.K}, dt={self.dt:.6f}, "
              f"t_schedule={ode_ts}, flow_infer_scale={self.flow_infer_scale}")
        self.fmir = get_model(fmir_cfg)
        self.mmse = get_model(mmse_cfg)
        self.enc = get_model(enc_cfg) if enc_cfg is not None else None

        # DINOv2 表示对齐 Projector (MMSE latent → DINOv2 feature space)
        self.dino_projector = None
        self.dino_spatial_proj = None
        self.sk_fusion = None
        if bool(fm_cfg.get("dino_align", False)):
            from ELIR.training.dino_align import DINOProjector
            self.dino_projector = DINOProjector(
                in_channels=int(fmir_cfg.get("params", {}).get("in_channels", 16)),
                dino_dim=768,
            )
            if bool(fm_cfg.get("sk_fusion", False)):
                from ELIR.training.dino_align import DinoSpatialProjector, SKFusion
                self.dino_spatial_proj = DinoSpatialProjector(dino_dim=768, out_channels=16)
                self.sk_fusion = SKFusion(channels=16)
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
        self.use_latent_cond = bool(fm_cfg.get("use_latent_cond", False))
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
            mode = str(self.wavelet_cfg.get("mode", "default")).lower()
            if mode == "lfg_sfg":
                from ELIR.models.wavelet_stem import LFGSFGWaveletStem
                self.wavelet_stem = LFGSFGWaveletStem(
                    in_channels=int(self.wavelet_cfg.get("in_channels", 3)),
                    base_channels=int(self.wavelet_cfg.get("base_channels", 16)),
                    use_ll=bool(self.wavelet_cfg.get("use_ll", True)),
                    use_band_attn=bool(self.wavelet_cfg.get("band_attn", False)),
                    time_cond=bool(self.wavelet_cfg.get("time_cond", False)),
                    time_dim=self.t_emb_dim,
                    gate_bias_init=float(self.wavelet_cfg.get("gate_bias_init", -4.0)),
                    hf_init=str(self.wavelet_cfg.get("hf_init", "small")),
                )
            else:
                self.wavelet_stem = WaveletStem(
                    in_channels=int(self.wavelet_cfg.get("in_channels", 3)),
                    base_channels=int(self.wavelet_cfg.get("base_channels", 16)),
                    use_ll=bool(self.wavelet_cfg.get("use_ll", True)),
                    use_band_attn=bool(self.wavelet_cfg.get("band_attn", False)),
                    time_cond=bool(self.wavelet_cfg.get("time_cond", False)),
                    time_dim=self.t_emb_dim,
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
                if bool((enc_cfg or {}).get("use_enc_gn", False)):
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
            if self.enc_trainable and mode:
                self.enc.train()
            else:
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
            # 裸 encoder 模块（如 tiny_enc），直接 __call__
            z = self.enc(x).float()
        if self.latent_norm is not None:
            z = self.latent_norm(z)
        return z

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
        """提取条件金字塔。wavelet_cond 非空 → FMIR 用频率条件，空 → Decoder 用空间条件。"""
        if x_lq is None or x_lq.shape[1] != 3:
            return None

        # FMIR 路径：优先返回 wavelet_cond（频率金字塔）
        if wavelet_cond is not None:
            return wavelet_cond

        # Decoder 路径：返回 CNN 空间金字塔
        if hasattr(self.fmir, "make_condition"):
            return self.fmir.make_condition(x_lq)
        if hasattr(self.fmir, "condition_stem"):
            cond = self.fmir.condition_stem(x_lq)
            if isinstance(cond, dict):
                if ("256" in cond) and ("128" in cond) and ("64" in cond):
                    out = {"256": cond["256"], "128": cond["128"], "64": cond["64"]}
                    if "32" in cond:
                        out["32"] = cond["32"]
                    return out
                if ("64" in cond) and ("32" in cond) and ("16" in cond):
                    return {"256": cond["64"], "128": cond["32"], "64": cond["16"]}
                return cond
            if isinstance(cond, (tuple, list)) and len(cond) == 3:
                return {"256": cond[0], "128": cond[1], "64": cond[2]}
        return None

    def _augment_cond_with_latent(self, cond, z_lq, z_mmse):
        """将 z_lq / (z_mmse - z_lq) 注入 cond dict，供 LUNet latent_cond_proj 使用。"""
        if not self.use_latent_cond:
            return cond
        if cond is None:
            cond = {}
        else:
            cond = dict(cond)
        cond["latent_lq"] = z_lq.detach()
        cond["latent_mmse_delta"] = (z_mmse - z_lq).detach()
        return cond

    @staticmethod
    def _recursive_detach(value):
        """递归 detach：兼容 Tensor、dict、list、tuple。"""
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {k: Elir._recursive_detach(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Elir._recursive_detach(v) for v in value]
        if isinstance(value, tuple):
            return tuple(Elir._recursive_detach(v) for v in value)
        return value

    def prepare_fmir_ode_condition(self, x_lq, z_lq, z_mmse, *, detach_wavelet=False):
        """统一构造 FMIR ODE condition（训练与推理共享）。

        使用 t=0 的 wavelet/time embedding，与正式 forward() 的推理起点一致。
        不依赖 CFM 随机采样的时间 t。

        Args:
            x_lq: LQ 输入 [B,3,H,W]（已 64-align）
            z_lq: Encoder(LQ) latent
            z_mmse: MMSE latent
            detach_wavelet: True 时 detach WaveletStem 输出（ODE 辅助 loss 用）
        Returns:
            cond: FMIR condition dict（含 wavelet + latent augmentation）
        """
        t0 = torch.zeros(x_lq.shape[0], device=x_lq.device, dtype=z_lq.dtype)
        t0_emb = pos_emb(t0, self.t_emb_dim).to(device=x_lq.device, dtype=z_lq.dtype)

        wavelet_cond = None
        if self.wavelet_stem is not None and self.use_wavelet_fmir_cond:
            wavelet_cond = self.wavelet_stem(x_lq, t_emb=t0_emb)
            if detach_wavelet:
                wavelet_cond = self._recursive_detach(wavelet_cond)

        cond = self._build_fmir_condition(x_lq, wavelet_cond=wavelet_cond)
        # z_lq / z_mmse 始终 detach（不更新 Encoder/MMSE 通过此路径）
        cond = self._augment_cond_with_latent(cond, z_lq.detach(), z_mmse.detach())
        return cond

    def run_fmir_ode(self, z_start, cond, *, track_grad=False):
        """共享 K-step FMIR ODE 积分。训练/推理/测试使用同一实现。

        ODE 区间 [0, 1]，K 步，dt=1/K，每步 Euler: z += dt * flow_infer_scale * v。
        t schedule: [0, 1/K, 2/K, ..., (K-1)/K]。

        Args:
            z_start: [B,C,H,W] 起点 latent
            cond: FMIR condition dict（应由 prepare_fmir_ode_condition 构造）
            track_grad: 是否保留计算图（训练=True，推理=False）
        Returns:
            z_final: ODE 终点 latent [B,C,H,W]
        """
        K = int(self.K)
        dt_val = 1.0 / K
        scale = getattr(self, "flow_infer_scale", 1.0)

        def _step(z_curr, t_val):
            t_tensor = torch.full((z_curr.shape[0], 1, 1, 1), t_val,
                                  device=z_curr.device, dtype=z_curr.dtype)
            t_emb = pos_emb(t_val, self.t_emb_dim).to(z_curr.device)
            v = self.fmir(z_curr, t_emb, cond=cond, t=t_tensor)
            return z_curr + dt_val * scale * v

        if track_grad:
            z = z_start
            t = 0.0
            for _ in range(K):
                z = _step(z, t)
                t += dt_val
            return z
        else:
            with torch.no_grad():
                z = z_start
                t = 0.0
                for _ in range(K):
                    z = _step(z, t)
                    t += dt_val
                return z

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

    def _inference_noise(self, z, device):
        """推理噪声，保留 noise_mode 语义，用 inference_noise_scale 控制强度。

        scale=0 → zeros_like（clean start）。
        scale>0 → 按 noise_mode 分发，最终噪声按 scale/sigma_s 缩放到目标强度。
        """
        scale = getattr(self, "inference_noise_scale", 0.1)
        if scale <= 0:
            return torch.zeros_like(z)

        ratio = scale / max(self.sigma_s, 1e-8)
        mode = getattr(self, "noise_mode", "legacy_fixed")

        if mode == "legacy_fixed":
            noise = self.noise.to(device=device, dtype=z.dtype)
            if noise.shape[-2:] != z.shape[-2:]:
                noise = F.interpolate(noise, size=z.shape[-2:], mode="bilinear", align_corners=False)
            return (ratio * noise).expand_as(z)

        if mode == "dynamic":
            return scale * torch.randn_like(z)

        if mode == "force_static":
            raw = self._sample_noise(z, device)  # uses sigma_s internally
            return ratio * raw

        raise ValueError(f"Unknown noise_mode: {mode}")

    def _run_latent_ode(self, x, track_grad=True):
        x_lq = x if x.shape[1] == 3 else None
        cond = self._build_fmir_condition(x_lq)
        if track_grad:
            z = self._encode_input(x)
            noise = self._sample_noise(z, x.device)
            z_mmse_clean = self.mmse(z)
            z0 = z_mmse_clean + noise
            cond = self._augment_cond_with_latent(cond, z, z_mmse_clean)
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
            z_mmse_clean = self.mmse(z)
            z0 = z_mmse_clean + noise
            cond = self._augment_cond_with_latent(cond, z, z_mmse_clean)
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

    def _reinit_skipped_modules(self, skipped_keys):
        """对因 skip_prefixes 而被跳过的模块，重新执行零初始化。

        覆盖的模块类型：
          - CondFusion1x1 (decoder_cond_fusions / fmir_cond_fusions)
          - TanhSFT / TanhSFT_NoTime (decoder SFT 模块)
          - DetailGateFusion / DetailGateFusion4Way (decoder gate 模块)
          - ConditionStem + cond_down_* (FMIR 条件金字塔)
        """
        if not skipped_keys:
            return

        # 收集受影响的顶层子模块
        affected_modules = set()
        for key in skipped_keys:
            parts = key.split(".")
            if parts[0] == "fmir" and len(parts) >= 2:
                affected_modules.add(("fmir", parts[1]))
            elif parts[0] == "dec" and len(parts) >= 2:
                affected_modules.add(("dec", parts[1]))
            elif parts[0] == "decoder_cond_fusions":
                affected_modules.add(("decoder_cond_fusions", parts[1] if len(parts) > 1 else "*"))
            elif parts[0] == "fmir_cond_fusions":
                affected_modules.add(("fmir_cond_fusions", parts[1] if len(parts) > 1 else "*"))

        reinit_count = 0
        for module_path, sub_name in sorted(affected_modules):
            if module_path == "fmir":
                self._reinit_fmir_submodule(sub_name)
                reinit_count += 1
            elif module_path == "dec":
                self._reinit_dec_submodule(sub_name)
                reinit_count += 1
            elif module_path == "decoder_cond_fusions":
                self._reinit_cond_fusions(self.decoder_cond_fusions, sub_name)
                reinit_count += 1
            elif module_path == "fmir_cond_fusions":
                self._reinit_cond_fusions(self.fmir_cond_fusions, sub_name)
                reinit_count += 1

        if reinit_count > 0:
            print(f"[load_weights] re-initialized {reinit_count} module(s) "
                  f"to zero-init / identity state (spatial conditioning reset)")

    @staticmethod
    def _reinit_cond_fusions(fusions, sub_name):
        """重新零初始化 CondFusion1x1 模块。"""
        if fusions is None:
            return
        targets = [sub_name] if sub_name != "*" else list(fusions.keys())
        for key in targets:
            mod = fusions.get(key)
            if mod is not None and hasattr(mod, "reset_parameters"):
                mod.reset_parameters()

    def _reinit_fmir_submodule(self, sub_name):
        """重新初始化 FMIR 的条件相关子模块。"""
        fmir = self.fmir
        if fmir is None:
            return
        # condition_stem / cond_to_256 / cond_down_*
        cond_modules = {
            "condition_stem": getattr(fmir, "condition_stem", None),
            "cond_to_256": getattr(fmir, "cond_to_256", None),
            "cond_down_128": getattr(fmir, "cond_down_128", None),
            "cond_down_64": getattr(fmir, "cond_down_64", None),
            "cond_down_32": getattr(fmir, "cond_down_32", None),
        }
        if sub_name == "*":
            for mod in cond_modules.values():
                if mod is not None:
                    self._reset_conv_module(mod)
        elif sub_name in cond_modules:
            mod = cond_modules[sub_name]
            if mod is not None:
                self._reset_conv_module(mod)

    def _reinit_dec_submodule(self, sub_name):
        """重新初始化 Decoder 的 SFT/Gate 子模块。"""
        dec = self.dec
        if dec is None:
            return
        # 尝试获取 SFT/gate 模块
        for attr in ["sft_32", "sft_64", "sft_128", "sft_256",
                     "gate_32", "gate_64", "gate_128", "gate_256",
                     "smooth_256"]:
            if sub_name == "*" or sub_name == attr or sub_name.startswith("sft_") or sub_name.startswith("gate_"):
                mod = getattr(dec, attr, None)
                if mod is not None and hasattr(mod, "_zero_init"):
                    mod._zero_init()
                elif mod is not None:
                    self._reset_conv_module(mod)

    @staticmethod
    def _reset_conv_module(mod):
        """递归重置模块的所有 Conv2d: Kaiming init weight, zeros bias。"""
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Sequential):
            for child in mod.children():
                Elir._reset_conv_module(child)

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

    def _filter_mismatched_keys(self, state_dict, skip_prefixes=None):
        """过滤 shape 不匹配键及用户指定的前缀键。

        Args:
            state_dict: 待加载的 state_dict
            skip_prefixes: 要跳过的 key 前缀列表（如 ['fmir.condition_stem.', 'dec.sft_']）
        """
        model_sd = self.state_dict()
        filtered = {}
        skipped = []
        prefix_skipped = []

        _prefixes = skip_prefixes or []
        for k, v in state_dict.items():
            if k not in model_sd:
                filtered[k] = v
                continue
            # 用户指定前缀过滤（最高优先级）
            if _prefixes and any(k.startswith(p) for p in _prefixes):
                prefix_skipped.append(k)
                continue
            mv = model_sd[k]
            if isinstance(v, torch.Tensor) and isinstance(mv, torch.Tensor) and v.shape != mv.shape:
                skipped.append((k, tuple(v.shape), tuple(mv.shape)))
                continue
            filtered[k] = v

        if prefix_skipped:
            print(f"[load_weights] skipped {len(prefix_skipped)} keys by prefix filter "
                  f"(prefixes={_prefixes}):")
            for k in prefix_skipped[:10]:
                print(f"  - {k}")
            if len(prefix_skipped) > 10:
                print(f"  ... and {len(prefix_skipped) - 10} more")

        if skipped:
            print("[load_weights] skipped mismatched keys:")
            for k, s1, s2 in skipped:
                print(f"  - {k}: ckpt{s1} vs model{s2}")

        return filtered, prefix_skipped

    def load_weights(self, path, skip_prefixes=None):
        """加载 checkpoint 权重。

        Args:
            path: checkpoint 路径
            skip_prefixes: 要跳过的 key 前缀列表。
                          例: ['fmir.condition_stem.', 'fmir.cond_to_256.',
                                'fmir.cond_down_128.', 'fmir.cond_down_64.',
                                'fmir.cond_down_32.']
                          被跳过的模块将保持其初始（零初始化）状态。
                          若为 None，自动从 fm_cfg.ckpt_skip_prefixes 读取。
        """
        if skip_prefixes is None:
            skip_prefixes = self.fm_cfg.get("ckpt_skip_prefixes", None) if hasattr(self, 'fm_cfg') else None
        _all_skipped = []  # 收集所有路径中被跳过的 keys

        if not path:
            return
        if not os.path.exists(os.path.expanduser(path)):
            raise FileNotFoundError(
                f"[load_weights] CHECKPOINT NOT FOUND: {path}\n"
                f"  The file does not exist. The model will use RANDOM weights."
            )
        file_size = os.path.getsize(os.path.expanduser(path))
        if file_size == 0:
            raise RuntimeError(
                f"[load_weights] CHECKPOINT IS EMPTY (0 bytes): {path}\n"
                f"  The file exists but is empty. The model would use RANDOM weights."
            )
        if file_size < 1024:  # suspiciously small
            print(
                f"\n{'='*60}\n"
                f"[load_weights] WARNING: checkpoint is suspiciously small "
                f"({file_size} bytes): {path}\n"
                f"  This may indicate a corrupted or incomplete file.\n"
                f"{'='*60}\n"
            )

        if skip_prefixes:
            print(f"[load_weights] skip_prefixes: {skip_prefixes}")

        state_dict = torch.load(path, map_location="cpu")

        # Detect empty state dict
        if not state_dict or len(state_dict) == 0:
            raise RuntimeError(
                f"[load_weights] CHECKPOINT HAS NO KEYS (empty dict): {path}\n"
                f"  The file was loaded but contains no state dict entries."
            )

        if path.endswith(".ckpt"):
            if "state_dict" in state_dict:
                sd = state_dict["state_dict"]
                if not sd or len(sd) == 0:
                    raise RuntimeError(
                        f"[load_weights] CHECKPOINT 'state_dict' IS EMPTY: {path}"
                    )
                cleaned = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
                cleaned = self._upgrade_legacy_state_dict(cleaned)
                cleaned, skipped = self._filter_mismatched_keys(cleaned, skip_prefixes=skip_prefixes)
                _all_skipped.extend(skipped)
                if not cleaned:
                    raise RuntimeError(
                        f"[load_weights] No 'model.*' keys found in checkpoint state_dict "
                        f"({len(sd)} total keys). The model will use RANDOM weights.\n"
                        f"  First 10 keys in checkpoint: {list(sd.keys())[:10]}"
                    )
                missing, unexpected = self.load_state_dict(cleaned, strict=False)
                total_params = len(self.state_dict())
                loaded_frac = 1.0 - len(missing) / max(total_params, 1)
                if loaded_frac < 0.5:
                    print(
                        f"\n{'!'*60}\n"
                        f"[load_weights] WARNING: only {loaded_frac:.0%} of model params loaded!\n"
                        f"  missing={len(missing)}, unexpected={len(unexpected)}, "
                        f"total_params={total_params}\n"
                        f"  Checkpoint may be from a different model architecture.\n"
                        f"{'!'*60}\n"
                    )
                elif len(missing) > 0 or len(unexpected) > 0:
                    print(
                        f"[load_weights] model non-strict load: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            elif "state_dict_fmir" in state_dict:  # legacy full ckpt with split submodules
                # Legacy split format: apply skip_prefixes filtering to each submodule
                sd_fmir = state_dict["state_dict_fmir"]
                if skip_prefixes:
                    sd_fmir, fmir_skipped = self._filter_mismatched_keys(
                        sd_fmir, skip_prefixes=[p.replace("fmir.", "") for p in skip_prefixes if p.startswith("fmir.")])
                    _all_skipped.extend(f"fmir.{k}" for k in fmir_skipped)
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
                # Handle decoder skip_prefixes in legacy format
                if skip_prefixes and self.dec is not None:
                    dec_skipped = [p for p in skip_prefixes if p.startswith("dec.")]
                    _all_skipped.extend(dec_skipped)
                self.collapse()
            else:
                # Fallback: try loading directly
                state_dict = self._upgrade_legacy_state_dict(state_dict)
                state_dict, skipped = self._filter_mismatched_keys(state_dict, skip_prefixes=skip_prefixes)
                _all_skipped.extend(skipped)
                if not state_dict:
                    raise RuntimeError(
                        f"[load_weights] Checkpoint has no usable keys for this model: {path}\n"
                        f"  Available keys: {list(torch.load(path, map_location='cpu').keys())[:20]}"
                    )
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                total_params = len(self.state_dict())
                loaded_frac = 1.0 - len(missing) / max(total_params, 1)
                if loaded_frac < 0.5:
                    print(
                        f"\n{'!'*60}\n"
                        f"[load_weights] WARNING: only {loaded_frac:.0%} of model params loaded!\n"
                        f"  missing={len(missing)}, unexpected={len(unexpected)}, "
                        f"total_params={total_params}\n"
                        f"{'!'*60}\n"
                    )
                elif len(missing) > 0 or len(unexpected) > 0:
                    print(
                        f"[load_weights] fallback non-strict load: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )
        else:
            state_dict = self._upgrade_legacy_state_dict(state_dict)
            state_dict, skipped = self._filter_mismatched_keys(state_dict, skip_prefixes=skip_prefixes)
            _all_skipped.extend(skipped)
            if not state_dict:
                raise RuntimeError(
                    f"[load_weights] .pth checkpoint has no usable keys: {path}"
                )
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            total_params = len(self.state_dict())
            loaded_frac = 1.0 - len(missing) / max(total_params, 1)
            if loaded_frac < 0.5:
                print(
                    f"\n{'!'*60}\n"
                    f"[load_weights] WARNING: only {loaded_frac:.0%} of model params loaded!\n"
                    f"  missing={len(missing)}, unexpected={len(unexpected)}, "
                    f"total_params={total_params}\n"
                    f"{'!'*60}\n"
                )
            elif len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"[load_weights] model non-strict load: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )

        # 对被跳过的模块执行零初始化，确保它们回到 identity / near-zero 状态
        if _all_skipped:
            self._reinit_skipped_modules(_all_skipped)

    def load_selected_modules(self, ckpt_path, load_modules, skip_modules=None):
        """从 checkpoint 选择性加载指定模块，其余模块保持随机初始化。

        典型用法：
            model.load_selected_modules(
                ckpt_path,
                load_modules=['enc', 'dec', 'wavelet', 'fmir_condition'],
                skip_modules=['mmse', 'fmir'],
            )

        支持的模块名 → (checkpoint key, model attribute):
            enc            → (state_dict_enc,        enc)
            dec            → (state_dict_dec,        dec)
            mmse           → (state_dict_mmse,       mmse)
            fmir           → (state_dict_fmir,       fmir)
            wavelet        → (state_dict_wavelet,    wavelet_stem)
            sft            → (state_dict_sft,        sft_refiner)  # fallback to dec
            condition      → (state_dict_condition,  condition_stem)
            fmir_condition → 从 state_dict_fmir 中提取 condition_stem / cond_to_256
                              / cond_down_* 键，加载到 model.fmir

        注意：
            fmir_condition 用于 decoder 依赖 FMIR condition_stem 的场景。
            如果跳过整个 fmir 但 decoder 需要 condition，必须加载 fmir_condition。

        Args:
            ckpt_path: checkpoint 路径（.ckpt 或 .pth）。
            load_modules: 要加载的模块名列表。
            skip_modules: 要跳过的模块名列表（从 load_modules 中排除）。
        """
        import torch

        skip_modules = skip_modules or []

        # 模块映射表
        MODULE_MAP = {
            "enc":       ("state_dict_enc",       "enc"),
            "dec":       ("state_dict_dec",       "dec"),
            "mmse":      ("state_dict_mmse",      "mmse"),
            "fmir":      ("state_dict_fmir",      "fmir"),
            "wavelet":   ("state_dict_wavelet",   "wavelet_stem"),
            "sft":       ("state_dict_sft",       "sft_refiner"),
            "condition": ("state_dict_condition", "condition_stem"),
            "fmir_condition": ("__fmir_condition__", "fmir"),  # special: filter from state_dict_fmir
        }

        # 解析加载列表：load_modules 过滤掉 skip_modules
        to_load = [m for m in load_modules if m not in skip_modules]

        if not to_load:
            print("[load_selected] load_modules is empty after filtering — nothing loaded.")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"\n[load_selected] Loading checkpoint: {ckpt_path}")
        print(f"[load_selected] Requested:  {load_modules}")
        print(f"[load_selected] Skip-filter: {skip_modules}")
        print(f"[load_selected] Will load:   {to_load}")

        loaded = []
        not_in_ckpt = []
        no_attr = []
        unknown = []
        report_lines = []

        for name in to_load:
            # ---- 特殊处理：fmir_condition ----
            # 从 state_dict_fmir 中提取 condition_stem / cond_to_256 /
            # cond_down_* 相关键，加载到 model.fmir。
            if name == "fmir_condition":
                fmir_module = getattr(self, "fmir", None)
                if fmir_module is None:
                    print("[load_selected] WARNING: model.fmir is None — skipping 'fmir_condition'.")
                    no_attr.append(name)
                    continue

                # 从 state_dict_fmir 提取 condition 相关键
                fmir_sd = ckpt.get("state_dict_fmir", None)
                if fmir_sd is None:
                    # 回退：从 Lightning state_dict 按前缀提取
                    if "state_dict" in ckpt:
                        sd_full = {}
                        for k, v in ckpt["state_dict"].items():
                            clean_k = k[len("model."):] if k.startswith("model.") else k
                            sd_full[clean_k] = v
                        fmir_sd = {k[len("fmir."):]: v for k, v in sd_full.items() if k.startswith("fmir.")}
                if fmir_sd is None or not fmir_sd:
                    print("[load_selected] WARNING: state_dict_fmir not found — skipping 'fmir_condition'.")
                    not_in_ckpt.append(name)
                    continue

                COND_PREFIXES = (
                    "condition_stem.", "cond_to_256.", "cond_down_128.",
                    "cond_down_64.", "cond_down_32.",
                )
                cond_sd = {k: v for k, v in fmir_sd.items()
                           if any(k.startswith(p) for p in COND_PREFIXES)}
                if not cond_sd:
                    print("[load_selected] WARNING: no condition keys in state_dict_fmir — skipping 'fmir_condition'.")
                    not_in_ckpt.append(name)
                    continue

                missing, unexpected = fmir_module.load_state_dict(cond_sd, strict=False)
                n_cond_params = sum(v.numel() for v in cond_sd.values())
                loaded.append(name)
                report_lines.append(
                    f"  [fmir_condition] → model.fmir (condition only) | "
                    f"params: {n_cond_params:,} | "
                    f"missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}"
                )
                if len(missing) > 0:
                    report_lines.append(f"         missing sample: {missing[:5]}")
                if len(unexpected) > 0:
                    report_lines.append(f"         unexpected sample: {unexpected[:5]}")
                continue

            if name not in MODULE_MAP:
                print(f"[load_selected] WARNING: unknown module '{name}' — skipping.")
                unknown.append(name)
                continue

            ckpt_key, attr = MODULE_MAP[name]
            module = getattr(self, attr, None)

            if module is None:
                # Fallback: sft → dec
                if name == "sft" and hasattr(self, "dec") and self.dec is not None:
                    module = self.dec
                    attr = "dec (fallback from sft)"
                else:
                    print(f"[load_selected] WARNING: model.{attr} is None — skipping '{name}'.")
                    no_attr.append(name)
                    continue

            # 尝试从 checkpoint 获取 state dict
            if ckpt_key in ckpt:
                sd = ckpt[ckpt_key]
            elif "state_dict" in ckpt:
                # 回退：从 Lightning 全量 state_dict 按前缀过滤
                prefix_map = {
                    "enc": "enc.",
                    "dec": "dec.",
                    "mmse": "mmse.",
                    "fmir": "fmir.",
                    "wavelet_stem": "wavelet_stem.",
                    "sft_refiner": "sft_refiner.",
                    "condition_stem": "condition_stem.",
                }
                prefix = prefix_map.get(attr.split(" ")[0], None)  # handle fallback case
                if prefix is None:
                    # try attr directly
                    prefix = prefix_map.get(attr, None)
                if prefix is None:
                    print(f"[load_selected] WARNING: no prefix mapping for attr='{attr}' — skipping '{name}'.")
                    not_in_ckpt.append(name)
                    continue

                # 去掉 "model." 前缀（Lightning 格式）
                sd_full = {}
                for k, v in ckpt["state_dict"].items():
                    clean_k = k[len("model."):] if k.startswith("model.") else k
                    sd_full[clean_k] = v

                sd = {k[len(prefix):]: v for k, v in sd_full.items() if k.startswith(prefix)}
                if not sd:
                    print(f"[load_selected] WARNING: no keys with prefix '{prefix}' in state_dict — skipping '{name}'.")
                    not_in_ckpt.append(name)
                    continue
            else:
                print(f"[load_selected] WARNING: '{ckpt_key}' not found in checkpoint — skipping '{name}'.")
                not_in_ckpt.append(name)
                continue

            # 升级旧版 wavelet（9ch→12ch）
            if name == "wavelet":
                sd = self._upgrade_legacy_state_dict(sd)

            # 过滤 shape 不匹配的键
            sd, _ = self._filter_mismatched_keys(sd)

            # 加载
            missing, unexpected = module.load_state_dict(sd, strict=False)
            n_params = sum(p.numel() for p in module.parameters())
            loaded.append(name)
            report_lines.append(
                f"  [{name}] → model.{attr} | params: {n_params:,} | "
                f"missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}"
            )
            if len(missing) > 0:
                report_lines.append(f"         missing sample: {missing[:5]}")
            if len(unexpected) > 0:
                report_lines.append(f"         unexpected sample: {unexpected[:5]}")

        # ---- 汇总报告 ----
        print("\n" + "=" * 64)
        print("[load_selected] Selective Checkpoint Loading Report")
        print("=" * 64)
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Loaded modules:   {loaded if loaded else '(none)'}")
        if unknown:
            print(f"  Unknown names:    {unknown}")
        if not_in_ckpt:
            print(f"  Not in ckpt:      {not_in_ckpt}")
        if no_attr:
            print(f"  No model attr:    {no_attr}")
        if report_lines:
            print("-" * 64)
            for line in report_lines:
                print(line)
        print("=" * 64)

        # ---- 确认 skipped 模块仍为随机初始化 ----
        for name in skip_modules:
            if name in MODULE_MAP:
                _, attr = MODULE_MAP[name]
                module = getattr(self, attr, None)
                if module is not None:
                    print(f"[load_selected] Verified: '{name}' (model.{attr}) — "
                          f"NOT loaded, kept random init.")
        print()

    def _one_step_flow_decode(self, x_lq, scale=None, t_val=None):
        """Calibrated one-step flow inference — forward / diagnostics / sweep 共用。

        Args:
            x_lq: low-quality input [B,3,H,W]（需已 64-aligned padded）
            scale: flow calibration scale（默认 self.flow_infer_scale）
            t_val: inference time（默认 self.inference_t_val）

        Returns:
            (pred_pixel, z_final, v_pred)
        """
        scale = self.flow_infer_scale if scale is None else scale
        t_val = self.inference_t_val if t_val is None else t_val

        # ---- wavelet / FMIR / decoder 条件（与 scale sweep 完全一致） ----
        wavelet_cond = None
        if self.wavelet_stem is not None:
            t_emb_init = pos_emb(torch.zeros(x_lq.shape[0]), self.t_emb_dim).to(x_lq.device)
            wavelet_cond = self.wavelet_stem(x_lq, t_emb=t_emb_init)

        fmir_cond = self._build_fmir_condition(x_lq, wavelet_cond=wavelet_cond)
        decoder_cond = self._build_fmir_condition(x_lq, wavelet_cond=None)

        # ---- latent path ----
        z = self._encode_input(x_lq)
        z_mmse = self.mmse(z)
        fmir_cond = self._augment_cond_with_latent(fmir_cond, z, z_mmse)

        t_vec = torch.full((z_mmse.shape[0],), t_val, device=z_mmse.device, dtype=z_mmse.dtype)
        v_pred = self.fmir(z_mmse, pos_emb(t_vec, self.t_emb_dim).to(z_mmse.device),
                           cond=fmir_cond, t=t_vec[:, None, None, None])
        z_final = z_mmse + scale * v_pred

        # ---- decode（双路分离：Decoder 纯空间条件，不接收 wavelet） ----
        pred = self._decode_latent(z_final, cond=decoder_cond, x_lq=x_lq,
                                   wavelet_cond=None)
        return pred, z_final, v_pred

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

        # ---- 消融：跳过 FMIR ODE，但保留 MMSE ----
        if self.ablation_skip_fmir:
            with torch.no_grad():
                z = self._encode_input(x_safe)
                z0 = self.mmse(z)
            # build decoder cond (needed even for ablation)
            _dec_cond = self._build_fmir_condition(x_lq, wavelet_cond=None)
            y_padded = self._decode_latent(z0, cond=_dec_cond, x_lq=x_lq, wavelet_cond=None)
            y_restored = y_padded[:, :, :ori_h, :ori_w]
            return y_restored

        # ---- 消融：跳过 MMSE ----
        if self.ablation_skip_mmse:
            z = self._encode_input(x_safe)
            noise = self._inference_noise(z, x_safe.device)
            z0 = z + noise
            _fm_cond = self._build_fmir_condition(x_lq, wavelet_cond=None)
            _fm_cond = self._augment_cond_with_latent(_fm_cond, z, z)
            dt = 0
            for _ in range(self.K):
                t_tensor = torch.full((z0.shape[0], 1, 1, 1), dt, device=x_safe.device, dtype=z0.dtype)
                z0 = z0 + self.dt * self.fmir(
                    z0, pos_emb(dt, self.t_emb_dim).to(x_safe.device),
                    cond=_fm_cond, t=t_tensor)
                dt += self.dt
            _dec_cond = self._build_fmir_condition(x_lq, wavelet_cond=None)
            y_padded = self._decode_latent(z0, cond=_dec_cond, x_lq=x_lq, wavelet_cond=None)
            y_restored = y_padded[:, :, :ori_h, :ori_w]
            return y_restored

        # ---- one_step shortcut: 共享函数，与 scale sweep 完全一致 ----
        if self.inference_mode == "one_step":
            track_grad = (
                self.training
                and torch.is_grad_enabled()
                and not self.detach_latent_path
                and not self._latent_path_frozen()
            )
            with torch.set_grad_enabled(track_grad):
                pred_padded, _, _ = self._one_step_flow_decode(x_lq)
            y_restored = pred_padded[:, :, :ori_h, :ori_w]
            if not self._one_step_logged:
                print(f"[forward one_step] scale={self.flow_infer_scale}, t={self.inference_t_val}")
                self._one_step_logged = True
            return y_restored

        # Decoder 条件：纯空间金字塔（CNN ConditionStem, 无 wavelet 融合）
        decoder_cond = self._build_fmir_condition(x_lq, wavelet_cond=None)

        # Wavelet 可视化（独立路径，不影响 ODE）
        if self.wavelet_stem is not None and self.wavelet_visualize and x_lq is not None:
            t_emb_init = pos_emb(torch.zeros(x_lq.shape[0]), self.t_emb_dim).to(x_lq.device)
            _, wavelet_debug = self.wavelet_stem(x_lq, return_debug=True, t_emb=t_emb_init)
            self._maybe_save_wavelet_debug(x_lq, wavelet_debug)

        # ---- ODE 推理（one_step / ablation 已在前面 early-return） ----
        # FMIR ODE condition 由 prepare_fmir_ode_condition 统一构建
        z = self._encode_input(x_safe)
        noise = self._inference_noise(z, x_safe.device)
        z_mmse_clean = self.mmse(z)
        z0 = z_mmse_clean + noise
        fmir_cond = self.prepare_fmir_ode_condition(x_safe, z, z_mmse_clean)
        track_grad = (
            self.training
            and torch.is_grad_enabled()
            and not self.detach_latent_path
            and not self._latent_path_frozen()
        )
        z0 = self.run_fmir_ode(z0, fmir_cond, track_grad=track_grad)
        if not track_grad:
            z0 = z0.detach()
        # 双路分离：Decoder 纯空间条件，wavelet_cond 仅给 FMIR
        y_padded = self._decode_latent(z0, cond=decoder_cond, x_lq=x_lq, wavelet_cond=None)
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
        noise = self._inference_noise(z, x.device)
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
        z0 = self.mmse(z) + self._inference_noise(z, x.device)
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
