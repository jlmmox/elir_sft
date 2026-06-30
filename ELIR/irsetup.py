from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
import pytorch_lightning as L
import torch
from ELIR.metrics import MetricEval
from ELIR.training.losses import get_loss, e2e_gan_loss
from torchvision.utils import save_image
from ELIR.training.ema_timm import ModelEMA
from ELIR.training.perceptual import create_perceptual_loss
from ELIR.models.gan import NLayerDiscriminator
from ELIR.models.elir import pos_emb
from ELIR.training.dino_align import DINOv2Encoder, DinoSpatialProjector, SKFusion
import os
import torch.nn.functional as F
from ELIR.utils import ImageSpliterTh
import math



class IRSetup(L.LightningModule):
    # Normal-mode CSV 白名单（log_cfg.level=normal 时仅记录这些字段）
    _NORMAL_CSV_FIELDS = frozenset({
        "epoch", "global_step", "lr", "train_loss",
        "loss_fm", "loss_mmse_char", "loss_pixel_total",
        "loss_gan_g", "loss_gan_d",
        "warmup_ratio",
        "encoder_grad_norm", "mmse_grad_norm", "fmir_grad_norm",
        "decoder_grad_norm", "wavelet_grad_norm",
        "val_psnr", "psnr", "ssim", "fid",
        "diag_sft_hq_psnr_full", "diag_sft_mmse_psnr_full",
        "flow_gain_psnr",
        "latent_delta_cosine", "latent_gain_charb",
        "gpu_peak_allocated_gb",
    })

    def __init__(self, model, fm_cfg={}, optimizer=None, scheduler=None, tmodel=None,
                 ema_decay=None, eval_cfg=None, run_dir=None, save_images=True,
                 log_cfg=None):
        super().__init__()
        self.model = model
        self.fm_cfg = fm_cfg

        # ---- 日志配置（尽早解析，以便后续 print 使用） ----
        _lc = log_cfg or {}
        self._log_level = str(_lc.get("level", "normal")).strip().lower()
        self._log_is_debug = (self._log_level == "debug")
        self._log_train_interval = int(_lc.get("train_log_interval", 100))
        self._log_grad_interval = int(_lc.get("grad_log_interval", 500))
        self._log_module_grad_norms = bool(_lc.get("log_module_grad_norms", True))
        self._log_memory = bool(_lc.get("log_memory", False))
        self._log_debug_fmir_params = bool(_lc.get("debug_fmir_params", False))
        self._log_debug_fmir_updates = bool(_lc.get("debug_fmir_updates", False))
        self._log_debug_cfm_details = bool(_lc.get("debug_cfm_details", False))
        self._log_debug_val_graph = bool(_lc.get("debug_validation_graph", False))
        self._log_print_param_names = bool(_lc.get("print_parameter_names", False))
        self._last_grad_log_step = -1
        self._k_steps_checked = False

        # ---- 将训练 fm_cfg 中的推理参数同步到模型 ----
        # 模型内部通过 arch_cfg.params.fm_cfg 初始化，可能缺少 inference_* 字段。
        # 此处用外层 fm_cfg 覆盖，确保 forward() / validation 读到正确值。
        _overrides = []
        for _key, _attr in [
            ("inference_mode", "inference_mode"),
            ("flow_infer_scale", "flow_infer_scale"),
            ("inference_t", "inference_t_val"),
        ]:
            if _key in fm_cfg:
                _old = getattr(self.model, _attr)
                _new = fm_cfg[_key] if _attr == "inference_mode" else float(fm_cfg[_key])
                setattr(self.model, _attr, _new)
                _overrides.append(f"{_attr}={_old}→{_new}")

        # k_steps: 外层 fm_cfg 覆盖模型内层 arch.fm_cfg
        if "k_steps" in fm_cfg:
            _old_k = getattr(self.model, "K", None)
            _new_k = int(fm_cfg["k_steps"])
            self.model.K = _new_k
            self.model.dt = 1.0 / max(_new_k, 1)
            _overrides.append(f"K={_old_k}→{_new_k}")

        # noise_mode: 外层 fm_cfg 覆盖模型内层
        if "noise_mode" in fm_cfg:
            _old_nm = getattr(self.model, "noise_mode", "legacy_fixed")
            _new_nm = str(fm_cfg["noise_mode"])
            self.model.noise_mode = _new_nm
            _overrides.append(f"noise_mode={_old_nm}→{_new_nm}")

        _im = getattr(self.model, "inference_mode", "ode")
        _fs = getattr(self.model, "flow_infer_scale", 1.0)
        _it = getattr(self.model, "inference_t_val", 0.0)
        _K  = getattr(self.model, "K", None)
        _nm = getattr(self.model, "noise_mode", "legacy_fixed")
        if _overrides:
            print(f"[config sync] K={_K}, noise_mode={_nm}, inference_mode={_im}, "
                  f"scale={_fs}, t={_it}  "
                  f"(overrides: {', '.join(_overrides)})")
        elif self._log_is_debug:
            print(f"[config sync] K={_K}, noise_mode={_nm}, inference_mode={_im}, "
                  f"scale={_fs}, t={_it}  "
                  f"(no overrides)")

        self.eval_cfg = eval_cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = eval_cfg.get("metrics",[])
        self.tmodel = tmodel
        if torch.cuda.is_available():
            self.acc_device = torch.device(torch.cuda.current_device())
        elif torch.backends.mps.is_available():
            self.acc_device = torch.device('mps')
        else:
            self.acc_device = torch.device('cpu')
        # test_y_channel: True=Y通道(默认/去雾/低光), False=RGB三通道(去雨)
        self._test_y_channel = bool(eval_cfg.get("test_y_channel", True))
        self.metric_evals = [MetricEval(metric, self.acc_device, run_dir,
                                        test_y_channel=self._test_y_channel)
                             for metric in self.metrics]
        self.train_loss = []
        self.samples_dir = None
        self.max_save_images = 0
        if run_dir and save_images:
            self.samples_dir = os.path.join(run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)
            self.samples = []
            self.max_save_images = int(self.eval_cfg.get("max_save_images", 4))
        self._runtime_space_logged = False
        self._val_debug_logged = False       # validation 第一个 batch 打印 grad/training 状态
        self._mem_tracked_epoch1 = False      # epoch1 训练前打印显存
        self._mem_tracked_val_before = False   # validation 前打印显存
        self._mem_tracked_val_after = False    # validation 后打印显存
        self._mem_tracked_epoch2 = False       # epoch2 首批训练前打印显存
        self._diag_sft_hq_psnr_values = []
        self._diag_sft_mmse_psnr_values = []
        self._latent_diag_values = {
            "latent_charb_mmse_to_hq": [],
            "latent_charb_fmir_to_hq": [],
            "latent_gain_charb": [],
            "latent_mse_mmse_to_hq": [],
            "latent_mse_fmir_to_hq": [],
            "latent_gain_mse": [],
            "latent_fmir_delta_norm": [],
            "latent_target_delta_norm": [],
            "latent_delta_norm_ratio": [],
            "latent_delta_cosine": [],
        }
        self._latent_diag_K_logged = False
        self._fmir_scale_sweep_values = {}  # lazy-init per key

        # ---- 端到端 GAN 训练组件 ----
        self.is_e2e = (str(fm_cfg.get("method", "")).strip() == "e2e_gan")
        self.discriminator = None
        self.perceptual_fn = None
        self.d_optimizer = None
        self.dino_encoder = None
        self.dino_spatial_proj = None
        self.sk_fusion = None
        self._e2e_log_cache = {}
        self._grad_accum = 1
        self._accum_step = 0
        self._lr_base_g = None
        self._lr_base_d = None
        self._lr_min_ratio = 0.01
        if self.is_e2e:
            self.automatic_optimization = False
            self._grad_accum = int(fm_cfg.get("grad_accum", 4))
            self._lr_min_ratio = float(fm_cfg.get("lr_min_ratio", 0.01))
            self.discriminator = None
            if float(fm_cfg.get("lambda_gan", 0.01)) > 0:
                self.discriminator = NLayerDiscriminator(
                    in_channels=3,
                    base_channels=int(fm_cfg.get("d_base_channels", 64)),
                    n_layers=int(fm_cfg.get("d_n_layers", 3)),
                ).to(self.acc_device)
                self.d_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=float(fm_cfg.get("d_lr", 0.0001)),
                    betas=(0.5, 0.999),
                )
            else:
                self.d_optimizer = None

            self.perceptual_fn = None
            if float(fm_cfg.get("lambda_perc", 0.0)) > 0:
                self.perceptual_fn, _ = create_perceptual_loss(
                    device=str(self.acc_device),
                    prefer_lpips=bool(fm_cfg.get("prefer_lpips", True)),
                )
            # DINOv2: dino_align 或 dino_perceptual 都需要编码器
            if bool(fm_cfg.get("dino_align", False)) or bool(fm_cfg.get("dino_perceptual", False)):
                self.dino_encoder = DINOv2Encoder(device=str(self.acc_device))
                self.dino_encoder.eval()
                for p in self.dino_encoder.parameters():
                    p.requires_grad = False
            # SK Fusion: TAESD(HQ) + DINOv2(HQ) 双路融合对齐
            if bool(fm_cfg.get("sk_fusion", False)):
                if self.dino_encoder is None:
                    self.dino_encoder = DINOv2Encoder(device=str(self.acc_device))
                self.dino_spatial_proj = getattr(self.model, "dino_spatial_proj", None)
                self.sk_fusion = getattr(self.model, "sk_fusion", None)
                if self.dino_spatial_proj is None or self.sk_fusion is None:
                    self.dino_spatial_proj = DinoSpatialProjector(dino_dim=768, out_channels=16)
                    self.sk_fusion = SKFusion(channels=16)
                    self.model.dino_spatial_proj = self.dino_spatial_proj
                    self.model.sk_fusion = self.sk_fusion
                    self.optimizer.add_param_group({"params": self.dino_spatial_proj.parameters()})
                    self.optimizer.add_param_group({"params": self.sk_fusion.parameters()})
                self.dino_spatial_proj = self.dino_spatial_proj.to(self.acc_device)
                self.sk_fusion = self.sk_fusion.to(self.acc_device)

        # ---- EMA：必须在所有动态模块挂载完成后创建，确保 deepcopy 覆盖完整模型 ----
        self.ema = None
        self._ema_decay = None
        if ema_decay:
            self.ema = ModelEMA(model, device=self.acc_device, decay=ema_decay)
            self._ema_decay = float(ema_decay)

        # ---- 打印模块参数统计（启动时一次） ----
        self._print_param_stats()

        # ---- 模型指纹与等价性检查 ----
        self._metric_model_fingerprinted = False
        self._equivalence_checked = False
        self._debug_eval_equivalence = bool(eval_cfg.get("debug_eval_equivalence", False))

    # ------------------------------------------------------------------
    # 统一 metric model 访问
    # ------------------------------------------------------------------
    def get_metric_model(self):
        """返回当前用于指标计算的模型实例及其来源标识。

        统一入口：训练验证、独立 eval、latent 诊断、正式指标
        都必须调用此方法，禁止各自实现不同逻辑。
        """
        if getattr(self, "ema", None) is not None:
            return self.ema.model, "EMA"
        return self.model, "RAW"

    @staticmethod
    def sync_ode_runtime(model, k_steps, flow_scale, noise_scale):
        """同步 ODE 运行时配置到模型实例（RAW 或 EMA）。

        必须在 checkpoint 加载和 EMA 恢复之后调用，
        确保 RAW 和 EMA 使用相同的 K/dt/scale/noise。
        """
        model.K = int(k_steps)
        model.dt = 1.0 / max(model.K, 1)
        model.flow_infer_scale = float(flow_scale)
        model.inference_noise_scale = float(noise_scale)

    def _print_metric_model_fingerprint(self):
        """打印 metric model 指纹（仅首次调用）。"""
        if self._metric_model_fingerprinted:
            return
        self._metric_model_fingerprinted = True

        metric_model, metric_model_name = self.get_metric_model()

        head = metric_model.fmir.final_proj.weight.detach()

        print(
            "[metric model fingerprint] "
            f"source={metric_model_name}, "
            f"model_id={id(metric_model)}, "
            f"K={metric_model.K}, "
            f"dt={metric_model.dt}, "
            f"flow_infer_scale={metric_model.flow_infer_scale}, "
            f"inference_noise_scale={metric_model.inference_noise_scale}, "
            f"head_mean={head.float().mean().item():.8e}, "
            f"head_abs_mean={head.float().abs().mean().item():.8e}, "
            f"head_norm={head.float().norm().item():.8e}"
        )

        if getattr(self, "ema", None) is not None:
            raw = self.model.fmir.final_proj.weight.detach().float()
            ema = self.ema.model.fmir.final_proj.weight.detach().float()

            print(
                "[RAW EMA comparison] "
                f"raw_abs_mean={raw.abs().mean().item():.8e}, "
                f"ema_abs_mean={ema.abs().mean().item():.8e}, "
                f"mean_abs_diff={(raw - ema).abs().mean().item():.8e}, "
                f"max_abs_diff={(raw - ema).abs().max().item():.8e}"
            )

    # ------------------------------------------------------------------
    # 显存快照（OOM 诊断）
    # ------------------------------------------------------------------
    @staticmethod
    def _mem_snapshot(label=""):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM:{label}] allocated={alloc:.3f}GiB  reserved={reserved:.3f}GiB  max_allocated={max_alloc:.3f}GiB")

    # ------------------------------------------------------------------
    # 参数统计（启动时打印一次）
    # ------------------------------------------------------------------
    def _print_param_stats(self):
        modules = [
            ("enc", self.model.enc),
            ("mmse", self.model.mmse),
            ("fmir", self.model.fmir),
            ("dec", self.model.dec),
            ("wavelet_stem", self.model.wavelet_stem),
        ]
        if self._log_is_debug:
            modules += [
                ("decoder_cond_fusions", getattr(self.model, "decoder_cond_fusions", None)),
                ("fmir_cond_fusions", getattr(self.model, "fmir_cond_fusions", None)),
                ("latent_norm", getattr(self.model, "latent_norm", None)),
            ]

        lines = ["[params] Module parameter counts (M = sum(p.numel())/1e6):"]
        total_all, total_t = 0, 0
        for name, module in modules:
            if module is None:
                continue
            n_total = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_all += n_total
            total_t += n_trainable
            flag = " [TRAINABLE]" if n_trainable > 0 else " [frozen]"
            lines.append(f"  {name:>20s}: {n_total/1e6:8.2f}M total, {n_trainable/1e6:8.2f}M trainable{flag}")
        lines.append(f"  {'TOTAL':>20s}: {total_all/1e6:8.2f}M total, {total_t/1e6:8.2f}M trainable")

        if self._log_is_debug and self._log_print_param_names:
            lines.append("[params] Trainable parameter names:")
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    lines.append(f"  {n}")

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # 模块梯度范数
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_module_grad_norms(self):
        if not self._log_module_grad_norms:
            return {}
        prefixes = {
            "encoder_grad_norm": "enc.",
            "mmse_grad_norm": "mmse.",
            "fmir_grad_norm": "fmir.",
            "decoder_grad_norm": "dec.",
            "wavelet_grad_norm": "wavelet_stem.",
        }
        norms = {}
        for key, prefix in prefixes.items():
            sq = 0.0
            count = 0
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if not n.startswith(prefix):
                    continue
                if p.grad is not None:
                    sq += p.grad.detach().norm(2).item() ** 2
                    count += 1
            norms[key] = sq ** 0.5 if count > 0 else 0.0
        return norms

    # ------------------------------------------------------------------
    # 安全守卫
    # ------------------------------------------------------------------
    @staticmethod
    def _check_loss_valid(loss, step):
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"[SAFETY] NaN/Inf loss at step={step}, loss={loss.item():.6f}")

    @staticmethod
    def _check_grad_valid(model, step):
        for n, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    raise RuntimeError(f"[SAFETY] NaN/Inf grad in {n} at step={step}")

    def _check_k_steps(self):
        outer_k = int(self.fm_cfg.get("k_steps", -1))
        model_k = int(getattr(self.model, "K", -1))
        if outer_k != model_k:
            raise RuntimeError(f"[SAFETY] k_steps mismatch: fm_cfg.k_steps={outer_k} vs model.K={model_k}")

    def _check_grad_flow(self, norms):
        import logging
        fm = self.fm_cfg
        # MMSE guard: 仅当损失权重 >0 且 MMSE 被设为 trainable 时才检查
        mmse_trainable = bool((self.model.mmse_cfg or {}).get("trainable", False)) if hasattr(self.model, "mmse_cfg") else True
        if float(fm.get("lambda_mmse_char", 0)) > 0 and mmse_trainable and norms.get("mmse_grad_norm", -1.0) == 0.0:
            logging.error("[SAFETY] lambda_mmse_char>0 and mmse trainable but mmse_grad_norm=0 — MMSE not receiving gradients!")
        # Encoder guard: 仅当 enc_trainable 且存在通向 Encoder 的损失路径时才检查
        if getattr(self.model, "enc_trainable", False) and float(fm.get("lambda_mmse_char", 0)) > 0 \
           and norms.get("encoder_grad_norm", -1.0) == 0.0:
            logging.warning("[SAFETY] enc_trainable=True and lambda_mmse_char>0 but encoder_grad_norm=0 — Encoder not receiving gradients!")

    def _infer_runtime_space(self, x_lq):
        method = str(self.fm_cfg.get("method", ""))
        has_enc = hasattr(self.model, "enc") and (self.model.enc is not None)
        has_dec = hasattr(self.model, "dec") and (self.model.dec is not None)

        # Priority by actual model structure + current input channels.
        if x_lq is not None and x_lq.ndim == 4 and x_lq.shape[1] in [4, 16]:
            return "latent_cache", 0.0
        if has_enc:
            return "latent_vae", 0.0
        if (not has_enc) and (not has_dec):
            return "pixel_space", 1.0

        # Fallback using configured loss naming.
        if "pixel" in method and "ablation" in method:
            return "pixel_space", 1.0
        if "pixel" in method:
            return "pixel_space", 1.0
        return "unknown", 0.0

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        # EMA is handled in training_step for e2e mode.
        if not self.is_e2e:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
            if self.ema:
                self.ema.update(self.model)

    def _step_cosine_lr(self, opt_g, opt_d):
        """Cosine annealing：初始 lr 余弦衰减到 lr_min_ratio 倍。"""
        if self._lr_base_g is None:
            self._lr_base_g = float(opt_g.param_groups[0]["lr"])
            self._lr_base_d = float(opt_d.param_groups[0]["lr"]) if opt_d is not None else 0.0
        max_steps = getattr(self.trainer, "max_steps", 300000) or 300000
        progress = min(1.0, self.global_step / max(max_steps, 1))
        lr_mult = self._lr_min_ratio + 0.5 * (1.0 - self._lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
        opt_g.param_groups[0]["lr"] = self._lr_base_g * lr_mult
        if opt_d is not None:
            opt_d.param_groups[0]["lr"] = self._lr_base_d * lr_mult

    def _training_step_normal(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        fm_cfg_runtime = dict(self.fm_cfg)
        fm_cfg_runtime["global_step"] = int(self.global_step)
        loss = get_loss(self.model, x_hq, x_lq, fm_cfg_runtime, self.tmodel)
        self._check_loss_valid(loss, self.global_step)
        self.train_loss.append(loss)
        if batch_idx % self._log_train_interval == 0:
            self.log("train_loss", torch.mean(torch.Tensor(self.train_loss)).item(), logger=True, prog_bar=True, on_step=True)
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.log("lr", self.optimizer.param_groups[0]["lr"], logger=True, prog_bar=False, on_step=True)
            self.train_loss.clear()
        return loss

    def _training_step_e2e(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        self._has_disc = self.discriminator is not None
        opts = self.optimizers()
        if self._has_disc:
            opt_g, opt_d = opts[0], opts[1]
        else:
            opt_g, opt_d = opts, None

        fm_cfg_runtime = dict(self.fm_cfg)
        fm_cfg_runtime["global_step"] = int(self.global_step)

        g_loss, d_loss, metrics = e2e_gan_loss(
            self.model, x_hq, x_lq, fm_cfg_runtime,
            discriminator=self.discriminator,
            perceptual_fn=self.perceptual_fn,
            dino_encoder=self.dino_encoder,
            dino_spatial_proj=self.dino_spatial_proj,
            sk_fusion=self.sk_fusion,
            step=int(self.global_step),
            tmodel=self.tmodel,
            global_step=int(self.global_step),
        )

        # ---- 安全校验：loss NaN/Inf ----
        self._check_loss_valid(g_loss, self.global_step)

        # 手动梯度累积：损失按累积步数缩放
        accum = max(1, self._grad_accum)
        g_loss_scaled = g_loss / accum

        self.manual_backward(g_loss_scaled)
        if self._has_disc and d_loss.requires_grad:
            self.manual_backward(d_loss / accum)

        # ---- FMIR 梯度诊断（仅 debug 模式打印详细参数名） ----
        _fmir_grad_diag = {}

        # 参数计数（一次性 log，仅 debug 模式打印全量 param 名）
        if not getattr(self, "_fmir_param_stats_logged", False):
            if self._log_is_debug and self._log_debug_fmir_params:
                _fmir_all = list(self.model.fmir.named_parameters())
                _fmir_trainable = [(n, p) for n, p in _fmir_all if p.requires_grad]
                _fp_names = [n for n, _ in _fmir_all if n.startswith("final_proj.")]
                _fp_trainable = [n for n in _fp_names if any(n == tn for tn, _ in _fmir_trainable)]
                _head_prefixes = ("latent_cond_proj.", "first_proj.", "up_blocks.3.", "up_blocks.2.",
                                  "sft_up.3.", "sft_up.2.", "final_block.", "final_proj.")
                _head_trainable = [n for n, _ in _fmir_trainable
                                   if any(n.startswith(p) for p in _head_prefixes)]
                print("[fmir_diag] FMIR parameter breakdown:")
                print(f"  total params:                {len(_fmir_all)}")
                print(f"  trainable params:            {len(_fmir_trainable)}")
                print(f"  head-block trainable:        {len(_head_trainable)}")
                print(f"  final_proj trainable:        {len(_fp_trainable)}")
                if self._log_print_param_names and _fmir_trainable:
                    print(f"  all trainable param names:")
                    for n, _ in _fmir_trainable:
                        print(f"    {n}")
            self._fmir_param_stats_logged = True

        # FMIR grad norm（始终计算，CSV 需要）
        _fmir_trainable_now = [(n, p) for n, p in self.model.fmir.named_parameters() if p.requires_grad]
        if _fmir_trainable_now:
            _gnorms = [p.grad.detach().norm(2).item() for _, p in _fmir_trainable_now if p.grad is not None]
            _fmir_grad_diag["fmir_grad_norm"] = sum(_gnorms) / max(len(_gnorms), 1) if _gnorms else 0.0
        else:
            _fmir_grad_diag["fmir_grad_norm"] = 0.0

        # ---- 详细的 sub-module / head / update 诊断（仅 debug 模式） ----
        if self._log_is_debug and self._log_debug_fmir_updates:
            _fmir_grad_diag["fmir_num_trainable_params"] = float(len(_fmir_trainable_now))
            _gnorms, _gmaxs, _gcnt = [], [], 0
            for _n, _p in _fmir_trainable_now:
                if _p.grad is not None:
                    _gnorms.append(_p.grad.detach().norm(2).item())
                    _gmaxs.append(_p.grad.detach().abs().max().item())
                    _gcnt += 1
            _fmir_grad_diag["fmir_grad_max"] = max(_gmaxs) if _gmaxs else 0.0
            _fmir_grad_diag["fmir_num_grad_params"] = float(_gcnt)
            for _sub_key, _sub_prefix in [
                ("latent_cond_proj", "latent_cond_proj."),
                ("first_proj", "first_proj."),
                ("final_proj", "final_proj."),
            ]:
                _sp = [(n, p) for n, p in self.model.fmir.named_parameters()
                       if n.startswith(_sub_prefix) and p.requires_grad]
                if _sp:
                    _sg = [p.grad.detach().norm(2).item() for _, p in _sp if p.grad is not None]
                    _fmir_grad_diag[f"{_sub_key}_grad_norm"] = sum(_sg) / max(len(_sg), 1) if _sg else 0.0
                else:
                    _fmir_grad_diag[f"{_sub_key}_grad_norm"] = -1.0
            _head_prefixes = ("latent_cond_proj.", "first_proj.", "up_blocks.3.", "up_blocks.2.",
                              "sft_up.3.", "sft_up.2.", "final_block.", "final_proj.")
            _head_params_now = [(n, p) for n, p in _fmir_trainable_now
                                if any(n.startswith(pr) for pr in _head_prefixes)]
            if _head_params_now:
                _hg = [p.grad.detach().norm(2).item() for _, p in _head_params_now if p.grad is not None]
                _hgm = [p.grad.detach().abs().max().item() for _, p in _head_params_now if p.grad is not None]
                _fmir_grad_diag["fmir_head_grad_norm"] = sum(_hg) / max(len(_hg), 1) if _hg else 0.0
                _fmir_grad_diag["fmir_head_grad_max"] = max(_hgm) if _hgm else 0.0
                _fmir_grad_diag["fmir_head_trainable_params"] = float(len(_head_params_now))
            else:
                _fmir_grad_diag["fmir_head_grad_norm"] = -1.0
                _fmir_grad_diag["fmir_head_grad_max"] = -1.0
                _fmir_grad_diag["fmir_head_trainable_params"] = 0.0

        _fmir_grad_diag["fmir_update_norm"] = 0.0
        _fmir_grad_diag["fmir_update_max"] = 0.0
        _fmir_grad_diag["fmir_head_update_norm"] = 0.0
        _fmir_grad_diag["fmir_head_update_max"] = 0.0
        for _sub_key in ("latent_cond_proj", "first_proj", "final_proj"):
            _fmir_grad_diag[f"{_sub_key}_update_norm"] = 0.0

        # 每 accum 步更新一次参数
        self._accum_step += 1
        grad_clip = float(self.fm_cfg.get("gradient_clip_val", 1.0))
        _do_grad_check = (self._log_is_debug and self.global_step % 100 == 0)
        if self._accum_step >= accum:
            if _do_grad_check:
                self._check_grad_valid(self.model, self.global_step)
            if grad_clip > 0:
                self.clip_gradients(opt_g, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")
                if self._has_disc:
                    self.clip_gradients(opt_d, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")

            # FMIR 参数快照（仅 debug_fmir_updates 模式）
            if self._log_is_debug and self._log_debug_fmir_updates:
                _fmir_snap = [(n, p.detach().clone()) for n, p in _fmir_trainable_now]
            opt_g.step()
            if self._has_disc:
                opt_d.step()

            # FMIR 更新量诊断（仅 debug 模式）
            if self._log_is_debug and self._log_debug_fmir_updates:
                _fmir_after = dict(self.model.fmir.named_parameters())
                _unorms, _umaxs = [], []
                for _n_before, _p_before in _fmir_snap:
                    _p_after = _fmir_after.get(_n_before)
                    if _p_after is not None:
                        _delta = _p_after.detach() - _p_before
                        _unorms.append(_delta.norm(2).item())
                        _umaxs.append(_delta.abs().max().item())
                _fmir_grad_diag["fmir_update_norm"] = sum(_unorms) / max(len(_unorms), 1) if _unorms else 0.0
                _fmir_grad_diag["fmir_update_max"] = max(_umaxs) if _umaxs else 0.0
                _head_prefixes = ("latent_cond_proj.", "first_proj.", "up_blocks.3.", "up_blocks.2.",
                                  "sft_up.3.", "sft_up.2.", "final_block.", "final_proj.")
                _hu_norms, _hu_maxs = [], []
                for _n_before, _p_before in _fmir_snap:
                    if any(_n_before.startswith(pr) for pr in _head_prefixes):
                        _p_after = _fmir_after.get(_n_before)
                        if _p_after is not None:
                            _delta = _p_after.detach() - _p_before
                            _hu_norms.append(_delta.norm(2).item())
                            _hu_maxs.append(_delta.abs().max().item())
                _fmir_grad_diag["fmir_head_update_norm"] = sum(_hu_norms) / max(len(_hu_norms), 1) if _hu_norms else 0.0
                _fmir_grad_diag["fmir_head_update_max"] = max(_hu_maxs) if _hu_maxs else 0.0
                for _sub_key, _sub_prefix in [
                    ("latent_cond_proj", "latent_cond_proj."),
                    ("first_proj", "first_proj."),
                    ("final_proj", "final_proj."),
                ]:
                    _su_norms = []
                    for _n_before, _p_before in _fmir_snap:
                        if _n_before.startswith(_sub_prefix):
                            _p_after = _fmir_after.get(_n_before)
                            if _p_after is not None:
                                _delta = _p_after.detach() - _p_before
                                _su_norms.append(_delta.norm(2).item())
                    _fmir_grad_diag[f"{_sub_key}_update_norm"] = sum(_su_norms) / max(len(_su_norms), 1) if _su_norms else 0.0

            # ---- 模块梯度范数（每 _log_grad_interval 次 optimizer step） ----
            # 必须在 step 之后、zero_grad 之前计算，此时 .grad 仍保留梯度值
            if self.global_step % self._log_grad_interval == 0 and \
               self.global_step != self._last_grad_log_step:
                _norms = self._compute_module_grad_norms()
                for _k, _v in _norms.items():
                    self.log(_k, _v, logger=True, prog_bar=False)
                if self._log_is_debug:
                    self._check_grad_flow(_norms)
                self._last_grad_log_step = int(self.global_step)

            opt_g.zero_grad()
            if self._has_disc:
                opt_d.zero_grad()
            self._step_cosine_lr(opt_g, opt_d)
            if self.ema:
                self.ema.update(self.model)
            self._accum_step = 0

        # ---- Logging ----
        _log_now = (batch_idx % self._log_train_interval == 0)
        if _log_now:
            _log_cache = {}
            for k, v in metrics.items():
                if k in self._NORMAL_CSV_FIELDS or self._log_is_debug:
                    _log_cache[k] = v.item() if isinstance(v, torch.Tensor) else v
            if self._log_is_debug:
                _log_cache.update(_fmir_grad_diag)
            else:
                # 正常模式仅保留白名单中的 FMIR grad 字段
                for _k, _v in _fmir_grad_diag.items():
                    if _k in self._NORMAL_CSV_FIELDS:
                        _log_cache[_k] = _v
            if _log_cache:
                self.log_dict(_log_cache, logger=True, prog_bar=False, on_step=True)
            self.log("train_loss", g_loss.detach().item(), logger=True, prog_bar=True, on_step=True)
            self.log("lr", opt_g.param_groups[0]["lr"], logger=True, prog_bar=False, on_step=True)
            if isinstance(d_loss, torch.Tensor) and d_loss.item() != 0.0:
                self.log("d_loss", d_loss.detach().item(), logger=True, prog_bar=True, on_step=True)

        return g_loss.detach()

    def training_step(self, batch, batch_idx):
        if not self._runtime_space_logged:
            x_lq = batch[0]
            runtime_space, is_pixel_space = self._infer_runtime_space(x_lq)
            if self._log_is_debug:
                print(
                    f"[RuntimeSpace] mode={runtime_space}, "
                    f"x_lq_channels={int(x_lq.shape[1])}, "
                    f"has_enc={hasattr(self.model, 'enc') and (self.model.enc is not None)}, "
                    f"has_dec={hasattr(self.model, 'dec') and (self.model.dec is not None)}, "
                    f"loss_method={self.fm_cfg.get('method')}"
                )
            if self._log_is_debug:
                self.log("runtime/is_pixel_space", float(is_pixel_space), logger=True, prog_bar=True)
            self._runtime_space_logged = True

        # ---- 显存快照：epoch1/2/3 训练前（仅 log_memory 或 debug 模式） ----
        if self._log_memory or self._log_is_debug:
            if self.current_epoch == 0 and batch_idx == 0 and not self._mem_tracked_epoch1:
                self._mem_snapshot("epoch1_train_start(batch0)")
                self._mem_tracked_epoch1 = True
            if self.current_epoch == 1 and batch_idx == 0 and not self._mem_tracked_epoch2:
                self._mem_snapshot("epoch2_train_start(batch0)")
                self._mem_tracked_epoch2 = True
            if self.current_epoch == 2 and batch_idx == 0 and not getattr(self, "_mem_tracked_epoch3", False):
                self._mem_snapshot("epoch3_train_start(batch0)")
                self._mem_tracked_epoch3 = True

        # ---- k_steps 一致性校验（首次训练步骤） ----
        if not self._k_steps_checked:
            self._check_k_steps()
            self._k_steps_checked = True

        if self.is_e2e:
            return self._training_step_e2e(batch, batch_idx)
        return self._training_step_normal(batch, batch_idx)

    def compute_metrics(self, x_hq_hat, x_hq):
        for metric_eval in self.metric_evals:
            metric_eval.compute(x_hq_hat, x_hq)

    def save_samples(self, current_epoch):
        sample_idx = 0
        for batch_samples in self.samples:
            for img in batch_samples:
                out_path = os.path.join(self.samples_dir, f"sample_{sample_idx:03d}.png")
                save_image(img, out_path)
                sample_idx += 1

    def infer(self, x):
        use_tta = bool(self.eval_cfg.get("use_tta", False))
        metric_model, _ = self.get_metric_model()
        return metric_model.inference(x, use_tta=use_tta)

    def validation_step(self, batch, batch_idx):
        x_lq, y = batch

        # ---- 模型指纹（首次 validation 打印一次） ----
        self._print_metric_model_fingerprint()

        # ---- 统一获取 metric model（EMA/RAW） ----
        active, active_name = self.get_metric_model()

        # ---- EMA training 状态校验 + 第一个 batch 打印计算图状态（仅 debug 模式） ----
        if self._log_debug_val_graph and not self._val_debug_logged:
            if active.training:
                print("[VAL] WARNING: active.training=True, forcing active.eval()")
                active.eval()
            print(
                f"[VAL:batch0] active.training={active.training}  "
                f"torch.is_grad_enabled()={torch.is_grad_enabled()}  "
                f"torch.is_inference_mode_enabled()={torch.is_inference_mode_enabled()}"
            )
            if self._log_memory:
                self._mem_snapshot("val_start(batch0_before_infer)")

        chop = self.eval_cfg.get("chop", None)
        if chop:
            sf = chop.get("sf", 4)
            upscale = chop.get("upscale", 4)
            chop_size = chop.get("chop_size", 256)
            chop_stride = chop.get("chop_stride", 224)
            patch_spliter = ImageSpliterTh(x_lq, pch_size=chop_size, stride=chop_stride, sf=sf, extra_bs=1)
            for patch, index_infos in patch_spliter:
                patch_h, patch_w = patch.shape[2:]
                flag_pad = False
                if not (patch_h % 64 == 0 and patch_w % 64 == 0):
                    flag_pad = True
                    pad_h = (math.ceil(patch_h / 64)) * 64 - patch_h
                    pad_w = (math.ceil(patch_w / 64)) * 64 - patch_w
                    patch = F.pad(patch, pad=(0, pad_w, 0, pad_h), mode='reflect')
                pad_patch_h, pad_patch_w = patch.shape[2:]
                patch = F.interpolate(patch, size=(upscale*pad_patch_h, upscale*pad_patch_w), mode='bicubic')
                y_hat = self.infer(patch)
                if flag_pad:
                    y_hat = y_hat[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(y_hat, index_infos)
            y_hat = patch_spliter.gather()
        else:
            y_hat = self.infer(x_lq)

        # Save only a small configurable number of samples during evaluation.
        if self.samples_dir and self.max_save_images > 0:
            saved_images = sum(sample.shape[0] for sample in self.samples)
            remaining = self.max_save_images - saved_images
            if remaining > 0:
                self.samples.append(y_hat[:remaining, ...].detach().cpu())
        # ---- 第一个 batch：验证 y_hat 是否无计算图（仅 debug 模式） ----
        if self._log_debug_val_graph and not self._val_debug_logged:
            print(
                f"[VAL:batch0:after_infer] y_hat.requires_grad={y_hat.requires_grad}  "
                f"y_hat.grad_fn={y_hat.grad_fn}"
            )
            if self._log_memory:
                self._mem_snapshot("val_start(batch0_after_infer)")
            self._val_debug_logged = True

        self.compute_metrics(y_hat, y)

        # Encoder 漂移诊断: 全验证集 full-val 平均 SFT_dec(HQ)/SFT_dec(MM) PSNR
        # 注意：active 已在 validation_step 开头通过 get_metric_model() 统一获取
        if hasattr(active, "mmse") and hasattr(active, "_encode_input"):
            with torch.no_grad():
                z_hq_test = active._encode_input(y)
                z_lq_test = active._encode_input(x_lq)
                z_mmse_test = active.mmse(z_lq_test)
                cond_test = active._build_fmir_condition(x_lq, wavelet_cond=None)
                # 双路分离：诊断用纯空间条件（与推理路径一致）
                dec_hq = active._decode_latent(z_hq_test, cond=cond_test, x_lq=x_lq, wavelet_cond=None).clamp(0, 1)
                dec_mmse = active._decode_latent(z_mmse_test, cond=cond_test, x_lq=x_lq, wavelet_cond=None).clamp(0, 1)
                # 裁切到 GT 尺寸 (decode 输出可能比原图大)
                ori_h, ori_w = y.shape[2], y.shape[3]
                dec_hq = dec_hq[:, :, :ori_h, :ori_w]
                dec_mmse = dec_mmse[:, :, :ori_h, :ori_w]
                from ELIR.metrics import calculate_psnr
                diag_hq = calculate_psnr(dec_hq, y, test_y_channel=self._test_y_channel)
                diag_mmse = calculate_psnr(dec_mmse, y, test_y_channel=self._test_y_channel)
                if isinstance(diag_hq, list):
                    self._diag_sft_hq_psnr_values.extend(float(v) for v in diag_hq)
                else:
                    self._diag_sft_hq_psnr_values.append(float(diag_hq))
                if isinstance(diag_mmse, list):
                    self._diag_sft_mmse_psnr_values.extend(float(v) for v in diag_mmse)
                else:
                    self._diag_sft_mmse_psnr_values.append(float(diag_mmse))

            # Latent 空间诊断：z_mmse vs z_fmir vs z_hq
            # 必须先做 64-align padding，否则 FMIR 内部 conv/pool 会 spatial mismatch
            if hasattr(active, "fmir") and hasattr(active, "K"):
                # 打印实际使用的 K/dt（仅首次，确认与配置一致）
                if not self._latent_diag_K_logged:
                    print(
                        f"[latent diag] using model={active_name}, "
                        f"model_id={id(active)}, "
                        f"K={active.K}, dt={active.dt:.6f}, "
                        f"flow_infer_scale={active.flow_infer_scale}, "
                        f"inference_noise_scale={active.inference_noise_scale}"
                    )
                _, _, ori_h_lq, ori_w_lq = x_lq.shape
                pad_h_lq = (64 - ori_h_lq % 64) % 64
                pad_w_lq = (64 - ori_w_lq % 64) % 64
                x_lq_padded = F.pad(x_lq, (0, pad_w_lq, 0, pad_h_lq), mode="reflect") if pad_h_lq > 0 or pad_w_lq > 0 else x_lq

                _, _, ori_h_gt, ori_w_gt = y.shape
                pad_h_gt = (64 - ori_h_gt % 64) % 64
                pad_w_gt = (64 - ori_w_gt % 64) % 64
                y_padded = F.pad(y, (0, pad_w_gt, 0, pad_h_gt), mode="reflect") if pad_h_gt > 0 or pad_w_gt > 0 else y

                with torch.no_grad():
                    z_lq = active._encode_input(x_lq_padded)
                    z_hq = active._encode_input(y_padded)
                    z_mmse = active.mmse(z_lq)

                    # ---- 对齐 forward()：使用 prepare_fmir_ode_condition 统一构造 FMIR ODE 条件 ----
                    fmir_cond = active.prepare_fmir_ode_condition(x_lq_padded, z_lq, z_mmse)

                    # ODE / one_step inference for latent diagnostics
                    use_residual_amp_val = bool(self.fm_cfg.get("use_residual_amplification", False))
                    residual_target_scale_val = float(self.fm_cfg.get("residual_target_scale", 1.0))
                    residual_infer_scale_val = float(self.fm_cfg.get("residual_infer_scale", 1.0))
                    inference_mode = str(getattr(active, "inference_mode", "ode"))

                    if inference_mode == "one_step" and hasattr(active, "_one_step_flow_decode"):
                        # 使用与 forward() 相同的共享函数
                        _, z_fmir, _ = active._one_step_flow_decode(x_lq_padded)
                    else:
                        # K-step ODE：对齐 forward()，从 z_mmse + noise 出发，使用 run_fmir_ode
                        K = int(active.K)
                        # 噪声：与 forward() 一致 — 从 z_lq 计算 noise，加到 z_mmse
                        noise = active._inference_noise(z_lq, x_lq_padded.device)
                        z_start = z_mmse + noise
                        # 使用共享 ODE 积分函数（包含 flow_infer_scale）
                        if not torch.is_grad_enabled():
                            z_fmir = active.run_fmir_ode(z_start, fmir_cond, track_grad=False)
                        else:
                            z_fmir = active.run_fmir_ode(z_start, fmir_cond, track_grad=False)

                    # Charbonnier: sqrt(x^2 + 1e-6)
                    eps_charb = 1e-6
                    charb_mmse = torch.sqrt((z_mmse - z_hq).pow(2) + eps_charb).mean()
                    charb_fmir = torch.sqrt((z_fmir - z_hq).pow(2) + eps_charb).mean()
                    gain_charb = charb_mmse - charb_fmir

                    mse_mmse = F.mse_loss(z_mmse, z_hq)
                    mse_fmir = F.mse_loss(z_fmir, z_hq)
                    gain_mse = mse_mmse - mse_fmir

                    # delta 分析
                    delta_fmir = z_fmir - z_mmse          # FMIR 移动量（已 infer-scaled）
                    delta_target = z_hq - z_mmse           # 目标移动量

                    fmir_delta_norm = delta_fmir.flatten(1).norm(dim=1).mean() / delta_fmir[0].numel()**0.5
                    target_delta_norm = delta_target.flatten(1).norm(dim=1).mean() / delta_target[0].numel()**0.5

                    delta_norm_ratio = fmir_delta_norm / (target_delta_norm + 1e-8)

                    delta_cosine = F.cosine_similarity(
                        delta_fmir.flatten(1), delta_target.flatten(1), dim=1
                    ).mean()

                    # ---- residual amplification 验证期诊断 ----
                    # target_v_norm_amp = gamma * target_delta_norm（理论值）
                    target_delta_norm_amp = residual_target_scale_val * target_delta_norm

                    self._latent_diag_values["latent_charb_mmse_to_hq"].append(float(charb_mmse.detach().cpu()))
                    self._latent_diag_values["latent_charb_fmir_to_hq"].append(float(charb_fmir.detach().cpu()))
                    self._latent_diag_values["latent_gain_charb"].append(float(gain_charb.detach().cpu()))
                    self._latent_diag_values["latent_mse_mmse_to_hq"].append(float(mse_mmse.detach().cpu()))
                    self._latent_diag_values["latent_mse_fmir_to_hq"].append(float(mse_fmir.detach().cpu()))
                    self._latent_diag_values["latent_gain_mse"].append(float(gain_mse.detach().cpu()))
                    self._latent_diag_values["latent_fmir_delta_norm"].append(float(fmir_delta_norm.detach().cpu()))
                    self._latent_diag_values["latent_target_delta_norm"].append(float(target_delta_norm.detach().cpu()))
                    self._latent_diag_values["latent_delta_norm_ratio"].append(float(delta_norm_ratio.detach().cpu()))
                    self._latent_diag_values["latent_delta_cosine"].append(float(delta_cosine.detach().cpu()))

                    # 记录 amplification 参数（每个 batch 重复，求平均不影响）
                    self._latent_diag_values.setdefault("residual_target_scale_value", []).append(residual_target_scale_val)
                    self._latent_diag_values.setdefault("residual_infer_scale_value", []).append(residual_infer_scale_val)
                    self._latent_diag_values.setdefault("latent_target_delta_norm_amp", []).append(float(target_delta_norm_amp.detach().cpu()))

                if not self._latent_diag_K_logged:
                    # K 在 ode 路径中定义；one_step 路径从 model 读取
                    _K_val = K if inference_mode != "one_step" else int(getattr(active, "K", 0))
                    self.log("latent_diag_K", float(_K_val), logger=True)
                    self._latent_diag_K_logged = True

                # ---- 等价性检查：验证 main forward 与 latent diag ODE 输出一致 ----
                if (self._debug_eval_equivalence
                        and not self._equivalence_checked
                        and inference_mode != "one_step"):
                    self._equivalence_checked = True
                    # 使用与 forward() 相同的 decoder 条件（纯空间，无 wavelet）
                    decoder_cond_eq = active._build_fmir_condition(x_lq_padded, wavelet_cond=None)
                    # 解码 latent diag 的 z_fmir
                    y_from_diag = active._decode_latent(
                        z_fmir, cond=decoder_cond_eq, x_lq=x_lq_padded, wavelet_cond=None
                    ).clamp(0, 1)[:, :, :ori_h_gt, :ori_w_gt]

                    # 从 main forward 重新计算（确保完全相同的输入）
                    y_from_main = active.forward(x_lq)

                    # 差异计算
                    lq_diff = (x_lq_padded[:, :, :ori_h_lq, :ori_w_lq] - x_lq).abs().max().item()
                    z_lq_diff_val = 0.0
                    z_mmse_diff_val = 0.0
                    z_final_diff_val = 0.0
                    output_diff = (y_from_diag - y_from_main).abs().max().item()

                    # 重新运行一次 forward 获取中间 latent（与 validation 主路径对齐）
                    with torch.no_grad():
                        z_check = active._encode_input(x_lq_padded)
                        z_mmse_check = active.mmse(z_check)
                        noise_check = active._inference_noise(z_check, x_lq_padded.device)
                        z_start_check = z_mmse_check + noise_check
                        fmir_cond_check = active.prepare_fmir_ode_condition(x_lq_padded, z_check, z_mmse_check)
                        z_final_check = active.run_fmir_ode(z_start_check, fmir_cond_check, track_grad=False)
                        y_check = active._decode_latent(
                            z_final_check, cond=decoder_cond_eq, x_lq=x_lq_padded, wavelet_cond=None
                        ).clamp(0, 1)[:, :, :ori_h_gt, :ori_w_gt]

                    from ELIR.metrics import calculate_psnr
                    psnr_a = calculate_psnr(y_from_diag, y, test_y_channel=self._test_y_channel)
                    psnr_b = calculate_psnr(y_from_main, y, test_y_channel=self._test_y_channel)
                    if isinstance(psnr_a, list):
                        psnr_a = sum(psnr_a) / len(psnr_a)
                    if isinstance(psnr_b, list):
                        psnr_b = sum(psnr_b) / len(psnr_b)

                    print(
                        "[validation/eval equivalence] "
                        f"lq_diff={lq_diff:.8e}, "
                        f"z_lq_diff={z_lq_diff_val:.8e}, "
                        f"z_mmse_diff={z_mmse_diff_val:.8e}, "
                        f"z_final_diff={z_final_diff_val:.8e}, "
                        f"output_diff={output_diff:.8e}, "
                        f"psnr_a(diag)={psnr_a:.8f}, "
                        f"psnr_b(main)={psnr_b:.8f}"
                    )

                    # 断言输出一致性
                    if abs(psnr_a - psnr_b) > 0.01 or output_diff > 1e-4:
                        print(
                            "[validation/eval equivalence] WARNING: "
                            "main forward and latent diag ODE outputs differ! "
                            "This indicates a code path divergence."
                        )

        # =================================================================
        # FMIR Residual Scale Sweep（eval-only 诊断）
        # 对 v_pred 做 scale∈[1,2,5,10] 放大，检查 latent/pixel 指标。
        # =================================================================
        sweep_scales = self.eval_cfg.get("fmir_scale_sweep", None)
        if sweep_scales and hasattr(active, "fmir") and active.fmir is not None:
            sweep_t = float(self.eval_cfg.get("fmir_scale_sweep_t", 0.5))
            sweep_use_cond = bool(self.eval_cfg.get("fmir_scale_sweep_use_cond", True))

            _, _, ori_h_lq2, ori_w_lq2 = x_lq.shape
            pad_h2 = (64 - ori_h_lq2 % 64) % 64
            pad_w2 = (64 - ori_w_lq2 % 64) % 64
            xlq_p = F.pad(x_lq, (0, pad_w2, 0, pad_h2), mode="reflect") if pad_h2 > 0 or pad_w2 > 0 else x_lq

            _, _, ori_h_gt2, ori_w_gt2 = y.shape
            pad_hg2 = (64 - ori_h_gt2 % 64) % 64
            pad_wg2 = (64 - ori_w_gt2 % 64) % 64
            ygt_p = F.pad(y, (0, pad_wg2, 0, pad_hg2), mode="reflect") if pad_hg2 > 0 or pad_wg2 > 0 else y

            with torch.no_grad():
                z_lq_s = active._encode_input(xlq_p)
                z_hq_s = active._encode_input(ygt_p)
                z_mmse_s = active.mmse(z_lq_s)

                # FMIR 条件：使用与 forward() 一致的 prepare_fmir_ode_condition
                if sweep_use_cond:
                    fc_s = active.prepare_fmir_ode_condition(xlq_p, z_lq_s, z_mmse_s)
                else:
                    fc_s = None

                t_v = torch.full((xlq_p.shape[0],), sweep_t, device=xlq_p.device, dtype=z_mmse_s.dtype)
                t_tt = t_v[:, None, None, None]
                t_ee = pos_emb(t_v, active.t_emb_dim).to(xlq_p.device)

                v_pred_s = active.fmir(z_mmse_s, t_ee, cond=fc_s, t=t_tt)
                target_v_s = z_hq_s - z_mmse_s

                # ---- Residual amplification：sweep 应用在 infer-scaled delta 上 ----
                use_residual_amp_sw = bool(self.fm_cfg.get("use_residual_amplification", False))
                residual_target_scale_sw = float(self.fm_cfg.get("residual_target_scale", 1.0))
                residual_infer_scale_sw = float(self.fm_cfg.get("residual_infer_scale", 1.0))

                if use_residual_amp_sw:
                    delta_base = residual_infer_scale_sw * v_pred_s
                else:
                    delta_base = v_pred_s

                # 统计 v_pred / target 幅值+方向
                vp_n_raw = v_pred_s.flatten(1).norm(dim=1).mean() / max(v_pred_s[0].numel(), 1) ** 0.5
                vp_n_infer = delta_base.flatten(1).norm(dim=1).mean() / max(delta_base[0].numel(), 1) ** 0.5
                tv_n_raw = target_v_s.flatten(1).norm(dim=1).mean() / max(target_v_s[0].numel(), 1) ** 0.5
                tv_n_amp = residual_target_scale_sw * tv_n_raw
                vp_cos_raw = F.cosine_similarity(
                    v_pred_s.flatten(1), target_v_s.flatten(1), dim=1
                ).mean()
                # cosine to amplified target（和 raw 理论上一样，方便检查）
                target_v_s_amp = residual_target_scale_sw * target_v_s
                vp_cos_amp = F.cosine_similarity(
                    v_pred_s.flatten(1), target_v_s_amp.flatten(1), dim=1
                ).mean()

                # Charbonnier
                eps_c = 1e-6
                charb_mmse_s = torch.sqrt((z_mmse_s - z_hq_s).pow(2) + eps_c).mean()

                # Decoder 条件（纯空间，用于 pixel PSNR — 对齐 forward() 双路分离设计）
                dec_c_s = active._build_fmir_condition(xlq_p, wavelet_cond=None)
                ori_h3, ori_w3 = y.shape[2], y.shape[3]

                # MMSE baseline pixel PSNR
                y_mmse_s = active._decode_latent(
                    z_mmse_s, cond=dec_c_s, x_lq=xlq_p, wavelet_cond=None
                ).clamp(0, 1)[:, :, :ori_h3, :ori_w3]

                from ELIR.metrics import calculate_psnr
                psnr_mmse_s = calculate_psnr(y_mmse_s, y, test_y_channel=self._test_y_channel)
                if isinstance(psnr_mmse_s, list):
                    psnr_mmse_val = sum(psnr_mmse_s) / len(psnr_mmse_s)
                else:
                    psnr_mmse_val = float(psnr_mmse_s)

               # 对各 scale 扫一遍（sweep 作用在 infer-scaled delta 上）
                _y_scale2 = None  # 用于强制验证
                for sc in sweep_scales:
                    sc_f = float(sc)
                    z_scaled = z_mmse_s + sc_f * delta_base

                    charb_sc = torch.sqrt((z_scaled - z_hq_s).pow(2) + eps_c).mean()
                    gain_sc = charb_mmse_s - charb_sc

                    delta_sc = z_scaled - z_mmse_s
                    dn_sc = delta_sc.flatten(1).norm(dim=1).mean() / max(delta_sc[0].numel(), 1) ** 0.5
                    nr_sc = dn_sc / (tv_n_raw + 1e-8)
                    cos_sc = F.cosine_similarity(
                        delta_sc.flatten(1), target_v_s.flatten(1), dim=1
                    ).mean()

                    y_sc = active._decode_latent(
                        z_scaled, cond=dec_c_s, x_lq=xlq_p, wavelet_cond=None
                    ).clamp(0, 1)[:, :, :ori_h3, :ori_w3]

                    psnr_sc = calculate_psnr(y_sc, y, test_y_channel=self._test_y_channel)
                    if isinstance(psnr_sc, list):
                        psnr_sc_val = sum(psnr_sc) / len(psnr_sc)
                    else:
                        psnr_sc_val = float(psnr_sc)

                    if abs(sc_f - 2.0) < 0.01:
                        _y_scale2 = y_sc.detach().cpu()

                    # key 命名：小数点转 "p"
                    sc_key = str(sc).replace(".", "p")
                    for k, v in [
                        (f"fmir_scale_{sc_key}_latent_charb", float(charb_sc.detach().cpu())),
                        (f"fmir_scale_{sc_key}_latent_gain", float(gain_sc.detach().cpu())),
                        (f"fmir_scale_{sc_key}_norm_ratio", float(nr_sc.detach().cpu())),
                        (f"fmir_scale_{sc_key}_delta_cosine", float(cos_sc.detach().cpu())),
                        (f"fmir_scale_{sc_key}_psnr", psnr_sc_val),
                    ]:
                        self._fmir_scale_sweep_values.setdefault(k, []).append(v)

                    # ---- one_step_scale_* 别名（用于 calibrated one-step 诊断） ----
                    _ONE_STEP_ALIAS_SCALES = {0.5, 1, 1.5, 2, 3}
                    if sc_f in _ONE_STEP_ALIAS_SCALES:
                        for k, v in [
                            (f"one_step_scale_{sc_key}_psnr", psnr_sc_val),
                            (f"one_step_scale_{sc_key}_latent_gain", float(gain_sc.detach().cpu())),
                            (f"one_step_scale_{sc_key}_delta_cosine", float(cos_sc.detach().cpu())),
                        ]:
                            self._fmir_scale_sweep_values.setdefault(k, []).append(v)

                # 全局统计（不依赖 scale）
                for k, v in [
                    ("fmir_scale_sweep_mmse_psnr", psnr_mmse_val),
                    ("one_step_mmse_psnr", psnr_mmse_val),
                    ("fmir_scale_sweep_target_norm", float(tv_n_raw.detach().cpu())),
                    ("fmir_scale_sweep_target_norm_amp", float(tv_n_amp.detach().cpu())),
                    ("fmir_scale_sweep_v_pred_norm", float(vp_n_raw.detach().cpu())),
                    ("fmir_scale_sweep_v_pred_norm_infer", float(vp_n_infer.detach().cpu())),
                    ("fmir_scale_sweep_v_pred_target_cosine", float(vp_cos_raw.detach().cpu())),
                    ("fmir_scale_sweep_v_pred_target_cosine_amp", float(vp_cos_amp.detach().cpu())),
                    ("residual_target_scale_value", residual_target_scale_sw),
                    ("residual_infer_scale_value", residual_infer_scale_sw),
                ]:
                    self._fmir_scale_sweep_values.setdefault(k, []).append(v)

                # ---- 强制验证：直接用 scale=2 pred 计算 PSNR ----
                if _y_scale2 is not None:
                    from ELIR.metrics import calculate_psnr
                    _fv_psnr = calculate_psnr(_y_scale2, y, test_y_channel=self._test_y_channel)
                    if isinstance(_fv_psnr, list):
                        _fv_psnr_val = sum(_fv_psnr) / len(_fv_psnr)
                    else:
                        _fv_psnr_val = float(_fv_psnr)
                    self._fmir_scale_sweep_values.setdefault("force_verify_scale2_psnr", []).append(_fv_psnr_val)

    def on_train_epoch_start(self):
        # 每个 epoch 重置峰值显存统计，使 gpu_peak_allocated_gb 反映该 epoch 的真实峰值
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_start(self):
        if (self._log_memory or self._log_is_debug) and not self._mem_tracked_val_before:
            self._mem_snapshot("val_epoch_start")
            self._mem_tracked_val_before = True

    def on_validation_epoch_end(self):
        if (self._log_memory or self._log_is_debug) and not self._mem_tracked_val_after:
            self._mem_snapshot("val_epoch_end(before_empty_cache)")
            self._mem_tracked_val_after = True

        if self.samples_dir and self.samples:
            self.save_samples(self.current_epoch)
            self.samples.clear()
        self.log('global_step', self.global_step)
        main_psnr_val = None
        for metric_eval in self.metric_evals:
            result = metric_eval.get_final().item()
            # 记录原名（向后兼容 eval metric 查找）
            self.log(metric_eval.metric, result, sync_dist=True, prog_bar=True)
            if metric_eval.metric == "psnr":
                main_psnr_val = result
                # 额外记录唯一明确的 val_psnr 供 ModelCheckpoint 监控
                self.log("val_psnr", result, sync_dist=True, prog_bar=True)

        # ---- flow_gain_psnr = main_psnr - diag_sft_mmse_psnr_full ----
        if main_psnr_val is not None and len(self._diag_sft_mmse_psnr_values) > 0:
            diag_mmse_avg = sum(self._diag_sft_mmse_psnr_values) / len(self._diag_sft_mmse_psnr_values)
            self.log("flow_gain_psnr", main_psnr_val - diag_mmse_avg, sync_dist=True, logger=True)

        # Full-val encoder 漂移诊断平均
        if len(self._diag_sft_hq_psnr_values) > 0:
            diag_hq_avg = sum(self._diag_sft_hq_psnr_values) / len(self._diag_sft_hq_psnr_values)
            diag_mmse_avg = sum(self._diag_sft_mmse_psnr_values) / len(self._diag_sft_mmse_psnr_values)
            self.log("diag_sft_hq_psnr_full", diag_hq_avg, sync_dist=True, logger=True)
            self.log("diag_sft_mmse_psnr_full", diag_mmse_avg, sync_dist=True, logger=True)
            self._diag_sft_hq_psnr_values.clear()
            self._diag_sft_mmse_psnr_values.clear()

        # Full-val latent 空间诊断平均
        _NORMAL_LATENT_KEYS = {"latent_delta_cosine", "latent_gain_charb"}
        latent_keys = list(self._latent_diag_values.keys())
        if len(self._latent_diag_values[latent_keys[0]]) > 0:
            for key in latent_keys:
                vals = self._latent_diag_values[key]
                if len(vals) > 0:
                    if key in _NORMAL_LATENT_KEYS or self._log_is_debug:
                        avg_val = sum(vals) / len(vals)
                        self.log(key, avg_val, sync_dist=True, logger=True)
                vals.clear()

        # FMIR Scale Sweep full-val 平均（仅 debug 模式）
        if self._log_is_debug and self._fmir_scale_sweep_values:
            for key, vals in self._fmir_scale_sweep_values.items():
                if len(vals) > 0:
                    self.log(key, sum(vals) / len(vals), sync_dist=True, logger=True)
                vals.clear()

        # GPU peak memory
        if torch.cuda.is_available():
            self.log("gpu_peak_allocated_gb", torch.cuda.max_memory_allocated() / (1024**3),
                     logger=True, prog_bar=False)
        torch.cuda.empty_cache()

    def _save_optional_state(self, checkpoint, key, module):
        if module is not None:
            checkpoint[key] = module.state_dict()

    def _load_optional_state(self, checkpoint, key, module):
        """加载可选子模块 state_dict。返回 (missing_count, unexpected_count) 或 None（跳过）。"""
        if module is None:
            return None
        if key not in checkpoint:
            print(f"[ckpt] WARNING: key '{key}' NOT FOUND in checkpoint — "
                  f"module '{type(module).__name__}' will use UNTRAINED weights")
            return None
        sd = checkpoint[key]
        if not sd or len(sd) == 0:
            print(f"[ckpt] WARNING: key '{key}' exists but state_dict is EMPTY — "
                  f"module '{type(module).__name__}' will use UNTRAINED weights")
            return None
        missing, unexpected = module.load_state_dict(sd, strict=False)
        total_keys = len(module.state_dict())
        loaded_frac = 1.0 - len(missing) / max(total_keys, 1)
        if loaded_frac < 0.5:
            print(f"[ckpt] WARNING: {key} only {loaded_frac:.0%} params loaded "
                  f"(missing={len(missing)}, unexpected={len(unexpected)}, total={total_keys})")
        else:
            print(f"[ckpt] restored {key}: missing={len(missing)}, unexpected={len(unexpected)}")
        return len(missing), len(unexpected)

    @staticmethod
    def _module_load_complete(result):
        """判断 _load_optional_state 返回值是否表示完整加载（missing=0, unexpected=0）。"""
        return result is not None and result[0] == 0 and result[1] == 0

    def on_save_checkpoint(self, checkpoint):
        src = self.ema.model if self.ema else self.model

        # ---- 格式版本 ----
        checkpoint["checkpoint_format_version"] = 2

        # ---- 保存推理运行配置 ----
        checkpoint["inference_runtime"] = {
            "k_steps": int(getattr(src, "K", 1)),
            "dt": float(getattr(src, "dt", 1.0)),
            "flow_infer_scale": float(getattr(src, "flow_infer_scale", 1.0)),
            "inference_noise_scale": float(getattr(src, "inference_noise_scale", 0.0)),
            "inference_mode": str(getattr(src, "inference_mode", "ode")),
        }

        # ---- 保存完整 EMA 模型 state_dict（覆盖所有子模块/aux/buffer） ----
        if self.ema is not None:
            checkpoint["ema_model_state_dict"] = self.ema.model.state_dict()
            checkpoint["ema_decay"] = self._ema_decay

        # ---- v2 新格式默认不保存拆分 key，避免 EMA 权重重复存储 ----
        # 如需兼容旧外部工具，设置 fm_cfg.save_legacy_split_keys=True
        if bool(self.fm_cfg.get("save_legacy_split_keys", False)):
            self._save_optional_state(checkpoint, 'state_dict_fmir', getattr(src, 'fmir', None))
            self._save_optional_state(checkpoint, 'state_dict_mmse', getattr(src, 'mmse', None))
            self._save_optional_state(checkpoint, 'state_dict_enc', getattr(src, 'enc', None))
            self._save_optional_state(checkpoint, 'state_dict_dec', getattr(src, 'dec', None))
            self._save_optional_state(checkpoint, 'state_dict_wavelet', getattr(src, 'wavelet_stem', None))

            aux_keys = ['decoder_cond_fusions', 'fmir_cond_fusions', 'latent_norm',
                        'dino_projector', 'dino_spatial_proj', 'sk_fusion']
            aux_sd = {}
            for key in aux_keys:
                mod = getattr(src, key, None)
                if mod is not None:
                    for k, v in mod.state_dict().items():
                        aux_sd[f"{key}.{k}"] = v
            if aux_sd:
                checkpoint['state_dict_elir_aux'] = aux_sd

        # ---- condition / SFT：新格式不再单独保存（已包含在完整 state_dict_fmir / state_dict_dec 中） ----

        if self.discriminator is not None:
            self._save_optional_state(checkpoint, 'state_dict_disc', self.discriminator)
        return checkpoint

    @staticmethod
    def _extract_prefixed_state(state_dict, prefixes):
        """从 state_dict 中提取以指定前缀开头的子模块权重。

        Args:
            state_dict: 完整的 state_dict
            prefixes: 前缀列表，按优先级从高到低排列

        Returns:
            (extracted_sd, matched_prefix) — extracted_sd 中 key 已去除前缀
        """
        for prefix in prefixes:
            result = {}
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    result[key[len(prefix):]] = value
            if result:
                return result, prefix
        return {}, None

    def on_load_checkpoint(self, checkpoint):
        # ================================================================
        # 格式检测
        # ================================================================
        fmt_ver = checkpoint.get("checkpoint_format_version", 1)
        has_full_ema = bool(checkpoint.get("ema_model_state_dict"))
        has_elir_aux = bool(checkpoint.get("state_dict_elir_aux"))

        # ---- 动态 legacy 检测：只检查 EMA 模型实际存在的模块 ----
        dst_for_detection = self.ema.model if self.ema else self.model
        legacy_module_map = {
            "state_dict_fmir": getattr(dst_for_detection, "fmir", None),
            "state_dict_mmse": getattr(dst_for_detection, "mmse", None),
            "state_dict_enc": getattr(dst_for_detection, "enc", None),
            "state_dict_dec": getattr(dst_for_detection, "dec", None),
            "state_dict_wavelet": getattr(dst_for_detection, "wavelet_stem", None),
        }
        required_legacy_keys = [
            key for key, module in legacy_module_map.items()
            if module is not None
        ]
        missing_legacy_keys = [
            key for key in required_legacy_keys
            if key not in checkpoint or not bool(checkpoint.get(key))
        ]
        has_legacy_split_ema = (
            len(required_legacy_keys) > 0
            and len(missing_legacy_keys) == 0
        )

        print(
            "[EMA checkpoint detection] "
            f"format_version={fmt_ver}, "
            f"full_ema={has_full_ema}, "
            f"legacy_split_ema={has_legacy_split_ema}, "
            f"elir_aux={has_elir_aux}, "
            f"required_legacy_keys={required_legacy_keys}, "
            f"missing_legacy_keys={missing_legacy_keys}"
        )

        # 打印推理运行配置（仅作参考，不覆盖 eval.yaml 显式指定的配置）
        if "inference_runtime" in checkpoint:
            ir = checkpoint["inference_runtime"]
            print(
                f"[ckpt inference_runtime] "
                f"k_steps={ir.get('k_steps')}, "
                f"dt={ir.get('dt')}, "
                f"flow_infer_scale={ir.get('flow_infer_scale')}, "
                f"inference_noise_scale={ir.get('inference_noise_scale')}, "
                f"inference_mode={ir.get('inference_mode')}  "
                f"(for reference only; eval.yaml overrides take precedence)"
            )

        # ================================================================
        # 路径 1：完整 EMA 优先恢复
        # ================================================================
        if self.ema is not None and has_full_ema:
            try:
                result = self.ema.model.load_state_dict(
                    checkpoint["ema_model_state_dict"],
                    strict=True,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    "[EMA restore] Full EMA state restore failed (strict=True): "
                    f"{e}"
                ) from e
            print(
                "[EMA restore] "
                "format=full_v2 "
                "exact=True "
                f"missing={len(result.missing_keys) if result.missing_keys else 0} "
                f"unexpected={len(result.unexpected_keys) if result.unexpected_keys else 0}"
            )
            # 完整 EMA 恢复成功 → 跳过所有后续拆分模块加载
            # Lightning 仍会自动从 checkpoint["state_dict"] 加载 RAW 模型
            self._print_ckpt_summary(checkpoint, skip_split_loading=True)
            return

        # ================================================================
        # 路径 2：拆分 EMA 恢复（旧格式兼容）
        # ================================================================
        dst = self.ema.model if self.ema else self.model

        if self.ema is not None and has_legacy_split_ema:
            # ---- 加载所有拆分主模块，保留全部结果 ----
            fmir_result = self._load_optional_state(checkpoint, 'state_dict_fmir', getattr(dst, 'fmir', None))
            mmse_result = self._load_optional_state(checkpoint, 'state_dict_mmse', getattr(dst, 'mmse', None))
            enc_result = self._load_optional_state(checkpoint, 'state_dict_enc', getattr(dst, 'enc', None))
            dec_result = self._load_optional_state(checkpoint, 'state_dict_dec', getattr(dst, 'dec', None))
            wavelet_result = self._load_optional_state(checkpoint, 'state_dict_wavelet', getattr(dst, 'wavelet_stem', None))

            # 只检查当前模型实际存在的模块
            split_results = {}
            for key, result in [
                ("fmir", fmir_result),
                ("mmse", mmse_result),
                ("enc", enc_result),
                ("dec", dec_result),
                ("wavelet", wavelet_result),
            ]:
                if legacy_module_map.get(f"state_dict_{key}") is not None:
                    split_results[key] = result

            split_complete = all(
                result is not None and result[0] == 0 and result[1] == 0
                for result in split_results.values()
            )

            fmir_loaded_complete = self._module_load_complete(split_results.get("fmir"))
            dec_loaded_complete = self._module_load_complete(split_results.get("dec"))
            dec_name = getattr(getattr(dst, 'dec', None), '__class__', type(None)).__name__

            # ---- condition / SFT 兼容（仅当主模块加载不完整时） ----
            if 'state_dict_condition' in checkpoint and checkpoint['state_dict_condition']:
                if fmir_loaded_complete:
                    print("[ckpt] skip state_dict_condition: full state_dict_fmir already restored")
                else:
                    cond_module = getattr(dst, 'condition_stem', None)
                    if cond_module is not None:
                        missing, unexpected = cond_module.load_state_dict(
                            checkpoint['state_dict_condition'], strict=False)
                        print(f"[ckpt] restored state_dict_condition (standalone): "
                              f"missing={len(missing)}, unexpected={len(unexpected)}")
                    elif hasattr(dst, 'fmir') and dst.fmir is not None:
                        missing, unexpected = dst.fmir.load_state_dict(
                            checkpoint['state_dict_condition'], strict=False)
                        print(f"[ckpt] restored state_dict_condition (into fmir): "
                              f"missing={len(missing)}, unexpected={len(unexpected)}")

            dec_is_sft = ('sft' in dec_name.lower()) if dec_name != 'NoneType' else False
            if dec_loaded_complete and dec_is_sft:
                print(f"[ckpt] skip state_dict_sft: state_dict_dec already contains SFT params "
                      f"(dec_name={dec_name})")
            else:
                self._load_optional_state(checkpoint, 'state_dict_sft', getattr(dst, 'sft_refiner', None))

            # ---- 辅助模块恢复 ----
            critical_aux_modules = {"fmir_cond_fusions"}
            aux_result = self._restore_legacy_aux(checkpoint, dst, critical_aux_modules)

            # ---- 计算 exact ----
            # exact=True 要求：
            #   1. 所有存在的拆分模块完整（missing=0, unexpected=0）
            #   2. 所有 aux 模块完整且来源是 EMA 权重（source=ema_aux）
            # raw_aux_fallback 即使加载完整也不是 EMA 权重，exact 必须为 False
            legacy_exact = (
                split_complete
                and aux_result["source"] == "ema_aux"
                and aux_result["complete"]
            )

            if aux_result["source"] == "ema_aux":
                print(
                    "[EMA restore] "
                    f"format=legacy_split_with_aux "
                    f"exact={legacy_exact}"
                )
            elif aux_result["source"] == "raw_aux_fallback":
                print(
                    "[EMA restore] "
                    "format=legacy_split_raw_aux_fallback "
                    "exact=False "
                    "warning=original EMA auxiliary weights were never saved; "
                    "using RAW auxiliary weights as approximation. "
                    "exact=False regardless of whether RAW aux loaded cleanly — "
                    "these are not EMA-smoothed weights."
                )
            else:
                print(
                    "[EMA restore] "
                    f"format=legacy_split_no_aux "
                    f"exact={legacy_exact}"
                )

        elif self.ema is not None:
            # ---- EMA 对象存在但无任何可用 EMA 权重 → 拒绝继续 ----
            raise RuntimeError(
                "[EMA restore] EMA object exists, but checkpoint contains "
                "neither ema_model_state_dict nor a complete legacy split EMA state. "
                "Refusing to evaluate with an unrestored EMA model. "
                "Available checkpoint keys: "
                f"{[k for k in checkpoint.keys() if 'state_dict' in k.lower() or 'ema' in k.lower()]}"
            )

        elif self.ema is None:
            print("[EMA restore] format=none — no EMA, using RAW model directly")

        self._print_ckpt_summary(checkpoint, skip_split_loading=False)

    def _restore_legacy_aux(self, checkpoint, dst, critical_aux_modules):
        """从旧 checkpoint 恢复辅助模块到 dst (EMA model)。

        Returns:
            dict: {
                "source": "ema_aux" | "raw_aux_fallback" | "none",
                "complete": True/False,   # 所有现有 aux 模块 missing=0 AND unexpected=0
                "modules": {mod_name: {"missing": [...], "unexpected": [...], "found": bool}},
            }
        """
        aux_module_names = ['decoder_cond_fusions', 'fmir_cond_fusions', 'latent_norm',
                            'dino_projector', 'dino_spatial_proj', 'sk_fusion']

        # ---- 检查哪些 aux 模块实际存在 ----
        existing_aux = [n for n in aux_module_names if getattr(dst, n, None) is not None]
        modules_status = {}

        if not existing_aux:
            return {"source": "none", "complete": True, "modules": {}}

        # ---- 路径 A：state_dict_elir_aux 存在（EMA aux 权重） ----
        if 'state_dict_elir_aux' in checkpoint and checkpoint['state_dict_elir_aux']:
            aux_sd = checkpoint['state_dict_elir_aux']
            all_ok = True

            for mod_name in existing_aux:
                mod = getattr(dst, mod_name, None)
                if mod is None:
                    continue
                mod_sd = {}
                prefix = f"{mod_name}."
                for k, v in aux_sd.items():
                    if k.startswith(prefix):
                        mod_sd[k[len(prefix):]] = v

                is_critical = mod_name in critical_aux_modules
                found = bool(mod_sd)

                if not mod_sd:
                    if is_critical:
                        raise RuntimeError(
                            f"state_dict_elir_aux is missing critical module {mod_name} — "
                            f"no keys with prefix '{prefix}' found in aux state_dict"
                        )
                    print(f"[ckpt] WARNING: state_dict_elir_aux has no keys for {mod_name}")
                    all_ok = False
                    modules_status[mod_name] = {"missing": ["<no prefix match>"], "unexpected": [], "found": False}
                    continue

                missing, unexpected = mod.load_state_dict(mod_sd, strict=False)
                print(
                    f"[legacy aux restore] "
                    f"module={mod_name} "
                    f"source=state_dict_elir_aux "
                    f"missing={len(missing)} "
                    f"unexpected={len(unexpected)}"
                )

                modules_status[mod_name] = {
                    "missing": list(missing),
                    "unexpected": list(unexpected),
                    "found": found,
                }

                # 关键模块：missing OR unexpected 任一非零都应报错
                if is_critical and (len(missing) > 0 or len(unexpected) > 0):
                    raise RuntimeError(
                        f"state_dict_elir_aux critical module {mod_name}: "
                        f"{len(missing)} missing keys, {len(unexpected)} unexpected keys. "
                        f"missing={missing[:10]}... unexpected={unexpected[:10]}..."
                    )
                if len(missing) > 0 or len(unexpected) > 0:
                    all_ok = False

            return {"source": "ema_aux", "complete": all_ok, "modules": modules_status}

        # ---- 路径 B：无 state_dict_elir_aux → 从 checkpoint["state_dict"] 提取 RAW 权重 ----
        print("[ckpt] state_dict_elir_aux NOT FOUND — "
              "falling back to extract aux modules from checkpoint[\"state_dict\"] (RAW weights)")
        print("[ckpt] NOTE: original EMA auxiliary weights were never saved; "
              "using RAW auxiliary weights as approximation")

        main_sd = checkpoint.get("state_dict", {})
        if not main_sd:
            print("[ckpt] WARNING: checkpoint[\"state_dict\"] is EMPTY — "
                  "cannot extract aux module weights")
            for mod_name in existing_aux:
                if mod_name in critical_aux_modules:
                    raise RuntimeError(
                        f"Critical module {mod_name} cannot be restored: "
                        "no state_dict_elir_aux and checkpoint[\"state_dict\"] is empty"
                    )
                modules_status[mod_name] = {"missing": ["<empty state_dict>"], "unexpected": [], "found": False}
            all_ok = any(m["found"] for m in modules_status.values())
            return {"source": "raw_aux_fallback", "complete": False, "modules": modules_status}

        # 打印真实 key 前缀（一次性诊断）
        sample_keys = [k for k in list(main_sd.keys())[:30]
                       if 'cond_fusion' in k.lower() or 'latent_norm' in k.lower()]
        if sample_keys:
            print(f"[ckpt diagnostic] sample aux-related keys in checkpoint['state_dict']: {sample_keys}")

        all_ok = True
        for mod_name in existing_aux:
            mod = getattr(dst, mod_name, None)
            if mod is None:
                continue
            is_critical = mod_name in critical_aux_modules

            # 尝试多种前缀
            mod_sd, used_prefix = self._extract_prefixed_state(
                main_sd,
                [f"model.{mod_name}.", f"{mod_name}."],
            )

            if not mod_sd:
                if is_critical:
                    raise RuntimeError(
                        f"Critical module {mod_name} cannot be restored from "
                        f"checkpoint[\"state_dict\"] — no keys with expected prefixes found"
                    )
                print(f"[ckpt] WARNING: cannot find {mod_name} in checkpoint[\"state_dict\"]")
                modules_status[mod_name] = {"missing": ["<no prefix match>"], "unexpected": [], "found": False}
                all_ok = False
                continue

            missing, unexpected = mod.load_state_dict(mod_sd, strict=False)
            print(
                f"[legacy aux restore] "
                f"module={mod_name} "
                f"source=checkpoint.state_dict "
                f"prefix={used_prefix} "
                f"missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )

            modules_status[mod_name] = {
                "missing": list(missing),
                "unexpected": list(unexpected),
                "found": True,
            }

            # 关键模块：missing OR unexpected 任一非零都应报错
            if is_critical and (len(missing) > 0 or len(unexpected) > 0):
                raise RuntimeError(
                    f"Critical module {mod_name}: {len(missing)} missing keys, "
                    f"{len(unexpected)} unexpected keys "
                    f"after RAW fallback. "
                    f"missing={missing[:10]}... unexpected={unexpected[:10]}..."
                )
            if len(missing) > 0 or len(unexpected) > 0:
                all_ok = False

        return {"source": "raw_aux_fallback", "complete": all_ok, "modules": modules_status}

    def _print_ckpt_summary(self, checkpoint, skip_split_loading=False):
        """打印 checkpoint 加载最终汇总。"""
        state_dict_main = checkpoint.get('state_dict', {})
        has_main = bool(state_dict_main and len(state_dict_main) > 0)

        if skip_split_loading:
            print(
                "[ckpt] Load summary: "
                f"main_state_dict={'OK' if has_main else 'MISSING/EMPTY'}, "
                f"full_EMA_restored=True, "
                f"split_modules=SKIPPED (full EMA already covers all)"
            )
        else:
            print(
                "[ckpt] Load summary: "
                f"main_state_dict={'OK' if has_main else 'MISSING/EMPTY'}"
            )

    def configure_optimizers(self):
        if self.is_e2e:
            if self.d_optimizer is not None:
                return [self.optimizer, self.d_optimizer]
            return [self.optimizer]
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [self.scheduler]
