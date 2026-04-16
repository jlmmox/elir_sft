from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
import pytorch_lightning as L
import torch
from ELIR.metrics import MetricEval
from ELIR.training.losses import get_loss
from torchvision.utils import save_image
from ELIR.training.ema_timm import ModelEMA
import os
import torch.nn.functional as F
from ELIR.utils import ImageSpliterTh
import math



class IRSetup(L.LightningModule):
    def __init__(self, model, fm_cfg={}, optimizer=None, scheduler=None, tmodel=None,
                 ema_decay=None, eval_cfg=None, run_dir=None, save_images=True):
        super().__init__()
        self.model = model
        self.fm_cfg = fm_cfg
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
        self.metric_evals = [MetricEval(metric, self.acc_device, run_dir) for metric in self.metrics]
        self.train_loss = []
        self.ema = None
        if ema_decay:
            self.ema = ModelEMA(model, device=self.acc_device, decay=ema_decay)
        self.samples_dir = None
        self.max_save_images = 0
        if run_dir and save_images:
            self.samples_dir = os.path.join(run_dir, "samples")
            os.makedirs(self.samples_dir, exist_ok=True)  # run folder
            self.samples = []
            self.max_save_images = int(self.eval_cfg.get("max_save_images", 4))
        self._runtime_space_logged = False

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
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        x_lq, x_hq = batch[0], batch[1]
        if not self._runtime_space_logged:
            runtime_space, is_pixel_space = self._infer_runtime_space(x_lq)
            print(
                f"[RuntimeSpace] mode={runtime_space}, "
                f"x_lq_channels={int(x_lq.shape[1])}, "
                f"has_enc={hasattr(self.model, 'enc') and (self.model.enc is not None)}, "
                f"has_dec={hasattr(self.model, 'dec') and (self.model.dec is not None)}, "
                f"loss_method={self.fm_cfg.get('method')}"
            )
            self.log("runtime/is_pixel_space", float(is_pixel_space), logger=True, prog_bar=True)
            self._runtime_space_logged = True
        # Loss function
        fm_cfg_runtime = dict(self.fm_cfg)
        fm_cfg_runtime["global_step"] = int(self.global_step)
        loss = get_loss(self.model, x_hq, x_lq, fm_cfg_runtime, self.tmodel)

        self.train_loss.append(loss)
        if batch_idx % 5:
            self.log("train_loss", torch.mean(torch.Tensor(self.train_loss)).item(), logger=True, prog_bar=True)
            self.train_loss.clear()
        return loss

    def compute_metrics(self, x_hq_hat, x_hq):
        for metric_eval in self.metric_evals:
            metric_eval.compute(x_hq_hat, x_hq)

    def save_samples(self, current_epoch):
        sample_idx = 0
        for batch_samples in self.samples:
            for img in batch_samples:
                out_path = os.path.join(self.samples_dir, f"epoch_{current_epoch}_{sample_idx:03d}.png")
                save_image(img, out_path)
                sample_idx += 1

    def infer(self, x):
        if self.ema:
            return self.ema.model.inference(x)
        else:
            return self.model.inference(x)

    def validation_step(self, batch, batch_idx):
        x_lq, y = batch
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
        self.compute_metrics(y_hat, y)

    def on_validation_epoch_end(self):
        if self.samples_dir and self.samples:
            self.save_samples(self.current_epoch)
            self.samples.clear()
        self.log('global_step', self.global_step)
        for metric_eval in self.metric_evals:
            result = metric_eval.get_final().item()
            self.log(metric_eval.metric, result, sync_dist=True, prog_bar=True)
        torch.cuda.empty_cache()

    def _save_optional_state(self, checkpoint, key, module):
        if module is not None:
            checkpoint[key] = module.state_dict()

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            self._save_optional_state(checkpoint, 'state_dict_fmir', getattr(self.ema.model, 'fmir', None))
            self._save_optional_state(checkpoint, 'state_dict_mmse', getattr(self.ema.model, 'mmse', None))
            self._save_optional_state(checkpoint, 'state_dict_enc', getattr(self.ema.model, 'enc', None))
            self._save_optional_state(checkpoint, 'state_dict_dec', getattr(self.ema.model, 'dec', None))
            self._save_optional_state(checkpoint, 'state_dict_sft', getattr(self.ema.model, 'sft_refiner', None))
        else:
            self._save_optional_state(checkpoint, 'state_dict_fmir', getattr(self.model, 'fmir', None))
            self._save_optional_state(checkpoint, 'state_dict_mmse', getattr(self.model, 'mmse', None))
            self._save_optional_state(checkpoint, 'state_dict_enc', getattr(self.model, 'enc', None))
            self._save_optional_state(checkpoint, 'state_dict_dec', getattr(self.model, 'dec', None))
            self._save_optional_state(checkpoint, 'state_dict_sft', getattr(self.model, 'sft_refiner', None))
        return checkpoint

    def configure_optimizers(self):
        if self.scheduler is None:
            return [self.optimizer]
        return [self.optimizer], [self.scheduler]
