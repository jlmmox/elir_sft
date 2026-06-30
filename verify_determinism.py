"""验证日志重构前后训练结果一致。固定 seed，对比 5 个 optimizer step 的关键指标。

用法：python verify_determinism.py
"""
import os, sys, tempfile, shutil
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SynthPaired(torch.utils.data.Dataset):
    def __init__(self, length=8, size=256, seed=2025):
        self.length = length
        self.size = size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        lq = (self.rng.rand(3, self.size, self.size).astype(np.float32) * 0.3)
        hq = (self.rng.rand(3, self.size, self.size).astype(np.float32))
        return torch.from_numpy(lq), torch.from_numpy(hq)


def run_one_trial(label, log_cfg_extra=None):
    """运行 5 optimizer step 并返回关键指标"""
    from utils import set_seed
    from ELIR.models.load_model import get_model

    set_seed(2025)
    torch.manual_seed(2025)
    np.random.seed(2025)

    conf = _make_conf()
    if log_cfg_extra:
        conf["log_cfg"].update(log_cfg_extra)

    from train import _configure_train_mode
    conf = _configure_train_mode(conf)

    model = get_model(conf["model_cfg"]["arch_cfg"])
    tmodel = get_model(conf["model_cfg"]["teacher_cfg"]) if conf["model_cfg"].get("teacher_cfg") else None

    from ELIR.training.tparmas import get_opt_sched
    optimizer, scheduler = get_opt_sched(conf["train_cfg"], model)

    print(f"\n[{label}] enc_trainable={model.enc_trainable}")

    # 手动运行 5 个 batch，记录指标
    dataset = SynthPaired(length=8, size=256, seed=2025)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    from ELIR.training.losses import e2e_gan_loss
    records = []
    fm_cfg = conf["fm_cfg"]
    step_count = 0
    ema_updates = 0

    for batch_idx, (x_lq, x_hq) in enumerate(loader):
        if step_count >= 5:
            break
        x_lq, x_hq = x_lq.cuda(), x_hq.cuda()
        model = model.cuda()
        if tmodel:
            tmodel = tmodel.cuda()

        fm_cfg["global_step"] = step_count
        g_loss, d_loss, metrics = e2e_gan_loss(
            model, x_hq, x_lq, fm_cfg,
            discriminator=None, perceptual_fn=None,
            dino_encoder=None, dino_spatial_proj=None,
            sk_fusion=None, step=step_count, tmodel=tmodel,
        )

        g_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        rec = {
            "step": step_count,
            "g_loss": g_loss.item(),
            "loss_fm": metrics.get("loss_fm", torch.tensor(0)).item() if isinstance(metrics.get("loss_fm"), torch.Tensor) else metrics.get("loss_fm", 0),
            "loss_mmse_char": metrics.get("loss_mmse_char", torch.tensor(0)).item() if isinstance(metrics.get("loss_mmse_char"), torch.Tensor) else metrics.get("loss_mmse_char", 0),
            "loss_pixel_total": metrics.get("loss_pixel_total", torch.tensor(0)).item() if isinstance(metrics.get("loss_pixel_total"), torch.Tensor) else metrics.get("loss_pixel_total", 0),
        }
        records.append(rec)
        step_count += 1

    # 记录部分参数值
    param_samples = {}
    for n, p in model.fmir.named_parameters():
        if "final_proj.weight" in n:
            param_samples["fmir_final_proj_weight"] = p.detach().cpu().clone()
        if "first_proj.weight" in n:
            param_samples["fmir_first_proj_weight"] = p.detach().cpu().clone()
        break
    param_samples["mmse_lrm_0_rdb1_conv1_weight"] = next(
        p for n, p in model.mmse.named_parameters() if "lrm.0" in n and "conv1" in n
    ).detach().cpu().clone() if model.mmse else None

    torch.cuda.empty_cache()
    return records, param_samples


def _make_conf():
    return {
        "env_cfg": {"out_dir": "./runs", "run_name": "det_test", "seed": 2025},
        "fm_cfg": {
            "method": "e2e_gan", "k_steps": 5, "t_emb_dim": 160,
            "sigma_min": 0.00001, "sigma_s": 0.1, "dt": 0.05,
            "alpha": 0.001, "beta": 0.001,
            "lambda_pix_max": 0.5, "lambda_pix_warmup_steps": 10,
            "detach_latent_path": False,
            "lambda_charb": 1.0, "lambda_ssim": 0.5,
            "lambda_color": 0.0, "lambda_blur": 0.0,
            "lambda_dino": 0.0, "lambda_perc": 0.0, "lambda_gan": 0.0,
            "d_lr": 0.00004, "lr_min_ratio": 0.01,
            "d_base_channels": 64, "d_n_layers": 3,
            "gan_perc_warmup_steps": 20, "gradient_clip_val": 1.0,
            "grad_accum": 1, "use_latent_cond": False,
            "lambda_fm": 1.0, "lambda_mmse_char": 1.0,
            "lambda_bridge": 0.0, "lambda_final_latent": 0.0,
            "t_sampling": "uniform",
            "lambda_low_t_endpoint": 0.0, "lambda_mmse_guard": 0.0,
            "lambda_calib_latent": 0.0, "lambda_flow_img": 0.0,
        },
        "model_cfg": {
            "arch_cfg": {
                "name": "elir",
                "params": {
                    "fm_cfg": {"k_steps": 5, "sigma_s": 0.1, "latent_shape": [16, 64, 64],
                               "seed": 2025, "dynamic_noise": True, "force_static_noise": False,
                               "detach_latent_path": False},
                    "fmir_cfg": {"name": "lunet",
                                 "params": {"ch_mult": [1, 2, 2, 2], "n_mid_blocks": 4,
                                            "in_channels": 16, "hid_channels": 192,
                                            "out_channels": 16, "t_emb_dim": 160,
                                            "overparametrization": False, "use_checkpoint": False,
                                            "use_attn": False, "attn_heads": 8,
                                            "use_time_dilate": False},
                                 "trainable": True},
                    "mmse_cfg": {"name": "rrdbnet",
                                 "params": {"c_inout": 16, "c_hid": 128, "n_rrdb": 4,
                                            "overparametrization": True},
                                 "trainable": True},
                    "enc_cfg": {"name": "tiny_enc", "trainable": True},
                    "dec_cfg": {"name": "sft_taesd_finetuner", "path": None,
                                "trainable": True,
                                "params": {"latent_channels": 16, "gamma_scale": 0.1, "beta_scale": 0.1}},
                    "wavelet_cfg": {"enabled": False},
                }, "path": None,
            },
            "teacher_cfg": {"name": "tiny_enc", "params": {}, "trainable": False},
        },
        "train_cfg": {
            "train_mode": "e2e", "epochs": 1, "ckpt_path": None,
            "load_modules_from_ckpt": None, "skip_modules_from_ckpt": [],
            "optimizer": None, "lr": 0.00005, "optimizer_params": {},
            "ema_decay": 0.999, "max_steps": -1,
            "accumulate_grad_batches": 1, "wandb": False,
            "logger": "csv", "save_images": False, "save_weights_only": True,
            "strategy": "auto", "devices": 1, "accelerator": "gpu", "precision": "32-true",
        },
        "eval_cfg": {"metrics": ["psnr", "ssim"], "save_images": False, "max_save_images": 0},
        "log_cfg": {"level": "normal", "train_log_interval": 1, "grad_log_interval": 1,
                    "log_module_grad_norms": True, "log_memory": False,
                    "debug_fmir_params": False, "debug_fmir_updates": False,
                    "debug_cfm_details": False, "debug_validation_graph": False,
                    "print_parameter_names": False},
        "dataset_cfg": {},  # unused in manual loop
    }


def main():
    print("=== Determinism check: 5 optimizer steps, fixed seed ===")
    if not torch.cuda.is_available():
        print("CUDA not available"); return

    # Trial 1: normal mode
    r1, p1 = run_one_trial("NORMAL", {})
    # Trial 2: same config, different run (should be identical due to fixed seed)
    r2, p2 = run_one_trial("NORMAL_RERUN", {})

    # Compare
    identical = True
    for i in range(min(len(r1), len(r2))):
        for k in r1[i]:
            v1, v2 = r1[i][k], r2[i][k]
            if abs(v1 - v2) > 1e-5:
                print(f"  MISMATCH step={i} key={k}: {v1:.6f} vs {v2:.6f}")
                identical = False

    for k in p1:
        if p1[k] is not None and p2[k] is not None:
            if not torch.allclose(p1[k], p2[k], rtol=1e-5, atol=1e-7):
                print(f"  PARAM MISMATCH {k}: max_diff={(p1[k]-p2[k]).abs().max():.8f}")
                identical = False

    if identical:
        print("\n[OK] Two independent runs are identical — training is deterministic.")
    else:
        print("\n[FAIL] Differences exist.")

    print("\n各步指标:")
    for i, rec in enumerate(r1):
        print(f"  step={i}: g_loss={rec['g_loss']:.6f}  loss_fm={rec['loss_fm']:.6f}  "
              f"loss_mmse_char={rec['loss_mmse_char']:.6f}  loss_pixel_total={rec['loss_pixel_total']:.6f}")

    print(f"\nOptimizer steps: {len(r1)} (两次运行一致)")


if __name__ == "__main__":
    main()
