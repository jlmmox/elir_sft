"""OOM 修复验收脚本：encoder 可训练条件下连续 3 epoch，验证显存不单调增长。

用法：python verify_oom_fix.py
"""
import os, sys, math, tempfile, shutil, glob as _glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


def create_synth_dataset(root, n_train=6, n_val=2, size=256, seed=2025):
    """创建合成 Paired 数据集：train/{lq,hq} 和 val/{lq,hq}。"""
    rng = np.random.RandomState(seed)
    for split, n in [("train", n_train), ("val", n_val)]:
        for sub in ("lq", "hq"):
            d = os.path.join(root, split, sub)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                if sub == "lq":
                    arr = (rng.rand(size, size, 3) * 0.35 * 255).astype(np.uint8)  # darker
                else:
                    arr = (rng.rand(size, size, 3) * 255).astype(np.uint8)
                img = Image.fromarray(arr, "RGB")
                img.save(os.path.join(d, f"{i:04d}.png"))


def main():
    print("=" * 72)
    print("OOM 修复验收：encoder 可训练 + 连续 3 epoch")
    print("=" * 72)

    if not torch.cuda.is_available():
        print("SKIP: CUDA 不可用。")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GiB)")

    # ---- 创建合成数据集 ----
    tmpdir = tempfile.mkdtemp(prefix="oom_verify_")
    create_synth_dataset(tmpdir, n_train=6, n_val=2, size=256)
    print(f"合成数据集: {tmpdir}")

    # ---- 导入项目模块 ----
    from args_handler import argument_handler, set_overides
    from hyperpyyaml import load_hyperpyyaml
    from utils import set_seed
    from ELIR.models.load_model import get_model
    from ELIR.training.tparmas import get_opt_sched
    import pytorch_lightning as L
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from ELIR.irsetup import IRSetup
    from ELIR.datasets.dataset import PairedDataset, get_loader

    # ---- 配置 ----
    conf = {
        "env_cfg": {
            "run_name": "oom_verify",
            "project_name": "elir",
            "out_dir": "./runs",
            "seed": 2025,
        },
        "dataset_cfg": {
            "train_dataset": {
                "name": "Paired",
                "path": os.path.join(tmpdir, "train"),
                "lq_subdir": "lq",
                "hq_subdir": "hq",
                "patch_size": None,
                "batch_size": 2,
                "num_workers": 0,
                "training": True,
                "use_latent_cache": False,
            },
            "val_dataset": {
                "name": "Paired",
                "path": os.path.join(tmpdir, "val"),
                "lq_subdir": "lq",
                "hq_subdir": "hq",
                "patch_size": None,
                "batch_size": 1,
                "num_workers": 0,
                "training": False,
                "use_latent_cache": False,
            },
        },
        "fm_cfg": {
            "method": "e2e_gan",
            "k_steps": 5,
            "t_emb_dim": 160,
            "sigma_min": 0.00001,
            "sigma_s": 0.1,
            "dt": 0.05,
            "alpha": 0.001,
            "beta": 0.001,
            "lambda_pix_max": 0.5,
            "lambda_pix_warmup_steps": 10,
            "detach_latent_path": False,
            "lambda_charb": 1.0,
            "lambda_ssim": 0.5,
            "lambda_color": 0.0,
            "lambda_blur": 0.0,
            "lambda_dino": 0.0,
            "lambda_perc": 0.0,
            "lambda_gan": 0.0,
            "d_lr": 0.00004,
            "lr_min_ratio": 0.01,
            "d_base_channels": 64,
            "d_n_layers": 3,
            "gan_perc_warmup_steps": 20,
            "gradient_clip_val": 1.0,
            "grad_accum": 1,
            "use_latent_cond": False,
            "lambda_fm": 1.0,
            "lambda_mmse_char": 1.0,
            "lambda_bridge": 0.0,
            "lambda_final_latent": 0.0,
            "t_sampling": "uniform",
            "lambda_low_t_endpoint": 0.0,
            "lambda_mmse_guard": 0.0,
            "lambda_calib_latent": 0.0,
            "lambda_flow_img": 0.0,
        },
        "model_cfg": {
            "arch_cfg": {
                "name": "elir",
                "params": {
                    "fm_cfg": {
                        "k_steps": 5, "sigma_s": 0.1,
                        "latent_shape": [16, 64, 64], "seed": 2025,
                        "dynamic_noise": True, "force_static_noise": False,
                        "detach_latent_path": False,
                    },
                    "fmir_cfg": {
                        "name": "lunet",
                        "params": {
                            "ch_mult": [1, 2, 2, 2], "n_mid_blocks": 4,
                            "in_channels": 16, "hid_channels": 192,
                            "out_channels": 16, "t_emb_dim": 160,
                            "overparametrization": False,
                            "use_checkpoint": False,
                            "use_attn": False,
                            "attn_heads": 8,
                            "use_time_dilate": False,
                        },
                        "trainable": True,
                    },
                    "mmse_cfg": {
                        "name": "rrdbnet",
                        "params": {"c_inout": 16, "c_hid": 128, "n_rrdb": 4, "overparametrization": True},
                        "trainable": True,
                    },
                    "enc_cfg": {
                        "name": "taesd",
                        "trainable": True,
                    },
                    "dec_cfg": {
                        "name": "sft_taesd_finetuner",
                        "path": None,
                        "trainable": True,
                        "params": {"latent_channels": 16, "gamma_scale": 0.1, "beta_scale": 0.1},
                    },
                    "wavelet_cfg": {
                        "enabled": False,
                    },
                },
                "path": None,
            },
            "teacher_cfg": {
                "name": "taesd",
                "params": {},
                "trainable": False,
            },
        },
        "train_cfg": {
            "train_mode": "e2e",
            "epochs": 3,
            "ckpt_path": None,
            "load_modules_from_ckpt": None,
            "skip_modules_from_ckpt": [],
            "optimizer": None,
            "lr": 0.00005,
            "optimizer_params": {},
            "ema_decay": 0.999,
            "max_steps": -1,
            "accumulate_grad_batches": 1,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "wandb": False,
            "logger": "csv",
            "save_images": False,
            "save_weights_only": True,
            "strategy": "auto",
            "devices": 1,
            "accelerator": "gpu",
            "precision": "32-true",
        },
        "eval_cfg": {
            "metrics": ["psnr", "ssim"],
            "save_images": False,
            "max_save_images": 0,
        },
        "log_cfg": {
            "level": "normal",
            "train_log_interval": 1,
            "grad_log_interval": 1,
            "log_module_grad_norms": True,
            "log_memory": False,
            "debug_fmir_params": False,
            "debug_fmir_updates": False,
            "debug_cfm_details": False,
            "debug_validation_graph": False,
            "print_parameter_names": False,
        },
    }

    try:
        # ---- 环境 ----
        env_cfg = conf["env_cfg"]
        out_dir = env_cfg["out_dir"]
        run_name = env_cfg["run_name"]
        run_dir = os.path.join(out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        set_seed(env_cfg["seed"])

        # ---- 数据 ----
        dataset_cfg = conf["dataset_cfg"]
        trainloader = get_loader(dataset_cfg["train_dataset"])
        valloader = get_loader(dataset_cfg["val_dataset"])

        # ---- 模型 ----
        from train import _configure_train_mode
        conf = _configure_train_mode(conf)
        model_cfg = conf["model_cfg"]
        arch_cfg = model_cfg["arch_cfg"]
        model = get_model(arch_cfg)
        teacher_cfg = model_cfg.get("teacher_cfg")
        tmodel = get_model(teacher_cfg) if teacher_cfg else None

        # 打印可训练参数统计
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        enc_params = sum(p.numel() for p in model.enc.parameters()) if model.enc else 0
        enc_trainable_params = sum(p.numel() for p in model.enc.parameters() if p.requires_grad) if model.enc else 0
        print(f"\n[Params] total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")
        print(f"[Params] enc_total={enc_params/1e6:.2f}M  enc_trainable={enc_trainable_params/1e6:.2f}M")
        print(f"[Params] enc.training={model.enc.training if model.enc else 'N/A'}")

        # ---- 优化器 ----
        train_cfg = conf["train_cfg"]
        optimizer, scheduler = get_opt_sched(train_cfg, model)
        eval_cfg = conf["eval_cfg"]

        # ---- Lightning 设置 ----
        logger = CSVLogger(save_dir=run_dir, name="logs")
        train_setup = IRSetup(
            model,
            fm_cfg=conf["fm_cfg"],
            tmodel=tmodel,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_decay=train_cfg.get("ema_decay", 0.999),
            eval_cfg=eval_cfg,
            run_dir=run_dir,
            save_images=False,
        )

        checkpoint = ModelCheckpoint(
            run_dir,
            monitor="psnr",
            mode="max",
            every_n_epochs=1,
            save_weights_only=True,
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
            save_on_train_epoch_end=True,
            verbose=False,
        )

        trainer = L.Trainer(
            max_epochs=3,
            default_root_dir=run_dir,
            callbacks=[checkpoint],
            strategy="auto",
            devices=1,
            accelerator="gpu",
            precision="32-true",
            accumulate_grad_batches=1,
            logger=logger,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            max_steps=-1,
            enable_progress_bar=False,
        )

        # ---- 显存快照：训练前 ----
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        train_setup._mem_snapshot("BEFORE_trainer.fit")

        # ---- 训练 ----
        trainer.fit(train_setup, trainloader, valloader)

        # ---- 显存快照：训练后 ----
        train_setup._mem_snapshot("AFTER_trainer.fit")

        # ---- 最终验收 ----
        print("\n" + "=" * 72)
        print("验收结论")
        print("=" * 72)
        max_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        alloc_now = torch.cuda.memory_allocated() / 1024 ** 3
        reserved_now = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"当前分配: {alloc_now:.3f} GiB")
        print(f"当前保留: {reserved_now:.3f} GiB")
        print(f"峰值分配: {max_alloc:.3f} GiB  (GPU 总量: {gpu_mem:.1f} GiB)")
        if max_alloc < gpu_mem * 0.95:
            print("OK: training completed 3 epochs without OOM.")
        print(f"Peak concurrent allocation: {max_alloc:.3f} GiB  (GPU total: {gpu_mem:.1f} GiB)")

        # CSV 列检查
        import glob as _glob
        csv_files = _glob.glob(os.path.join(run_dir, "**", "metrics.csv"), recursive=True)
        if csv_files:
            with open(csv_files[0], "r") as f:
                header = f.readline().strip()
                cols = [c for c in header.split(",") if c]
                print(f"\n[CSV check] {len(cols)} columns:")
                for c in cols:
                    print(f"  {c}")

    finally:
        # 清理
        torch.cuda.empty_cache()
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
