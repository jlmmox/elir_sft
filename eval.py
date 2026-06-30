from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ELIR.irsetup import IRSetup
import warnings
import os
import csv
warnings.filterwarnings("ignore")


def run_eval(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Prepare datasets
    # ----------------------------
    dataset_cfg = conf.get("dataset_cfg")
    val_dataset = dataset_cfg.get('val_dataset')
    valloader = get_loader(val_dataset)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    eval_cfg = conf.get("eval_cfg", {})
    ckpt_path = eval_cfg.get("ckpt_path") or arch_cfg.get("path")

    out_dir = eval_cfg.get("out_dir", "./runs/eval")
    save_images = eval_cfg.get("save_images", False)
    os.makedirs(out_dir, exist_ok=True)
    logger = CSVLogger(save_dir=out_dir, name="logs")

    fm_cfg = conf.get("fm_cfg", {})
    ema_decay = eval_cfg.get("ema_decay", conf.get("train_cfg", {}).get("ema_decay", 0.999))

    if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
        print(f"[eval] loading checkpoint via load_from_checkpoint: {ckpt_path}")
        model = get_model(arch_cfg)
        setup = IRSetup.load_from_checkpoint(
            os.path.expanduser(ckpt_path),
            model=model,
            fm_cfg=fm_cfg,
            eval_cfg=eval_cfg,
            run_dir=out_dir,
            save_images=save_images,
            ema_decay=ema_decay,
            optimizer=None,
            scheduler=None,
            tmodel=None,
            strict=False,
            map_location="cpu",
        )
        print(f"[eval] EMA exists: {setup.ema is not None}")

        # ---- 同步 ODE 运行时配置到 RAW 和 EMA ----
        k_steps = int(fm_cfg.get("k_steps", getattr(setup.model, "K", 5)))
        flow_infer_scale = float(fm_cfg.get("flow_infer_scale", 1.0))
        inference_noise_scale = float(fm_cfg.get("inference_noise_scale", 0.0))

        print(
            f"[eval ODE sync] k_steps={k_steps}, "
            f"flow_infer_scale={flow_infer_scale}, "
            f"inference_noise_scale={inference_noise_scale}"
        )

        IRSetup.sync_ode_runtime(
            setup.model,
            k_steps,
            flow_infer_scale,
            inference_noise_scale,
        )

        if setup.ema is not None:
            IRSetup.sync_ode_runtime(
                setup.ema.model,
                k_steps,
                flow_infer_scale,
                inference_noise_scale,
            )

        # ---- 断言 ODE 配置一致性 ----
        assert setup.model.K == k_steps, \
            f"RAW model.K={setup.model.K} != k_steps={k_steps}"
        assert abs(setup.model.dt - 1.0 / max(k_steps, 1)) < 1e-12, \
            f"RAW model.dt={setup.model.dt} != 1/{k_steps}"

        if setup.ema is not None:
            assert setup.ema.model.K == k_steps, \
                f"EMA model.K={setup.ema.model.K} != k_steps={k_steps}"
            assert abs(setup.ema.model.dt - 1.0 / max(k_steps, 1)) < 1e-12, \
                f"EMA model.dt={setup.ema.model.dt} != 1/{k_steps}"
            assert setup.ema.model.flow_infer_scale == flow_infer_scale, \
                f"EMA flow_infer_scale={setup.ema.model.flow_infer_scale} != {flow_infer_scale}"
            assert setup.ema.model.inference_noise_scale == inference_noise_scale, \
                f"EMA inference_noise_scale={setup.ema.model.inference_noise_scale} != {inference_noise_scale}"

        print("[eval ODE sync] RAW and EMA K/dt/scale/noise synced and verified.")
    else:
        model = get_model(arch_cfg)
        setup = IRSetup(model, eval_cfg=eval_cfg, run_dir=out_dir, save_images=save_images)

    trainer = L.Trainer(logger=logger)
    # ---- 打印 ModelCheckpoint 配置（eval 不使用 checkpoint callback，此处仅为诊断） ----
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            print(
                "[checkpoint callback] "
                f"monitor={cb.monitor}, "
                f"mode={cb.mode}, "
                f"save_top_k={cb.save_top_k}, "
                f"best_model_score={cb.best_model_score}, "
                f"best_model_path={cb.best_model_path}"
            )
    set_seed(seed)
    results = trainer.validate(setup, dataloaders=valloader)
    metrics = eval_cfg.get("metrics")
    # Print and also save a one-line CSV summary.
    summary_path = os.path.join(out_dir, "metrics_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric in metrics:
            val = results[0][metric]
            print("{}: {:0.4f}".format(metric, val), end=", ")
            writer.writerow([metric, val])
    print(f"\nMetrics saved to {summary_path}. Logs at {logger.log_dir}.")

if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Eval
    # ----------------------------
    run_eval(conf)