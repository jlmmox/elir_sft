from args_handler import argument_handler, set_overides
import yaml
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
from ELIR.training.tparmas import get_opt_sched
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from ELIR.irsetup import IRSetup
import os
import torch
import warnings
warnings.filterwarnings("ignore")


def _configure_train_mode(conf):
    """Normalize module trainability based on a single train_mode switch."""
    train_cfg = conf.get("train_cfg", {})
    mode = str(train_cfg.get("train_mode", "manual")).strip().lower()
    if mode in ["", "manual", "none"]:
        return conf

    model_cfg = conf.get("model_cfg", {})
    arch_cfg = model_cfg.get("arch_cfg", {})
    params = arch_cfg.get("params", {})
    fmir_cfg = params.get("fmir_cfg", {})
    mmse_cfg = params.get("mmse_cfg", {})
    enc_cfg = params.get("enc_cfg", {})
    dec_cfg = params.get("dec_cfg", {})
    wavelet_cfg = params.get("wavelet_cfg", {})
    fm_cfg = conf.get("fm_cfg", {})

    if mode in ["stage1", "stage1_backbone", "backbone"]:
        fmir_cfg["trainable"] = True
        mmse_cfg["trainable"] = True
        if isinstance(enc_cfg, dict):
            enc_cfg["trainable"] = False
        if isinstance(dec_cfg, dict):
            dec_cfg["trainable"] = False
            if dec_cfg.get("name") in ("sft_taesd_finetuner", "encoder_skip_fusion", "encoder_skip_wavelet_fusion"):
                dec_cfg["name"] = "taesd"
                dec_cfg.pop("params", None)
                dec_cfg.pop("path", None)
        if isinstance(wavelet_cfg, dict) and len(wavelet_cfg) > 0:
            wavelet_cfg["trainable"] = False

        # Stage1 backbone does not require SFT-specific decode supervision.
        if str(fm_cfg.get("method", "")).strip() in ["", "charbonnier_ssim_cfm_loss"]:
            fm_cfg["method"] = "pixel_space_l2_cfm_loss"

    elif mode in ["sft", "sft_decoder", "stage2"]:
        fmir_cfg["trainable"] = False
        mmse_cfg["trainable"] = False
        if isinstance(enc_cfg, dict):
            enc_cfg["trainable"] = False
        if isinstance(dec_cfg, dict):
            dec_cfg["trainable"] = True
            if dec_cfg.get("name") not in ("sft_taesd_finetuner", "encoder_skip_fusion", "encoder_skip_wavelet_fusion"):
                raise ValueError(
                    "train_mode=sft_decoder requires dec_cfg.name=sft_taesd_finetuner "
                    "or encoder_skip_fusion or encoder_skip_wavelet_fusion, "
                    f"got {dec_cfg.get('name')}"
                )
        if isinstance(wavelet_cfg, dict) and len(wavelet_cfg) > 0:
            wavelet_cfg["trainable"] = wavelet_cfg.get("enabled", True)

        # SFT decoder / encoder skip needs a decode-to-pixel loss term to receive gradients.
        if str(fm_cfg.get("method", "")).strip() in ["", "pixel_space_l2_cfm_loss"]:
            fm_cfg["method"] = "charbonnier_ssim_cfm_loss"

    elif mode in ["e2e", "end_to_end"]:
        fmir_cfg["trainable"] = True
        mmse_cfg["trainable"] = True
        if isinstance(enc_cfg, dict):
            enc_cfg["trainable"] = False
        if isinstance(dec_cfg, dict):
            dec_cfg["trainable"] = True
            if dec_cfg.get("name") not in ("sft_taesd_finetuner",):
                raise ValueError(
                    "train_mode=e2e requires dec_cfg.name=sft_taesd_finetuner "
                    f"(WaveletStem→TanhSFT decoder). Got: {dec_cfg.get('name')}"
                )
        if isinstance(wavelet_cfg, dict) and len(wavelet_cfg) > 0:
            wavelet_cfg["trainable"] = wavelet_cfg.get("enabled", True)
        fm_cfg["method"] = "e2e_gan"
        fm_cfg["detach_latent_path"] = False

    else:
        raise ValueError(
            "Unsupported train_mode. Expected one of: manual, stage1_backbone, sft_decoder, "
            "e2e (end_to_end). "
            f"Got: {mode}"
        )

    params["fmir_cfg"] = fmir_cfg
    params["mmse_cfg"] = mmse_cfg
    params["enc_cfg"] = enc_cfg
    params["dec_cfg"] = dec_cfg
    params["wavelet_cfg"] = wavelet_cfg
    arch_cfg["params"] = params
    model_cfg["arch_cfg"] = arch_cfg
    conf["model_cfg"] = model_cfg
    conf["fm_cfg"] = fm_cfg

    print(
        "[train_mode] mode={}, fmir_trainable={}, mmse_trainable={}, enc_trainable={}, "
        "dec_name={}, dec_trainable={}, loss_method={}".format(
            mode,
            fmir_cfg.get("trainable"),
            mmse_cfg.get("trainable"),
            enc_cfg.get("trainable") if isinstance(enc_cfg, dict) else None,
            dec_cfg.get("name") if isinstance(dec_cfg, dict) else None,
            dec_cfg.get("trainable") if isinstance(dec_cfg, dict) else None,
            fm_cfg.get("method"),
        )
    )
    return conf



def run_train(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Save configuration
    # ----------------------------
    out_dir = env_cfg.get("out_dir")
    os.makedirs(out_dir, exist_ok=True) # out folder
    run_name = env_cfg.get("run_name")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True) # run folder
    conf_path = os.path.join(run_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.dump(conf, outfile)
    print("Configuration: {}", conf)

    # ----------------------------
    # Prepare datasets
    # ----------------------------
    dataset_cfg = conf.get("dataset_cfg")
    train_dataset = dataset_cfg.get('train_dataset')
    trainloader = get_loader(train_dataset)
    val_dataset = dataset_cfg.get('val_dataset')
    valloader = get_loader(val_dataset)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # If ckpt is weights-only (no optimizer state), load weights manually and avoid passing ckpt_path to trainer.
    train_cfg = conf.get("train_cfg")
    ckpt_path = train_cfg.get("ckpt_path", None)
    resume_ckpt = ckpt_path
    if ckpt_path:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            opt_states = ckpt.get("optimizer_states")
            sched_states = ckpt.get("lr_schedulers")
            has_opt_state = opt_states is not None and len(opt_states) > 0
            has_sched_state = sched_states is not None and len(sched_states) > 0
            # Treat checkpoint as weights-only if it lacks optimizer/scheduler state or user explicitly saved weights only.
            if train_cfg.get("save_weights_only", False) or not (has_opt_state or has_sched_state):
                print(f"Checkpoint {ckpt_path} has no optimizer state; loading weights only and starting with a fresh optimizer.")
                model.load_weights(ckpt_path)
                resume_ckpt = None
        except Exception as exc:
            print(f"Could not inspect checkpoint {ckpt_path}: {exc}. Passing it to Trainer as-is.")
            resume_ckpt = ckpt_path

    # Teacher model
    tmodel_cfg = model_cfg.get("teacher_cfg", None)
    tmodel = get_model(tmodel_cfg) if tmodel_cfg is not None else None

    # ----------------------------
    # Training
    # ----------------------------
    fm_cfg = conf.get("fm_cfg",{})
    train_cfg = conf.get("train_cfg")
    optimizer, scheduler = get_opt_sched(train_cfg, model)
    eval_cfg = conf.get("eval_cfg")

    # Loggers
    logger = False
    wandbLogger = None
    if train_cfg.get("wandb",False):
        import wandb
        print("WandB is enable!")
        wandb.init(project=env_cfg.get("project_name"), dir=run_dir, group=run_name)
        wandbLogger = WandbLogger(project=env_cfg.get("project_name"), dir=run_dir)
        wandb.log(dict(**conf))
        logger = wandbLogger
    else:
        log_type = train_cfg.get("logger", "csv")
        if log_type == "csv":
            logger = CSVLogger(save_dir=run_dir, name="logs")
        elif log_type == "tensorboard":
            logger = TensorBoardLogger(save_dir=run_dir, name="logs")
        elif log_type in [None, "none", False]:
            logger = False

    # Training
    train_setup = IRSetup(model,
                         fm_cfg=fm_cfg,
                         tmodel=tmodel,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         ema_decay=train_cfg.get("ema_decay", 0.999),
                         eval_cfg=eval_cfg,
                         run_dir=run_dir,
                         save_images=train_cfg.get("save_images", True))
    checkpoint = ModelCheckpoint(run_dir,
                                 monitor="psnr",
                                 mode="max",
                                 every_n_epochs=1,
                                 save_weights_only=train_cfg.get("save_weights_only", False),
                                 save_top_k=1,
                                 save_last=True,
                                 enable_version_counter=False,
                                 save_on_train_epoch_end=True, verbose=False)

    strategy_cfg = train_cfg.get("strategy", "ddp")
    if str(strategy_cfg).lower() == "ddp" and bool(train_cfg.get("find_unused_parameters", False)):
        strategy_cfg = DDPStrategy(find_unused_parameters=True)

    trainer = L.Trainer(max_epochs=train_cfg.get("epochs"),
                        default_root_dir = run_dir,
                        callbacks=checkpoint,
                        # allow overriding strategy/devices/accelerator via yaml or CLI
                        strategy = strategy_cfg,
                        devices = train_cfg.get("devices", "auto"),
                        accelerator=train_cfg.get("accelerator", "gpu"),
                        precision=train_cfg.get("precision", "32-true"),
                        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
                        logger = logger,
                        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps",0),
                        check_val_every_n_epoch = train_cfg.get("check_val_every_n_epoch",1),
                        max_steps = train_cfg.get("max_steps", -1))
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    set_seed(seed)
    trainer.fit(train_setup, trainloader, valloader, ckpt_path=resume_ckpt)

    # ----------------------------
    # Evaluation
    # ----------------------------
    results = trainer.validate(dataloaders=valloader, ckpt_path="last")
    metrics = eval_cfg.get("metrics")
    for metric in metrics:
        metric_value = results[0][metric]
        print("{}: {:0.4f}".format(metric, metric_value), end =", ")
        if train_cfg.get("wandb", False):
            wandb.log({"final_"+metric: metric_value})

    # ----------------------------
    # Save model
    # ----------------------------
    def adjust_weights(sd):
        sd = dict((key.replace("model.",""), value) for (key, value) in sd.items())
        return sd

    # Save latest model (EMA if enabled, otherwise current training model)
    save_model = train_setup.ema.model if train_setup.ema is not None else train_setup.model
    if hasattr(save_model, "collapse"):
        save_model.collapse()
    state_dict = save_model.state_dict()
    state_dict = adjust_weights(state_dict)
    torch.save(state_dict, os.path.join(run_dir, "elir.pth"))



if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)
    conf = _configure_train_mode(conf)

    # ----------------------------
    # Train
    # ----------------------------
    run_train(conf)