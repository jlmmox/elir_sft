from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from ELIR.models.load_model import get_model
from ELIR.datasets.dataset import get_loader
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
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
    model = get_model(arch_cfg)

    # ----------------------------
    # Evaluation
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    out_dir = eval_cfg.get("out_dir", "./runs/eval")
    save_images = eval_cfg.get("save_images", False)
    os.makedirs(out_dir, exist_ok=True)

    # Use CSVLogger to record metrics; IRSetup will drop images into run_dir when save_images=True.
    logger = CSVLogger(save_dir=out_dir, name="logs")
    setup = IRSetup(model, eval_cfg=eval_cfg, run_dir=out_dir, save_images=save_images)
    trainer = L.Trainer(logger=logger)
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