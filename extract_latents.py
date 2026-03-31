import argparse
import os

import torch
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm

from ELIR.datasets.dataset import get_loader


class _PermissiveLoader(SafeLoader):
    pass


def _construct_unknown_tag(loader, tag_suffix, node):
    # Keep only the underlying value and ignore custom tags (e.g. !name:torch.optim.AdamW).
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return None


_PermissiveLoader.add_multi_constructor("!", _construct_unknown_tag)


def _load_dataset_cfg(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_PermissiveLoader)
    if "dataset_cfg" not in cfg:
        raise KeyError("dataset_cfg not found in config")
    return cfg["dataset_cfg"]


def _extract_from_loader(loader, output_root, vae, global_idx):
    output_root = os.path.abspath(os.path.expanduser(output_root))
    latents_hq_dir = os.path.join(output_root, "latents_hq")
    latents_lq_dir = os.path.join(output_root, "latents_lq")
    os.makedirs(latents_hq_dir, exist_ok=True)
    os.makedirs(latents_lq_dir, exist_ok=True)

    # Keep progress line compact to avoid wrapped multi-line redraw in narrow IDE consoles.
    split_tag = os.path.basename(os.path.normpath(output_root)) or "dataset"
    pbar = tqdm(
        loader,
        desc=f"Extract:{split_tag}",
        leave=False,
        dynamic_ncols=True,
        ncols=100,
        mininterval=0.2,
    )
    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_lq, x_hq = batch[0], batch[1]
        elif isinstance(batch, dict):
            if "x_lq" in batch and "x_hq" in batch:
                x_lq, x_hq = batch["x_lq"], batch["x_hq"]
            else:
                raise KeyError("Batch dict must contain x_lq and x_hq")
        else:
            raise TypeError("Unsupported batch format; expected (x_lq, x_hq) or dict with x_lq/x_hq")

        x_lq = x_lq.to("cuda", non_blocking=True)
        x_hq = x_hq.to("cuda", non_blocking=True)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                z_lq = vae.encoder(x_lq)
                z_hq = vae.encoder(x_hq)

        for i in range(z_lq.shape[0]):
            filename = f"{global_idx:07d}.pt"
            lq_path = os.path.join(latents_lq_dir, filename)
            hq_path = os.path.join(latents_hq_dir, filename)
            torch.save(z_lq[i].clone().cpu(), lq_path)
            torch.save(z_hq[i].clone().cpu(), hq_path)
            global_idx += 1

    return global_idx


def main(config_path, pretrained_path=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this extraction script.")

    dataset_cfg = _load_dataset_cfg(config_path)

    from ELIR.models.taesd import TAESD
    vae = TAESD(pretrained=True)  # 自动加载预训练权重
    vae = vae.to("cuda")
    vae.eval()
    vae = vae.to(torch.float16)

    global_idx = 0
    for split_name in ("train_dataset", "val_dataset"):
        if split_name not in dataset_cfg:
            continue

        ds_params = dict(dataset_cfg[split_name])
        ds_params["training"] = False
        ds_params["batch_size"] = 16
        # Extraction must read source images, not latent cache files.
        ds_params["use_latent_cache"] = False

        output_root = ds_params.get("path")
        if output_root is None:
            raise KeyError(f"{split_name} is missing path")
        output_root = os.path.abspath(os.path.expanduser(output_root))
        ds_params["path"] = output_root

        loader = get_loader(ds_params)
        global_idx = _extract_from_loader(loader, output_root, vae, global_idx)

    print(f"Finished latent extraction. Total samples saved: {global_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline latent extraction using TAESD (auto-loads pretrained weights)")
    parser.add_argument("--config", "-y", type=str, default="configs/elir_train_bfr.yaml", help="Path to training yaml config")
    args = parser.parse_args()

    main(config_path=args.config)
