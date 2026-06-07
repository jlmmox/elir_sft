"""输出流匹配 ODE 的每一步中间结果。

用法:
  python infer_trajectory.py -y configs/elir_infer_e2e_small_lol.yaml --lq_path <lq.png> --hq_path <hq.png> --out_dir ./traj_out
"""

import os
import sys
import torch
import argparse
from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from torchvision.utils import save_image


def save_trajectory(model, lq_path, hq_path, out_dir, k_steps=None):
    from PIL import Image
    from torchvision.transforms import v2

    os.makedirs(out_dir, exist_ok=True)

    to_tensor = v2.ToTensor()
    lq = to_tensor(Image.open(lq_path).convert("RGB")).unsqueeze(0)
    hq = to_tensor(Image.open(hq_path).convert("RGB")).unsqueeze(0)

    # 跟训练/eval 一样对 LQ 做 YCrCb 直方图均衡
    from ELIR.datasets.lol import LOLDataset
    dummy = LOLDataset.__new__(LOLDataset)
    lq_eq = dummy._equalize_low_light(Image.open(lq_path).convert("RGB"))
    lq = to_tensor(lq_eq).unsqueeze(0)

    model.eval()
    model.to(lq.device)

    # 确保 K 步数正确
    if k_steps is not None:
        original_k = model.K
        original_dt = model.dt
        model.K = k_steps
        model.dt = 1.0 / k_steps

    with torch.no_grad():
        trajs_pixel = model.trajectories_pixel(lq)

    if k_steps is not None:
        model.K = original_k
        model.dt = original_dt

    # 保存
    save_image(lq, os.path.join(out_dir, "00_lq.png"))
    save_image(hq, os.path.join(out_dir, f"{len(trajs_pixel)+1:02d}_hq.png"))
    for i, img in enumerate(trajs_pixel):
        img_clamped = torch.clamp(img, 0, 1)
        save_image(img_clamped, os.path.join(out_dir, f"{i+1:02d}_step.png"))

    print(f"Saved {len(trajs_pixel)} trajectory images to {out_dir}/")
    print(f"Files: 00_lq.png → 01_step.png ... {len(trajs_pixel):02d}_step.png → {len(trajs_pixel)+1:02d}_hq.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_path", type=str, required=True)
    parser.add_argument("--lq_path", type=str, required=True)
    parser.add_argument("--hq_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./traj_out")
    parser.add_argument("--k_steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        conf = load_hyperpyyaml(f)

    arch_cfg = conf["model_cfg"]["arch_cfg"]
    model = get_model(arch_cfg)
    save_trajectory(model, args.lq_path, args.hq_path, args.out_dir, args.k_steps)


if __name__ == "__main__":
    main()
