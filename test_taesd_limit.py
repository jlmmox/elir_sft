"""Measure TAESD reconstruction ceiling on clean validation images."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TAESD reconstruction limit.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/moxt/DiffUIR/Datasets/Restoration/RESIDE/SOTS/outdoor/test/val/clear",
        help="Directory containing clean validation images.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random images to evaluate.")
    parser.add_argument("--crop-size", type=int, default=256, help="CenterCrop size.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for sampling.")
    parser.add_argument(
        "--taesd-ckpt",
        type=str,
        default=None,
        help="Optional local TAESD checkpoint path. If omitted, use the pretrained madebyollin/taesd3 weights.",
    )
    return parser.parse_args()


def build_taesd(ckpt_path: str | None):
    from ELIR.models.taesd import TAESD

    if ckpt_path:
        model = TAESD(pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict_enc" in state_dict and "state_dict_dec" in state_dict:
            model.encoder.load_state_dict(state_dict["state_dict_enc"])
            model.decoder.load_state_dict(state_dict["state_dict_dec"])
        else:
            model.load_state_dict(state_dict)
    else:
        model = TAESD(pretrained=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this TAESD limit test.")

    device = torch.device("cuda")
    model = model.to(device).eval()
    enc = model.encoder.to(device).eval()
    dec = model.decoder.to(device).eval()
    return enc, dec, device


def list_images(data_root: str) -> list[Path]:
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No image files found under: {root}")
    return files


def load_image(path: Path, crop_size: int) -> torch.Tensor:
    transform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()])
    with Image.open(path) as image:
        image = image.convert("RGB")
        return transform(image)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="mean")
    return 10.0 * torch.log10((data_range * data_range) / torch.clamp(mse, min=1e-12))


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM expects identical shapes, got {pred.shape} and {target.shape}")

    channel = pred.shape[1]
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    padding = window_size // 2
    mu_x = F.conv2d(pred, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target, window, padding=padding, groups=channel)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-12)
    return ssim_map.mean()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    enc, dec, device = build_taesd(args.taesd_ckpt)
    image_paths = list_images(args.data_root)
    sample_count = min(args.num_samples, len(image_paths))
    sampled_paths = random.sample(image_paths, sample_count)

    all_psnr = []
    all_ssim = []

    for path in sampled_paths:
        x_hq = load_image(path, args.crop_size).unsqueeze(0).to(device=device, dtype=torch.float32)

        latent = enc(x_hq)
        x_recon = dec(latent)
        x_recon = torch.clamp(x_recon, 0.0, 1.0)

        all_psnr.append(psnr(x_recon, x_hq).item())
        all_ssim.append(ssim(x_recon, x_hq).item())

    mean_psnr = sum(all_psnr) / len(all_psnr)
    mean_ssim = sum(all_ssim) / len(all_ssim)

    print("\n=== TAESD 物理重建上限 ===")
    print(f"Device: {device}")
    print(f"Samples: {len(sampled_paths)}")
    print(f"Mean PSNR: {mean_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_ssim:.6f}")


if __name__ == "__main__":
    main()
