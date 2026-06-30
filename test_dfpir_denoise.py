"""
DFPIR-style Denoising Evaluation Sweep
======================================
Faithfully reproduces the DFPIR GitHub evaluation protocol:

  For each test dataset (CBSD68, Urban100):
    For each sigma ∈ {15, 25, 50}:
      1. Load clean image
      2. Generate Gaussian noise online:  noisy = clean + randn * sigma / 255
      3. Forward through model → denoised
      4. Compute PSNR / SSIM (denoised vs clean)
      5. Average over all images → report

Usage:
  python test_dfpir_denoise.py \
      --cbsd68_path   /data/CBSD68 \
      --urban100_path /data/Urban100 \
      --ckpt          ./runs/elir_denoise/last.ckpt \
      --gpu 0

Output:
  ===========  FINAL RESULTS  ===========
     Dataset     σ    PSNR     SSIM
     -------  ----  ------  ------
     CBSD68     15   34.xx   0.93xx
     CBSD68     25   31.xx   0.88xx
     CBSD68     50   28.xx   0.80xx
     Urban100   15   33.xx   0.92xx
     Urban100   25   30.xx   0.86xx
     Urban100   50   27.xx   0.78xx
  =======================================
"""

import os, sys, glob, math, argparse
from typing import List
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import ELIR model loading
from ELIR.models.load_model import get_model


# ============================================================================
# Test Dataset: clean images, one sigma, online noise per image
# ============================================================================

class DenoiseTestSet(Dataset):
    def __init__(self, root: str, sigma: float):
        self.sigma = sigma
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG"]
        self.paths = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        clean = torch.from_numpy(arr).permute(2, 0, 1).float()
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = (clean + noise).clamp(0.0, 1.0)
        fname = os.path.splitext(os.path.basename(path))[0]
        return noisy, clean, fname


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    return float("inf") if mse == 0 else float(20 * math.log10(1.0) - 10 * math.log10(mse.item()))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 1.0, window_size: int = 11) -> float:
    """Single-image SSIM (3×H×W)."""
    C, H, W = pred.shape
    pad = window_size // 2
    sigma_g = 1.5
    gauss = torch.exp(-0.5 * ((torch.arange(window_size, dtype=pred.dtype, device=pred.device) - pad) / sigma_g) ** 2)
    gauss = gauss / gauss.sum()
    w1d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    window = w1d.unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)

    p4 = pred.unsqueeze(0); t4 = target.unsqueeze(0)
    mu1 = F.conv2d(p4, window, padding=pad, groups=C)
    mu2 = F.conv2d(t4, window, padding=pad, groups=C)
    mu1_sq = mu1 ** 2; mu2_sq = mu2 ** 2; mu12 = mu1 * mu2
    s1_sq = F.conv2d(p4 * p4, window, padding=pad, groups=C) - mu1_sq
    s2_sq = F.conv2d(t4 * t4, window, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(p4 * t4, window, padding=pad, groups=C) - mu12
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2) + 1e-8)
    return float(ssim_map.mean().item())


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_one(model, root: str, sigma: float, device):
    ds = DenoiseTestSet(root, sigma)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"  σ={sigma:.0f}", unit="img", leave=False)

    for noisy, clean, fname in pbar:
        noisy, clean = noisy.to(device), clean.to(device)

        # Pad to multiple of 64 (TAESD encoder requirement)
        _, _, h, w = noisy.shape
        ph = (64 - h % 64) % 64
        pw = (64 - w % 64) % 64
        if ph or pw:
            noisy_p = F.pad(noisy, (0, pw, 0, ph), mode="reflect")
        else:
            noisy_p = noisy

        pred_p = model.inference(noisy_p)
        pred = pred_p[:, :, :h, :w]

        psnr_sum += compute_psnr(pred[0], clean[0])
        ssim_sum += compute_ssim(pred[0], clean[0])
        count += 1
        pbar.set_postfix(PSNR=f"{psnr_sum / count:.2f}", SSIM=f"{ssim_sum / count:.4f}")

    return psnr_sum / count, ssim_sum / count


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DFPIR Denoising Evaluation Sweep")
    parser.add_argument("--cbsd68_path", type=str, required=True,
                        help="Path to CBSD68 test set")
    parser.add_argument("--urban100_path", type=str, required=True,
                        help="Path to Urban100 test set")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[15, 25, 50])
    parser.add_argument("--arch", type=str, default="elir",
                        help="Model architecture name (default: elir)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build arch config (matching training config)
    arch_cfg = {
        "name": args.arch,
        "params": {
            "fm_cfg": {"k_steps": 5, "sigma_s": 0.1, "latent_shape": [16, 64, 64],
                       "seed": 2025, "dynamic_noise": True, "force_static_noise": False,
                       "detach_latent_path": False},
            "fmir_cfg": {"name": "lunet",
                         "params": {"ch_mult": [1, 2, 2, 2], "n_mid_blocks": 4,
                                    "in_channels": 16, "hid_channels": 192,
                                    "out_channels": 16, "t_emb_dim": 160,
                                    "overparametrization": False, "use_checkpoint": False,
                                    "use_attn": True, "attn_heads": 8,
                                    "use_time_dilate": False},
                         "trainable": False},
            "mmse_cfg": {"name": "rrdbnet",
                         "params": {"c_inout": 16, "c_hid": 128, "n_rrdb": 4,
                                    "overparametrization": True},
                         "trainable": False},
            "enc_cfg": {"name": "taesd", "trainable": False},
            "dec_cfg": {"name": "sft_taesd_finetuner", "path": None, "trainable": False,
                        "params": {"latent_channels": 16, "gamma_scale": 0.1, "beta_scale": 0.1}},
            "wavelet_cfg": {"enabled": True, "in_channels": 3, "base_channels": 16,
                            "trainable": False, "band_attn": True,
                            "fmir_use_wavelet_cond": True},
        },
        "path": args.ckpt,
    }

    print(f"\nLoading model from {args.ckpt} ...")
    model = get_model(arch_cfg)
    model.to(device)
    model.eval()
    print(f"Model loaded.\n")

    # --- Per-dataset, per-sigma evaluation ---
    datasets = []
    for name, path in [("CBSD68", args.cbsd68_path), ("Urban100", args.urban100_path)]:
        if not os.path.isdir(path):
            print(f"SKIP {name}: path not found ({path})")
            continue
        datasets.append((name, path))

    results = {}
    for ds_name, ds_path in datasets:
        print(f"{'='*50}")
        print(f"  Dataset: {ds_name}  ({ds_path})")
        print(f"{'='*50}")
        for sigma in args.sigmas:
            psnr, ssim = evaluate_one(model, ds_path, float(sigma), device)
            results[(ds_name, sigma)] = (psnr, ssim)
            print(f"  σ={sigma:.0f}  →  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}\n")

    # --- Final table ---
    print("\n" + "=" * 55)
    print("           DFPIR Denoising — FINAL RESULTS")
    print("=" * 55)
    print(f"{'Dataset':>12s}  {'σ':>4s}  {'PSNR':>8s}  {'SSIM':>8s}")
    print(f"{'----------':>12s}  {'----':>4s}  {'--------':>8s}  {'--------':>8s}")
    for (ds_name, sigma), (psnr, ssim) in sorted(results.items()):
        print(f"{ds_name:>12s}  {sigma:>4.0f}  {psnr:>8.2f}  {ssim:>8.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
