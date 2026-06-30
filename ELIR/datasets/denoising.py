"""
Denoising dataset: BSD400 + WED merged clean image pool with online Gaussian noise.

DFPIR-style training setup:
  - Merges BSD400 and WED clean images into one pool
  - Each __getitem__ randomly samples from the pool
  - Online Gaussian noise: sigma ∈ {15, 25, 50}
  - Returns (noisy, clean) as (LQ, HQ) pair

Usage in YAML:
  dataset_cfg:
    train_dataset:
      name: Denoising
      paths:                               # list of clean-image root dirs
        - /path/to/BSD400
        - /path/to/WED
      patch_size: 256
      batch_size: 8
      num_workers: 8
      training: true
      sigma_choices: [15, 25, 50]          # optional, default [15, 25, 50]
"""

import glob
import os
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ELIR.datasets.dataset import BasicLoader, Padding2Multiple


class DenoisingDataset(Dataset):
    """Clean-image pool with online Gaussian noise synthesis.

    Scans all images under `paths` (list of root dirs), merges them into a
    single pool.  Each __getitem__ samples one image uniformly at random,
    applies data augmentation, and synthesises Gaussian noise on the fly.
    """

    def __init__(
        self,
        paths,                    # single str or list of str
        patch_size=256,
        training=True,
        sigma_choices=(15, 25, 50),
        extensions=("png", "jpg", "jpeg"),
    ):
        super().__init__()
        if isinstance(paths, str):
            paths = [paths]
        self.paths = [os.path.abspath(os.path.expanduser(p)) for p in paths]
        self.patch_size = patch_size
        self.training = training
        self.sigma_choices = list(sigma_choices)
        self.extensions = tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)

        self.to_tensor = v2.ToTensor()
        self.pad = Padding2Multiple(pad_to_multiple=8)

        # Collect all clean images from all roots
        self.clean_paths = []
        for p in self.paths:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Denoising dataset path not found: {p}")
            for ext in self.extensions:
                self.clean_paths.extend(
                    sorted(glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True))
                )
        if len(self.clean_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.paths}")

        # Filter out images too small for patch cropping
        if training and patch_size is not None:
            keep = []
            skipped = 0
            for pth in self.clean_paths:
                try:
                    with Image.open(pth) as im:
                        w, h = im.size
                    if w >= patch_size and h >= patch_size:
                        keep.append(pth)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
            if skipped:
                print(f"[Denoising] Skipped {skipped} images smaller than {patch_size}x{patch_size}")
            self.clean_paths = keep

        print(f"[Denoising] Clean pool: {len(self.clean_paths)} images from {self.paths}")

    def __len__(self):
        # DFPIR-style: each "epoch" iterates ~N items with replacement
        return max(len(self.clean_paths), 4000)

    def _augment(self, img: Image.Image) -> Image.Image:
        """Random horizontal flip and 90° rotations."""
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = img.rotate(angle, expand=False)
        return img

    def _crop_patch(self, img: Image.Image) -> Image.Image:
        """Random crop to patch_size × patch_size."""
        w, h = img.size
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        return img.crop((left, top, left + self.patch_size, top + self.patch_size))

    def __getitem__(self, idx):
        # 1. Randomly sample a clean image from the pool (with replacement)
        path = random.choice(self.clean_paths)
        clean = Image.open(path).convert("RGB")

        if self.training:
            # 2a. Augmentation + crop + pad (training mode)
            clean = self._augment(clean)
            clean = self._crop_patch(clean)
            clean = self.pad(clean)
            clean_t = self.to_tensor(clean).clone()
            sigma = float(random.choice(self.sigma_choices))
        else:
            # 2b. Evaluation: keep original resolution, no aug/pad (same as LOL)
            clean_t = self.to_tensor(clean).clone()
            sigma = float(self.sigma_choices[0])  # single sigma for eval

        # 3. Online Gaussian noise
        noise = torch.randn_like(clean_t) * (sigma / 255.0)
        noisy_t = (clean_t + noise).clamp(0.0, 1.0)

        # Return: (LQ=noisy, HQ=clean)
        return noisy_t, clean_t


class Denoising(BasicLoader):
    """Loader wrapper compatible with get_loader()."""

    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        paths = dataset_params.get("paths")
        if paths is None:
            # Backward compat: allow single path
            paths = dataset_params.get("path")
        patch_size = dataset_params.get("patch_size", 256)
        batch_size = dataset_params.get("batch_size", 8)
        num_workers = dataset_params.get("num_workers", 4)
        training = dataset_params.get("training", True)
        sigma_choices = dataset_params.get("sigma_choices", [15, 25, 50])
        extensions = dataset_params.get("extensions", ("png", "jpg", "jpeg"))

        dataset = DenoisingDataset(
            paths=paths,
            patch_size=patch_size,
            training=training,
            sigma_choices=sigma_choices,
            extensions=extensions,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=training,
            persistent_workers=num_workers > 0,
        )
        return loader
