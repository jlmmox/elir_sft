import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ELIR.datasets.dataset import BasicLoader, Padding2Multiple, get_latent_cache_paths


class LOLDataset(Dataset):
    def __init__(
        self,
        root,
        lq_subdir="low",
        hq_subdir="high",
        patch_size=256,
        training=True,
        extensions=("png", "jpg", "jpeg"),
        use_latent_cache=False,
    ):
        super().__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self.lq_subdir = lq_subdir
        self.hq_subdir = hq_subdir
        self.patch_size = patch_size
        self.training = training
        self.use_latent_cache = use_latent_cache
        self.extensions = tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)
        self.to_tensor = v2.ToTensor()
        self.pad = Padding2Multiple(pad_to_multiple=8)
        self.pad_eval = Padding2Multiple(pad_to_multiple=32)
        if self.use_latent_cache:
            self.lq_pt_paths, self.hq_pt_paths = get_latent_cache_paths(self.root)
            self.pairs = []
        else:
            self.pairs = self._build_pairs()

    def _build_pairs(self):
        lq_root = os.path.join(self.root, self.lq_subdir)
        hq_root = os.path.join(self.root, self.hq_subdir)
        if not os.path.isdir(lq_root) or not os.path.isdir(hq_root):
            raise FileNotFoundError(
                f"Expected lq/hq folders at '{lq_root}' and '{hq_root}'"
            )

        lq_files = []
        for ext in self.extensions:
            lq_files.extend(
                sorted(glob.glob(os.path.join(lq_root, f"**/*{ext}"), recursive=True))
            )

        pairs = []
        skipped = 0
        for lq_path in lq_files:
            fname = os.path.relpath(lq_path, lq_root)
            hq_path = os.path.join(hq_root, fname)
            if not os.path.isfile(hq_path):
                continue

            if self.training and self.patch_size is not None:
                try:
                    with Image.open(lq_path) as lq_im, Image.open(hq_path) as hq_im:
                        lq_w, lq_h = lq_im.size
                        hq_w, hq_h = hq_im.size
                    if min(lq_w, hq_w) < self.patch_size or min(lq_h, hq_h) < self.patch_size:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue

            pairs.append((lq_path, hq_path))

        if len(pairs) == 0:
            raise FileNotFoundError(
                f"No valid image pairs found under '{lq_root}' and '{hq_root}'"
            )
        if skipped > 0:
            print(
                f"[LOLDataset] Skipped {skipped} pairs where image size < patch_size ({self.patch_size})"
            )
        return pairs

    def _equalize_low_light(self, img: Image.Image) -> Image.Image:
        # Use YCrCb luminance equalization to boost low-light contrast while preserving chroma.
        rgb = np.asarray(img)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        rgb_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(rgb_eq.astype(np.uint8))

    def _joint_aug(self, lq: Image.Image, hq: Image.Image):
        # 1. Joint random crop
        if self.patch_size is not None:
            lq_w, lq_h = lq.size
            hq_w, hq_h = hq.size
            crop_w = min(lq_w, hq_w)
            crop_h = min(lq_h, hq_h)

            hcrop = np.random.randint(0, crop_h - self.patch_size + 1)
            wcrop = np.random.randint(0, crop_w - self.patch_size + 1)
            box = (wcrop, hcrop, wcrop + self.patch_size, hcrop + self.patch_size)
            lq = lq.crop(box)
            hq = hq.crop(box)

        # 2. Joint 90° rotation (0/90/180/270)
        if torch.rand(1) < 0.5:
            angle = np.random.choice([0, 90, 180, 270])
            if angle != 0:
                lq = lq.rotate(angle, expand=False)
                hq = hq.rotate(angle, expand=False)

        # 3. Joint horizontal flip
        if torch.rand(1) < 0.5:
            lq = ImageOps.mirror(lq)
            hq = ImageOps.mirror(hq)

        # 4. Joint brightness/contrast perturbation (mild, not changing LQ nature)
        if torch.rand(1) < 0.3:
            from PIL import ImageEnhance
            bf = 0.9 + torch.rand(1).item() * 0.2    # [0.9, 1.1]
            lq = ImageEnhance.Brightness(lq).enhance(bf)
            hq = ImageEnhance.Brightness(hq).enhance(bf)
        if torch.rand(1) < 0.3:
            from PIL import ImageEnhance
            cf = 0.9 + torch.rand(1).item() * 0.2    # [0.9, 1.1]
            lq = ImageEnhance.Contrast(lq).enhance(cf)
            hq = ImageEnhance.Contrast(hq).enhance(cf)

        return lq, hq

    def __len__(self):
        if self.use_latent_cache:
            return len(self.lq_pt_paths)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.use_latent_cache:
            lq_path = self.lq_pt_paths[idx]
            hq_path = self.hq_pt_paths[idx]
            lq = torch.load(lq_path, weights_only=True).clone()
            hq = torch.load(hq_path, weights_only=True).clone()
            return lq, hq

        lq_path, hq_path = self.pairs[idx]
        lq = Image.open(lq_path).convert("RGB")
        hq = Image.open(hq_path).convert("RGB")

        lq = self._equalize_low_light(lq)

        if self.training:
            lq, hq = self._joint_aug(lq, hq)
            lq = self.pad(lq)
            hq = self.pad(hq)
        else:
            # Validation/test must keep exact original resolution.
            return self.to_tensor(lq).clone(), self.to_tensor(hq).clone()

        return self.to_tensor(lq), self.to_tensor(hq)


class LOL(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        lq_subdir = dataset_params.get("lq_subdir", "low")
        hq_subdir = dataset_params.get("hq_subdir", "high")
        patch_size = dataset_params.get("patch_size", 256)
        batch_size = dataset_params.get("batch_size", 8)
        num_workers = dataset_params.get("num_workers", 4)
        training = dataset_params.get("training", True)
        extensions = dataset_params.get("extensions", ("png", "jpg", "jpeg"))
        use_latent_cache = dataset_params.get("use_latent_cache", False)

        dataset = LOLDataset(
            path,
            lq_subdir=lq_subdir,
            hq_subdir=hq_subdir,
            patch_size=patch_size,
            training=training,
            extensions=extensions,
            use_latent_cache=use_latent_cache,
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
