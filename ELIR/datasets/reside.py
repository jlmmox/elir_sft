import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, ColorJitter
import torchvision.transforms.functional as TF

from ELIR.datasets.dataset import BasicLoader, get_latent_cache_paths


class RESIDEDataset(Dataset):
    """配对数据集，适用于 RESIDE（OTS / SOTS）去雾任务。

    LQ（含雾图）文件名格式：``<id>_<…>.png``，例如 ``0001_1_0.8.png``。
    HQ（清晰图）文件名格式：``<id>.png``，     例如 ``0001.png``。

    匹配逻辑：对每张 LQ 图，以下划线 ``_`` 分割文件名并取第一段作为前缀，
    再在 HQ 目录中查找同名（不含后缀的 <id>）图片。
    """

    def __init__(
        self,
        root,
        lq_subdir="haze",
        hq_subdir="clear",
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
        if self.use_latent_cache:
            self.lq_pt_paths, self.hq_pt_paths = get_latent_cache_paths(self.root)
            self.pairs = []
        else:
            self.pairs = self._build_pairs()

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------

    def _build_pairs(self):
        lq_root = os.path.join(self.root, self.lq_subdir)
        hq_root = os.path.join(self.root, self.hq_subdir)
        if not os.path.isdir(lq_root) or not os.path.isdir(hq_root):
            raise FileNotFoundError(
                f"Expected lq/hq folders at '{lq_root}' and '{hq_root}'"
            )

        # Collect all LQ files (supports nested subdirectories)
        lq_files = []
        for ext in self.extensions:
            lq_files.extend(
                sorted(glob.glob(os.path.join(lq_root, f"**/*{ext}"), recursive=True))
            )

        pairs = []
        skipped = 0
        for lq_path in lq_files:
            stem = os.path.splitext(os.path.basename(lq_path))[0]
            # e.g. "0001_1_0.8"  ->  prefix "0001"
            prefix = stem.split("_")[0]

            hq_path = None
            for ext in self.extensions:
                candidate = os.path.join(hq_root, f"{prefix}{ext}")
                if os.path.isfile(candidate):
                    hq_path = candidate
                    break

            if hq_path is None:
                # HQ 不存在时跳过（兼容只有 LQ 的测试集分割）
                continue

            # 过滤掉尺寸小于 patch_size 的图像对，避免训练时上采样引入伪影
            if self.patch_size is not None and self.training:
                try:
                    with Image.open(lq_path) as im:
                        W, H = im.size
                    if W < self.patch_size or H < self.patch_size:
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
            print(f"[RESIDEDataset] Skipped {skipped} pairs where image size < patch_size ({self.patch_size})")
        return pairs

    # ------------------------------------------------------------------
    # Augmentation (joint, so LQ and HQ receive identical transforms)
    # ------------------------------------------------------------------

    def _joint_aug(self, lq: Image.Image, hq: Image.Image):
        """对 LQ 和 HQ 施加完全一致的随机裁剪、水平翻转与色彩抖动。
        进入此函数的图像保证 W >= patch_size 且 H >= patch_size（由 _build_pairs 过滤保证）。
        
        色彩增强：使用相同的随机参数同时对 LQ (含雾) 和 HQ (清晰) 施加色彩抖动，
        确保配对一致性，防止模型在训练集因不对称色彩抖动而过拟合。
        """
        W, H = lq.size
        hcrop = np.random.randint(0, H - self.patch_size + 1)
        wcrop = np.random.randint(0, W - self.patch_size + 1)
        box = (wcrop, hcrop, wcrop + self.patch_size, hcrop + self.patch_size)
        lq = lq.crop(box)
        hq = hq.crop(box)
        
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            lq = ImageOps.mirror(lq)
            hq = ImageOps.mirror(hq)
        
        # 配对一致的色彩抖动：生成一组随机参数，再同时作用于 LQ 和 HQ
        if torch.rand(1) < 0.5:
            # 手动生成随机色彩增强参数，确保 LQ 和 HQ 使用完全相同的参数
            brightness_factor = np.random.uniform(0.95, 1.05)   # 亮度：95% ~ 105%
            contrast_factor = np.random.uniform(0.95, 1.05)     # 对比度：95% ~ 105%
            saturation_factor = np.random.uniform(0.95, 1.05)   # 饱和度：95% ~ 105%
            hue_factor = np.random.uniform(-0.02, 0.02)         # 色相：-0.02 ~ 0.02
            
            # 使用相同参数分别作用于 LQ 和 HQ，确保配对一致性
            lq = TF.adjust_brightness(lq, brightness_factor)
            hq = TF.adjust_brightness(hq, brightness_factor)
            
            lq = TF.adjust_contrast(lq, contrast_factor)
            hq = TF.adjust_contrast(hq, contrast_factor)
            
            lq = TF.adjust_saturation(lq, saturation_factor)
            hq = TF.adjust_saturation(hq, saturation_factor)
            
            lq = TF.adjust_hue(lq, hue_factor)
            hq = TF.adjust_hue(hq, hue_factor)
        
        return lq, hq

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        if self.use_latent_cache:
            return len(self.lq_pt_paths)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.use_latent_cache:
            lq_path = self.lq_pt_paths[idx]
            hq_path = self.hq_pt_paths[idx]
            lq = torch.load(lq_path, weights_only=True)
            hq = torch.load(hq_path, weights_only=True)
            return lq, hq

        lq_path, hq_path = self.pairs[idx]
        lq = Image.open(lq_path).convert("RGB")
        hq = Image.open(hq_path).convert("RGB")

        if self.training:
            # 随机裁剪后输出恰好为 patch_size × patch_size，无需额外填充
            lq, hq = self._joint_aug(lq, hq)
        else:
            # 验证/测试阶段严格保持原始分辨率，不做 resize/pad。
            return self.to_tensor(lq), self.to_tensor(hq)

        return self.to_tensor(lq), self.to_tensor(hq)


class RESIDE(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        lq_subdir = dataset_params.get("lq_subdir", "haze")
        hq_subdir = dataset_params.get("hq_subdir", "clear")
        patch_size = dataset_params.get("patch_size", 256)
        batch_size = dataset_params.get("batch_size", 8)
        num_workers = dataset_params.get("num_workers", 4)
        training = dataset_params.get("training", True)
        extensions = dataset_params.get("extensions", ("png", "jpg", "jpeg"))
        use_latent_cache = dataset_params.get("use_latent_cache", False)

        dataset = RESIDEDataset(
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
