from abc import abstractmethod
import glob
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image, ImageOps
import numpy as np




def get_loader(ds_params):
    ds_name = ds_params.get("name")
    if ds_name == "CelebA":
        from ELIR.datasets.celeba import CelebA
        dl = CelebA().create_loaders(ds_params)
    elif ds_name == "FFHQ":
        from ELIR.datasets.ffhq import FFHQ
        dl = FFHQ().create_loaders(ds_params)
    elif ds_name == "CelebAdult":
        from ELIR.datasets.celebadult import CelebAdult
        dl = CelebAdult().create_loaders(ds_params)
    elif ds_name == "WebPhoto":
        from ELIR.datasets.webphoto import WebPhoto
        dl = WebPhoto().create_loaders(ds_params)
    elif ds_name == "LFW":
        from ELIR.datasets.lfw import LFW
        dl = LFW().create_loaders(ds_params)
    elif ds_name == "Imagenet":
        from ELIR.datasets.imagenet import Imagenet
        dl = Imagenet().create_loaders(ds_params)
    elif ds_name == "Imagenet256":
        from ELIR.datasets.imagenet256 import Imagenet256
        dl = Imagenet256().create_loaders(ds_params)
    elif ds_name == "RealSet80":
        from ELIR.datasets.realset80 import RealSet80
        dl = RealSet80().create_loaders(ds_params)
    elif ds_name == "Paired":
        dl = Paired().create_loaders(ds_params)
    elif ds_name == "RESIDE":
        from ELIR.datasets.reside import RESIDE
        dl = RESIDE().create_loaders(ds_params)
    elif ds_name == "LOL":
        from ELIR.datasets.lol import LOL
        dl = LOL().create_loaders(ds_params)
    else:
        raise Exception("Dataset is unknown!")

    print("{} dataset was loaded! Number of samples: {}".format(ds_name, len(dl.dataset)))
    return dl


def get_latent_cache_paths(root):
    root = os.path.abspath(os.path.expanduser(root))
    lq_root = os.path.join(root, "latents_lq")
    hq_root = os.path.join(root, "latents_hq")
    if not os.path.isdir(lq_root) or not os.path.isdir(hq_root):
        raise FileNotFoundError(f"Expected latent cache folders at {lq_root} and {hq_root}")

    lq_pt_paths = sorted(glob.glob(os.path.join(lq_root, "**", "*.pt"), recursive=True))
    hq_pt_paths = sorted(glob.glob(os.path.join(hq_root, "**", "*.pt"), recursive=True))

    if len(lq_pt_paths) == 0 or len(hq_pt_paths) == 0:
        raise FileNotFoundError(f"No .pt latent files found under {lq_root} and {hq_root}")
    if len(lq_pt_paths) != len(hq_pt_paths):
        raise ValueError(
            f"Latent cache size mismatch: {len(lq_pt_paths)} lq files vs {len(hq_pt_paths)} hq files"
        )

    return lq_pt_paths, hq_pt_paths

class BasicLoader(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_loaders(self, dataset_params):
        raise NotImplemented(f'{self.__class__.__name__} have to be implemented!')

    def get_labels(self, dataset):
        labels = []
        for _,y in dataset:
            labels.append(y)
        return labels

    def get_mean_std(self, dataset):
        sum1, sum2 = 0, 0
        for img,_ in dataset:
            sum1 += torch.mean(img, dim=(0,2,3))
            sum2 += torch.mean(img ** 2, dim=[0,2,3])
        count = len(dataset)
        mean = sum1 / count
        var = (sum2 / count) - (mean ** 2)
        std = torch.sqrt(var)
        return mean, std

def aug(img: Image, patch_size, crop=True) -> Image:
    # Random Crop patch size
    if crop:
        W, H = img.size
        hcrop, wcrop = np.random.randint(max(1, H - patch_size)), np.random.randint(max(1, W - patch_size))
        img = img.crop((wcrop, hcrop, min(W, wcrop + patch_size), min(H, hcrop + patch_size)))
    # Random flip
    if torch.rand(1) < 0.5:
        img = ImageOps.mirror(img)
    return img

class Padding2Multiple(object):
    def __init__(self, pad_to_multiple=8):
        self.pad_to_multiple = pad_to_multiple

    def get_pad(self, x):
        return (x // self.pad_to_multiple + int(x % self.pad_to_multiple != 0)) * self.pad_to_multiple - x

    def __call__(self, img):
        W, H = img.size
        if W % self.pad_to_multiple == 0 and H % self.pad_to_multiple == 0:
            return img
        pad_h, pad_w = self.get_pad(H), self.get_pad(W)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class ResizeLongEdge(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if h == w:
            new_h, new_w = self.size, self.size
            pad_h, pad_w =0, 0
        elif h < w:
            new_h, new_w = int(h * (self.size / w)), self.size
            pad_h, pad_w = self.size - new_h, 0
        else:
            new_h, new_w = self.size, int(w * (self.size / h))
            pad_h, pad_w = 0, self.size - new_w

        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class Padding2Size(object):
    def __init__(self, H_target, W_target):
        self.H_target = H_target
        self.W_target = W_target

    def __call__(self, img):
        W, H = img.size
        pad_h, pad_w = max(0,self.H_target-H), max(0,self.W_target-W)
        img_mp = np.array(img)
        img_mp = np.pad(img_mp, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        return Image.fromarray(img_mp.astype(np.uint8))


class AddGaussianNoise(object):
    def __init__(self, std_high=1, std_low=0):
        self.std_high = std_high
        self.std_low = std_low

    def __call__(self, img):
        std = np.random.uniform(low=self.std_low, high=self.std_high)
        img = np.asarray(img)
        img = img + np.random.randn(*img.shape) * std
        img = img.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class MaskInpaint(object):
    def __init__(self, prob=0.9):
        self.prob = prob

    def __call__(self, img):
        x = np.asarray(img)
        total = x.shape[0] * x.shape[1]
        mask_vec = np.ones([1, x.shape[0] * x.shape[1]])
        samples = np.random.choice(x.shape[0] * x.shape[1], int(total * self.prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.reshape((x.shape[0], x.shape[1],1))
        mask_b = np.repeat(mask_b, 3, axis=2)
        mask = np.ones_like(mask_b)
        mask[:, ...] = mask_b
        y = x * mask
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class AddColorization(object):
    def __init__(self, std_high=1, std_low=0):
        self.std_high = std_high
        self.std_low = std_low

    def __call__(self, img):
        x = np.asarray(img)
        # RGB to gray
        y = np.mean(x, axis=-1, keepdims=True)
        # Add noise
        std = np.random.uniform(low=self.std_low, high=self.std_high)
        y = y + np.random.randn(*y.shape) * std
        # Back to 3 dimesions
        y = np.repeat(y, 3, axis=-1)
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)

class MaskSqaure(object):
    def __init__(self, h=0.25, w=0.25):
        self.h = h
        self.w = w

    def __call__(self, img):
        x = np.asarray(img)
        y_max, x_max = x.shape[0], x.shape[1]
        hp, wp = int(x.shape[0] * self.h), int(x.shape[1] * self.w)
        ycp = np.random.randint(low=0, high=y_max-hp, size=(1,))
        xcp = np.random.randint(low=0, high=x_max-wp, size=(1,))
        y_start, y_stop = int(ycp), int(ycp+hp)
        x_start, x_stop = int(xcp), int(xcp+wp)
        mask = np.ones_like(x)
        mask[y_start:y_stop:,x_start:x_stop,:] = 0
        y = x*mask
        img = y.round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)


class PairedDataset(Dataset):
    def __init__(
        self,
        root,
        lq_subdir="lq",
        hq_subdir="hq",
        patch_size=None,
        training=True,
        pad_to_multiple=8,
        extensions=("png", "jpg", "jpeg"),
        use_latent_cache=False,
    ):
        super().__init__()
        self.roots = self._normalize_roots(root)
        self.lq_subdir = lq_subdir
        self.hq_subdir = hq_subdir
        self.patch_size = patch_size
        self.training = training
        self.use_latent_cache = use_latent_cache
        self.extensions = tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)
        self.to_tensor = v2.ToTensor()
        self.pad = Padding2Multiple(pad_to_multiple=pad_to_multiple)
        if self.use_latent_cache:
            self.lq_pt_paths, self.hq_pt_paths = get_latent_cache_paths(self.roots[0])
            self.pairs = []
        else:
            self.pairs = self._build_pairs()

    def _normalize_roots(self, root):
        if isinstance(root, (list, tuple)):
            roots = [os.path.abspath(os.path.expanduser(x)) for x in root]
        else:
            roots = [os.path.abspath(os.path.expanduser(root))]
        return roots

    def _build_pairs(self):
        pairs = []
        for root in self.roots:
            lq_root = os.path.join(root, self.lq_subdir)
            hq_root = os.path.join(root, self.hq_subdir)
            if not os.path.isdir(lq_root) or not os.path.isdir(hq_root):
                raise FileNotFoundError(f"Expected lq/hq folders at {lq_root} and {hq_root}")

            lq_files = []
            for ext in self.extensions:
                lq_files.extend(sorted(glob.glob(os.path.join(lq_root, f"**/*{ext}"), recursive=True)))

            for lq_path in lq_files:
                fname = os.path.relpath(lq_path, lq_root)
                hq_path = os.path.join(hq_root, fname)
                if not os.path.isfile(hq_path):
                    raise FileNotFoundError(f"Missing HQ pair for {lq_path}")
                pairs.append((lq_path, hq_path))

        if len(pairs) == 0:
            raise FileNotFoundError(f"No image pairs found under {self.roots}")
        return pairs

    def _resize_if_small(self, lq: Image.Image, hq: Image.Image):
        if self.patch_size is None:
            return lq, hq

        lq_w, lq_h = lq.size
        hq_w, hq_h = hq.size
        min_w = min(lq_w, hq_w)
        min_h = min(lq_h, hq_h)

        if min_w >= self.patch_size and min_h >= self.patch_size:
            return lq, hq

        scale = max(self.patch_size / min_w, self.patch_size / min_h)

        def _resize_keep_ratio(img):
            w, h = img.size
            new_w = max(self.patch_size, int(round(w * scale)))
            new_h = max(self.patch_size, int(round(h * scale)))
            return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        return _resize_keep_ratio(lq), _resize_keep_ratio(hq)

    def _joint_aug(self, lq: Image.Image, hq: Image.Image):
        if self.patch_size is not None:
            lq, hq = self._resize_if_small(lq, hq)

            lq_w, lq_h = lq.size
            hq_w, hq_h = hq.size
            crop_w = min(lq_w, hq_w)
            crop_h = min(lq_h, hq_h)

            hcrop = np.random.randint(0, crop_h - self.patch_size + 1)
            wcrop = np.random.randint(0, crop_w - self.patch_size + 1)
            box = (wcrop, hcrop, wcrop + self.patch_size, hcrop + self.patch_size)
            lq = lq.crop(box)
            hq = hq.crop(box)

        if torch.rand(1) < 0.5:
            lq = ImageOps.mirror(lq)
            hq = ImageOps.mirror(hq)
        return lq, hq

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
            lq, hq = self._joint_aug(lq, hq)
            lq = self.pad(lq)
            hq = self.pad(hq)
        else:
            # Validation/test must keep exact original resolution.
            return self.to_tensor(lq), self.to_tensor(hq)

        return self.to_tensor(lq), self.to_tensor(hq)


class Paired(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        lq_subdir = dataset_params.get("lq_subdir", "lq")
        hq_subdir = dataset_params.get("hq_subdir", "hq")
        extensions = dataset_params.get("extensions", ("png", "jpg", "jpeg"))
        patch_size = dataset_params.get("patch_size", None)
        training = dataset_params.get("training", True)
        pad_to_multiple = dataset_params.get("pad_to_multiple", 8)
        batch_size = dataset_params.get("batch_size", 8)
        num_workers = dataset_params.get("num_workers", 4)
        use_latent_cache = dataset_params.get("use_latent_cache", False)

        dataset = PairedDataset(path, lq_subdir=lq_subdir, hq_subdir=hq_subdir,
                                patch_size=patch_size, training=training,
                                pad_to_multiple=pad_to_multiple, extensions=extensions,
                                use_latent_cache=use_latent_cache)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=training,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=training,
                            persistent_workers=num_workers > 0)

        return loader



