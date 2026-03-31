import glob
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from ELIR.datasets.dataset import BasicLoader, get_latent_cache_paths
import os
from PIL import Image



class LFWDataset(Dataset):
    def __init__(self, image_folder, use_latent_cache=False):
        super(LFWDataset, self).__init__()
        self.use_latent_cache = use_latent_cache
        if self.use_latent_cache:
            self.lq_pt_paths, self.hq_pt_paths = get_latent_cache_paths(image_folder)
            self.lq_images_path = []
        else:
            self.lq_images_path = self.get_file_path(image_folder)
        self.transform_LQ = self.preprocess()

    def preprocess(self):
        transform_LQ = v2.Compose([
            v2.ToTensor()])
        return transform_LQ

    def get_file_path(self, image_folder):
        images_path = []
        for file in sorted(glob.glob(os.path.join(image_folder,"*.png"))):
            images_path.append(os.path.join(image_folder, file))
        return images_path

    def __len__(self):
        if self.use_latent_cache:
            return len(self.lq_pt_paths)
        return len(self.lq_images_path)

    def __getitem__(self, index):
        if self.use_latent_cache:
            lq_path = self.lq_pt_paths[index]
            hq_path = self.hq_pt_paths[index]
            lq = torch.load(lq_path, weights_only=True)
            hq = torch.load(hq_path, weights_only=True)
            return lq, hq

        img_LQ_path = self.lq_images_path[index]
        img_LQ = Image.open(img_LQ_path).convert("RGB")
        lq = self.transform_LQ(img_LQ)
        return lq, lq


class LFW(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        use_latent_cache = dataset_params.get("use_latent_cache", False)

        # Datasets
        dataset_root = path if use_latent_cache else os.path.join(path, "test")
        dataset = LFWDataset(dataset_root, use_latent_cache=use_latent_cache)

        # Loaders
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)

        return loader
