import torch
from torchvision.transforms import v2
from ELIR.datasets.dataset import BasicLoader, get_latent_cache_paths
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import glob



class ImagenetValDataset(Dataset):
    def __init__(self, image_folder, patch_size, use_latent_cache=False):
        super(ImagenetValDataset, self).__init__()
        self.use_latent_cache = use_latent_cache
        if self.use_latent_cache:
            self.lq_pt_paths, self.hq_pt_paths = get_latent_cache_paths(image_folder)
            self.lq_images_path = []
            self.hq_images_path = []
        else:
            self.lq_images_path = self.get_file_path(image_folder, "lq")
            self.hq_images_path = self.get_file_path(image_folder, "gt")
        self.transform_LQ, self.transform_HQ = self.preprocess(patch_size)

    def get_file_path(self, image_folder, q):
        images_path = []
        image_folder_split = os.path.join(image_folder, q)
        for file in sorted(glob.glob(os.path.join(image_folder_split,"*.png"))):
            images_path.append(file)
        return images_path

    def __len__(self):
        if self.use_latent_cache:
            return len(self.lq_pt_paths)
        return len(self.hq_images_path)

    def preprocess(self, patch_size):
        transform_LQ = v2.Compose([
            v2.Resize((patch_size, patch_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToTensor()])
        transform_HQ = v2.Compose([
            v2.ToTensor()])
        return transform_LQ, transform_HQ

    def __getitem__(self, index):
        if self.use_latent_cache:
            lq_path = self.lq_pt_paths[index]
            hq_path = self.hq_pt_paths[index]
            lq = torch.load(lq_path, weights_only=True)
            hq = torch.load(hq_path, weights_only=True)
            return lq, hq

        img_path = self.lq_images_path[index]
        img = Image.open(img_path).convert("RGB")
        img_LQ = self.transform_LQ(img)
        img_path = self.hq_images_path[index]
        img = Image.open(img_path).convert("RGB")
        img_HQ = self.transform_HQ(img)
        return img_LQ, img_HQ



class Imagenet256(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        batch_size = dataset_params.get("batch_size", 32)
        num_workers = dataset_params.get("num_workers", 4)
        patch_size = dataset_params.get("patch_size", 64)
        use_latent_cache = dataset_params.get("use_latent_cache", False)

        # Dataset
        dataset = ImagenetValDataset(path, patch_size, use_latent_cache=use_latent_cache)
        # Loader
        loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

        return loader
