"""
指标计算模块。

PSNR/SSIM: 使用 DiffUIR / BasicSR 的 MATLAB 兼容实现
  - YCbCr 色彩空间的 Y 通道 (ITU-R BT.601)
  - 8-bit 量化 (×255 → round → uint8)
  - SSIM: cv2 高斯核 11×11 σ=1.5 + BORDER_REPLICATE
其余指标: pyiqa / torchmetrics
"""

import os
import shutil
import cv2
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
import pyiqa


# ============================================================================
# DiffUIR / BasicSR MATLAB 兼容 PSNR/SSIM
# ============================================================================

def _convert_input_type_range(img):
    img = img.astype(np.float32)
    if img.dtype == np.float32:
        pass
    elif img.dtype == np.uint8:
        img = img / 255.
    else:
        raise TypeError(f'Unsupported img type: {img.dtype}')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'Unsupported dst_type: {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img = img / 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """BGR → YCbCr, ITU-R BT.601, 像素级对齐 MATLAB."""
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214],
                  [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    return _convert_output_type_range(out_img, img_type)


def reorder_image(img, input_order='HWC'):
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """RGB → YCbCr Y 通道, [0,255] 范围."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _ssim_cly(img1, img2):
    """单通道 SSIM, 对齐 MATLAB ssim()."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _tensor2numpy(tensor):
    """torch [C,H,W] [0,1] → numpy uint8 HWC [0,255] 含 .round() 量化."""
    img = tensor.squeeze(0).detach().cpu().float().clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img


def calculate_psnr(tensor_pred, tensor_gt, test_y_channel=True):
    """DiffUIR 兼容 PSNR."""
    img_pred = _tensor2numpy(tensor_pred)
    img_gt = _tensor2numpy(tensor_gt)
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)

    if test_y_channel:
        img_pred = to_y_channel(img_pred)
        img_gt = to_y_channel(img_gt)

    mse = np.mean((img_pred - img_gt) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def calculate_ssim(tensor_pred, tensor_gt, test_y_channel=True):
    """DiffUIR 兼容 SSIM."""
    img_pred = _tensor2numpy(tensor_pred)
    img_gt = _tensor2numpy(tensor_gt)
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)

    if test_y_channel:
        img_pred = to_y_channel(img_pred)
        img_gt = to_y_channel(img_gt)
        return _ssim_cly(img_pred[..., 0], img_gt[..., 0])

    ssims = []
    for i in range(3):
        ssims.append(_ssim_cly(img_pred[..., i], img_gt[..., i]))
    return np.array(ssims).mean()


# ============================================================================
# MetricEval
# ============================================================================

class MetricEval(object):
    def __init__(self, metric, device, out_dir=None):
        self.metric = metric
        self.values = []
        self.count = 0
        self.image_size = 512
        self.out_dir = out_dir

        if metric == "fid":
            self.evaluater = FrechetInceptionDistance(normalize=True)
            self.evaluater.to(device)
        elif metric == "fid-g":
            self.evaluater = FrechetInceptionDistance(normalize=True, reset_real_features=False)
            load_fid_statistics(os.path.join(os.path.dirname(__file__), "datasets", "celeba_fid_stat.pt"),
                                self.evaluater, device)
            self.evaluater.to(device)
        elif metric == "fid-f":
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir, "out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
            self.evaluater = pyiqa.create_metric('fid', device=device)
        elif metric == "lpips":
            self.evaluater = pyiqa.create_metric(metric, device=device, net="vgg")
        elif metric in ("psnr", "ssim"):
            self.evaluater = None  # 使用 DiffUIR MATLAB 兼容实现
        elif metric == "save":
            self.out_dir = "out" if out_dir is None else os.path.join(self.out_dir, "out")
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            self.evaluater = pyiqa.create_metric(metric, device=device)

    def compute(self, x_hq_hat, x_hq=None):
        x_hq_hat = torch.nan_to_num(x_hq_hat.float(), nan=0.0, posinf=1.0, neginf=0.0)
        x_hq_hat = x_hq_hat.clamp(0.0, 1.0)
        if x_hq is not None:
            x_hq = torch.nan_to_num(x_hq.float(), nan=0.0, posinf=1.0, neginf=0.0)
            x_hq = x_hq.clamp(0.0, 1.0)

        bs = x_hq_hat.shape[0]

        # ---- DiffUIR MATLAB 兼容 PSNR/SSIM (逐图计算,存储单张的值) ----
        if self.metric == "psnr":
            for i in range(bs):
                val = calculate_psnr(x_hq_hat[i:i + 1], x_hq[i:i + 1], test_y_channel=True)
                self.values.append(torch.tensor([val]))
            self.count += bs
            return

        if self.metric == "ssim":
            for i in range(bs):
                val = calculate_ssim(x_hq_hat[i:i + 1], x_hq[i:i + 1], test_y_channel=True)
                self.values.append(torch.tensor([val]))
            self.count += bs
            return

        # ---- FID / 其他指标 ----
        if self.metric == "fid":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            self.evaluater.update(x_hq, real=True)
            self.count += bs
            return
        elif self.metric == "fid-g":
            self.evaluater.to(x_hq_hat.device)
            self.evaluater.update(x_hq_hat, real=False)
            self.count += bs
            return
        elif self.metric == "fid-f":
            self.image_size = x_hq.shape[-1]
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count) + ".png"))
                self.count += 1
            return
        elif self.metric in ["niqe", "clipiqa", "musiq"]:
            value = self.evaluater(x_hq_hat)
            self.count += bs
        elif self.metric == "save":
            for img in x_hq_hat:
                save_image(img, os.path.join(self.out_dir, str(self.count) + ".png"))
                self.count += 1
            return
        else:
            assert x_hq_hat.shape == x_hq.shape
            value = self.evaluater(x_hq_hat, x_hq)
            self.count += bs

        self.values.append(value.reshape(-1))

    def get_final(self):
        if self.metric in ("fid", "fid-g"):
            final_value = self.evaluater.compute()
            self.evaluater.reset()
        elif self.metric == "fid-f":
            final_value = self.evaluater(self.out_dir,
                                         dataset_name="FFHQ",
                                         dataset_res=self.image_size,
                                         dataset_split="trainval70k",
                                         verbose=False)
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)
        elif self.metric in ("psnr", "ssim"):
            final_value = torch.mean(torch.concat(self.values))
        else:
            final_value = torch.mean(torch.concat(self.values))
        self.count = 0
        self.values.clear()
        return final_value


def load_fid_statistics(stat_path, evaluator, device):
    real_features = torch.load(stat_path, map_location=device)
    evaluator.real_features_sum = real_features['real_features_sum']
    evaluator.real_features_cov_sum = real_features['real_features_cov_sum']
    evaluator.real_features_num_samples = real_features['real_features_num_samples']


def save_fid_statistics(evaluater):
    d = {"real_features_sum": evaluater.real_features_sum,
         "real_features_cov_sum": evaluater.real_features_cov_sum,
         "real_features_num_samples": evaluater.real_features_num_samples}
    torch.save(d, "a.pt")
