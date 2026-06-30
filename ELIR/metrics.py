"""
指标计算模块。

PSNR/SSIM: 严格对齐 DiffUIR 官方测试协议
  - tensor → BGR uint8 (匹配 tensor2img(rgb2bgr=True))
  - YCbCr Y 通道 (ITU-R BT.601, bgr2ycbcr)
  - 8-bit 量化 (×255 → round → uint8)
  - SSIM: 11×11 Gaussian σ=1.5 + BORDER_REPLICATE, C1/C2 基于 255
  - crop_border=0
  - 严格逐图平均（不先求 batch 均值再对 batch 均值平均）
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
# 色彩空间转换 —— 对齐 DiffUIR matlab_functions.py
# ============================================================================

def _convert_input_type_range(img):
    """将输入图像转为 np.float32，范围 [0, 1]。

    对齐 DiffUIR matlab_functions.py::_convert_input_type_range。
    注意：必须先保存原 dtype 再转换，否则 uint8 输入不会被 /255。
    """
    img_type = img.dtype            # ★ 先保存原类型，再转换
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'Unsupported img type: {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """将图像转换到目标类型和范围。

    对齐 DiffUIR matlab_functions.py::_convert_output_type_range。
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'Unsupported dst_type: {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img = img / 255.
    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """RGB → YCbCr, ITU-R BT.601, 像素级对齐 MATLAB rgb2ycbcr。

    来自 DiffUIR matlab_functions.py::rgb2ycbcr。
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0],
                  [128.553, -74.203, -93.786],
                  [24.966, 112.0, -18.214]]) + [16, 128, 128]
    return _convert_output_type_range(out_img, img_type)


def bgr2ycbcr(img, y_only=False):
    """BGR → YCbCr, ITU-R BT.601, 像素级对齐 MATLAB。

    来自 DiffUIR matlab_functions.py::bgr2ycbcr。
    注意：输入必须是 BGR 通道顺序（与 cv2.imread / DiffUIR tensor2img(rgb2bgr=True) 输出一致）。
    """
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
    """重排图像维度到 HWC。

    对齐 DiffUIR metrics/metric_util.py::reorder_image。
    """
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """BGR uint8 [0,255] HWC → YCbCr Y 通道, 返回 float64 [0,255]。

    对齐 DiffUIR metrics/metric_util.py::to_y_channel。
    注意：输入必须是 BGR 顺序 —— 与 DiffUIR tensor2img(rgb2bgr=True) 输出一致。
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return (img * 255.).astype(np.float64)


# ============================================================================
# SSIM 核心 —— 对齐 DiffUIR metrics/psnr_ssim.py::_ssim_cly
# ============================================================================

def _ssim_cly(img1, img2):
    """单通道 SSIM, 对齐 DiffUIR _ssim_cly。

    11×11 Gaussian kernel, σ=1.5, C1/C2 基于 255, BORDER_REPLICATE。
    """
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


# ============================================================================
# Tensor → numpy 转换 —— 对齐 DiffUIR tensor2img
# ============================================================================

def _tensor2numpy_single(tensor):
    """torch [C,H,W] [0,1] RGB → numpy uint8 HWC BGR [0,255]。

    严格对齐 DiffUIR tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0,1))。
    关键步骤：clamp → ×255 → round → uint8 → RGB2BGR。
    """
    img = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    # ★ 关键修复: RGB → BGR, 对齐 DiffUIR tensor2img(rgb2bgr=True)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ============================================================================
# PSNR / SSIM 单张计算
# ============================================================================

def _calc_psnr_single(tensor_pred, tensor_gt, test_y_channel=True):
    """单张图的 DiffUIR 兼容 PSNR。

    输入: torch [1,C,H,W] 或 [C,H,W], RGB, [0,1]。
    输出: float。
    """
    img_pred = _tensor2numpy_single(tensor_pred)   # BGR uint8
    img_gt   = _tensor2numpy_single(tensor_gt)     # BGR uint8
    img_pred = img_pred.astype(np.float64)
    img_gt   = img_gt.astype(np.float64)

    if test_y_channel:
        img_pred = to_y_channel(img_pred)   # BGR → Y, float64 [0,255]
        img_gt   = to_y_channel(img_gt)

    mse = np.mean((img_pred - img_gt) ** 2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img_pred.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


def _calc_ssim_single(tensor_pred, tensor_gt, test_y_channel=True):
    """单张图的 DiffUIR 兼容 SSIM。

    输入: torch [1,C,H,W] 或 [C,H,W], RGB, [0,1]。
    输出: float。
    """
    img_pred = _tensor2numpy_single(tensor_pred)   # BGR uint8
    img_gt   = _tensor2numpy_single(tensor_gt)     # BGR uint8
    img_pred = img_pred.astype(np.float64)
    img_gt   = img_gt.astype(np.float64)

    if test_y_channel:
        img_pred = to_y_channel(img_pred)
        img_gt   = to_y_channel(img_gt)
        return float(_ssim_cly(img_pred[..., 0], img_gt[..., 0]))

    # 三通道分别计算后取均值
    ssims = []
    for i in range(3):
        ssims.append(_ssim_cly(img_pred[..., i], img_gt[..., i]))
    return float(np.array(ssims).mean())


# ============================================================================
# PSNR / SSIM 批量计算 —— 返回逐图值列表, 保证严格逐图平均
# ============================================================================

def calculate_psnr(tensor_pred, tensor_gt, test_y_channel=True):
    """DiffUIR 兼容 PSNR。

    支持 [C,H,W] 单张或 [B,C,H,W] 批量。
    返回:
        - [C,H,W] 单张, 或 [1,C,H,W] → float
        - [B,C,H,W] 批量 (B>1) → list[float]: 每张图的 PSNR 值。
      批量时调用方自行对所有值取平均，确保最后一个不足 batch 不会权重错误。
    """
    if tensor_pred.dim() == 3:
        return _calc_psnr_single(tensor_pred, tensor_gt, test_y_channel=test_y_channel)

    bs = tensor_pred.shape[0]
    if bs == 1:
        return _calc_psnr_single(tensor_pred, tensor_gt, test_y_channel=test_y_channel)

    vals = []
    for i in range(bs):
        vals.append(_calc_psnr_single(
            tensor_pred[i:i + 1], tensor_gt[i:i + 1], test_y_channel=test_y_channel))
    return vals


def calculate_ssim(tensor_pred, tensor_gt, test_y_channel=True):
    """DiffUIR 兼容 SSIM。

    支持 [C,H,W] 单张或 [B,C,H,W] 批量。
    返回:
        - [C,H,W] 单张, 或 [1,C,H,W] → float
        - [B,C,H,W] 批量 (B>1) → list[float]: 每张图的 SSIM 值。
      批量时调用方自行对所有值取平均，确保最后一个不足 batch 不会权重错误。
    """
    if tensor_pred.dim() == 3:
        return _calc_ssim_single(tensor_pred, tensor_gt, test_y_channel=test_y_channel)

    bs = tensor_pred.shape[0]
    if bs == 1:
        return _calc_ssim_single(tensor_pred, tensor_gt, test_y_channel=test_y_channel)

    vals = []
    for i in range(bs):
        vals.append(_calc_ssim_single(
            tensor_pred[i:i + 1], tensor_gt[i:i + 1], test_y_channel=test_y_channel))
    return vals


# ============================================================================
# SOTS 去雾专用评测 —— 对齐 DiffUIR eval/SOTS.m
# ============================================================================

def evaluate_sots_dehaze(pred_dir, gt_dir, pred_ext='.png', gt_ext='.png'):
    """按 DiffUIR eval/SOTS.m 协议评测 SOTS 去雾结果。

    协议:
    1. 遍历预测图像，根据文件名提取 GT 名称（下划线分割取第一部分）；
    2. GT resize 到预测尺寸（bicubic, 对齐 MATLAB imresize）；
    3. 转 YCbCr Y 通道（BGR → bgr2ycbcr）；
    4. 逐图计算 PSNR/SSIM 并平均。

    文件命名规则（与 SOTS.m 一致）:
        预测:  {gt_basename}_{something}.png   (例如 "1400_0.8.png")
        GT:    {gt_basename}.png               (例如 "1400.png")

    Args:
        pred_dir: 预测图像目录。
        gt_dir:   GT 图像目录。
        pred_ext: 预测图像扩展名（默认 '.png'）。
        gt_ext:   GT 图像扩展名（默认 '.png'）。

    Returns:
        dict: {'psnr': float, 'ssim': float, 'count': int,
               'psnr_per_image': list[float], 'ssim_per_image': list[float]}
    """
    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if f.lower().endswith(pred_ext.lower())
    ])

    if not pred_files:
        raise ValueError(f'No prediction files with extension "{pred_ext}" found in {pred_dir}')

    psnr_vals = []
    ssim_vals = []

    for pred_name in pred_files:
        # 提取 GT 名称: 按 '_' 分割取第一部分
        base = os.path.splitext(pred_name)[0]
        gt_base = base.split('_')[0]
        gt_name = gt_base + gt_ext

        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        # 如果默认扩展名找不到，尝试常见图像扩展名
        if not os.path.exists(gt_path):
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                alt_path = os.path.join(gt_dir, gt_base + ext)
                if os.path.exists(alt_path):
                    gt_path = alt_path
                    found = True
                    break
            if not found:
                print(f'Warning: GT not found for {pred_name} (tried {gt_name}), skipping')
                continue

        pred_img = cv2.imread(pred_path, cv2.IMREAD_COLOR)   # BGR
        gt_img   = cv2.imread(gt_path,   cv2.IMREAD_COLOR)   # BGR

        if pred_img is None:
            print(f'Warning: Cannot read {pred_path}, skipping')
            continue
        if gt_img is None:
            print(f'Warning: Cannot read {gt_path}, skipping')
            continue

        # GT resize 到预测尺寸 (bicubic, 对齐 MATLAB imresize)
        h, w = pred_img.shape[:2]
        if gt_img.shape[:2] != (h, w):
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # 转 YCbCr Y 通道 (BGR 输入给 bgr2ycbcr, 对齐 DiffUIR)
        pred_y = to_y_channel(pred_img)[..., 0]   # float64 [0,255], 2D
        gt_y   = to_y_channel(gt_img)[..., 0]

        # PSNR
        mse = np.mean((pred_y - gt_y) ** 2)
        if mse == 0:
            psnr_vals.append(float('inf'))
        else:
            psnr_vals.append(20. * np.log10(255. / np.sqrt(mse)))

        # SSIM
        ssim_vals.append(float(_ssim_cly(pred_y, gt_y)))

    n = len(psnr_vals)
    if n == 0:
        raise ValueError('No valid image pairs evaluated')

    return {
        'psnr': float(np.mean(psnr_vals)),
        'ssim': float(np.mean(ssim_vals)),
        'count': n,
        'psnr_per_image': psnr_vals,
        'ssim_per_image': ssim_vals,
    }


# ============================================================================
# MetricEval
# ============================================================================

class MetricEval(object):
    def __init__(self, metric, device, out_dir=None, test_y_channel=True):
        self.metric = metric
        self.values = []          # 存储逐图指标值（psnr/ssim 为 per-image tensors）
        self.count = 0
        self.image_size = 512
        self.out_dir = out_dir
        self.test_y_channel = test_y_channel  # True=Y通道, False=RGB三通道

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

        # ---- DiffUIR MATLAB 兼容 PSNR/SSIM (逐图存储) ----
        if self.metric == "psnr":
            vals = calculate_psnr(x_hq_hat, x_hq, test_y_channel=self.test_y_channel)
            if isinstance(vals, list):
                self.values.extend([torch.tensor([v]) for v in vals])
                self.count += len(vals)
            else:
                self.values.append(torch.tensor([vals]))
                self.count += 1
            return

        if self.metric == "ssim":
            vals = calculate_ssim(x_hq_hat, x_hq, test_y_channel=self.test_y_channel)
            if isinstance(vals, list):
                self.values.extend([torch.tensor([v]) for v in vals])
                self.count += len(vals)
            else:
                self.values.append(torch.tensor([vals]))
                self.count += 1
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
            # ★ 关键修复: 逐图取平均, 而非先求 batch 均值再对 batch 均值平均
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
