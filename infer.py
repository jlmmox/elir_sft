from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
import torch.nn.functional as F
from torchvision.io import read_image
import os
from PIL import Image
import numpy as np
from ELIR.utils import ImageSpliterTh
from ELIR.models.load_model import get_model
from tqdm import tqdm
import math
from utils import get_device
import glob
import warnings
warnings.filterwarnings("ignore")
device = get_device()

IMAGE_EXTENSION = ('jpg','png','jpeg')


def to_tensor(img_tensor):
    img_tensor = img_tensor[:3,...]
    img_tensor = img_tensor.unsqueeze(0) / 255.0
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    img_tensor = img_tensor.to(device)
    return img_tensor


def pad_to_multiple(x, m=32):
    """将 BCHW 张量的 H、W 用 reflect 模式填充到 m 的倍数。"""
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


def tensor_to_pil(t):
    """将 BCHW float[0,1] 张量（batch=1）转换为 8-bit RGB PIL Image。"""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0)
    arr = arr.byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')

def run_infer(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Create models
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)


    # ----------------------------
    # Infer all images in folder
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    in_image_folder = eval_cfg.get("in_folder",[])
    img_size = eval_cfg.get("image_size",512)
    chop = eval_cfg.get("chop",None)
    out_image_folder = eval_cfg.get("out_folder","out")
    os.makedirs(out_image_folder, exist_ok=True) # run folder
    if chop:
        sf = chop.get("sf", 4)
        upscale = chop.get("upscale", 4)
        chop_size = chop.get("chop_size", 256)
        chop_stride = chop.get("chop_stride", 224)

    for img_path in tqdm(sorted(glob.glob(os.path.join(in_image_folder,"*.*")))):
        if not img_path.endswith(IMAGE_EXTENSION):
            continue
        x = to_tensor(read_image(img_path))
        # 记录原始图像尺寸，用于推理后裁剪黑边
        orig_H, orig_W = x.shape[2], x.shape[3]
        if chop:
            patch_spliter = ImageSpliterTh(x, pch_size=chop_size, stride=chop_stride, sf=sf, extra_bs=1)
            for patch, index_infos in patch_spliter:
                patch_h, patch_w = patch.shape[2:]
                flag_pad = False
                if not (patch_h % 64 == 0 and patch_w % 64 == 0):
                    flag_pad = True
                    pad_h = (math.ceil(patch_h / 64)) * 64 - patch_h
                    pad_w = (math.ceil(patch_w / 64)) * 64 - patch_w
                    patch = F.pad(patch, pad=(0, pad_w, 0, pad_h), mode='reflect')
                pad_patch_h, pad_patch_w = patch.shape[2:]
                patch = F.interpolate(patch, size=(upscale*pad_patch_h, upscale*pad_patch_w), mode='bicubic')
                im_sr_pch = model.inference(patch)
                if flag_pad:
                    im_sr_pch = im_sr_pch[:, :, :patch_h * sf, :patch_w * sf]
                patch_spliter.update(im_sr_pch, index_infos)
            out_img = patch_spliter.gather()
        elif img_size is not None:
            # 固定尺寸模式（兼容原有行为）
            x = F.interpolate(x, size=(img_size, img_size), mode='bicubic')
            out_img = model.inference(x)
        else:
            # 原始分辨率模式（去雾等任务）：
            # 填充到 32 的倍数 → 推理 → 裁剪回原始尺寸，去除黑边
            x_pad = pad_to_multiple(x, m=32)
            out_pad = model.inference(x_pad)
            out_img = out_pad[:, :, :orig_H, :orig_W]

        # 保留完整原始文件名（如 0001_1_0.8.png），以 .png 格式保存
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_image_folder, stem + '.png')
        # 转换为 8-bit PIL Image 再保存，避免浮点精度导致的色彩失真
        tensor_to_pil(out_img).save(out_path)


    print("Done! images are at {}".format(out_image_folder))

if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Eval
    # ----------------------------
    run_infer(conf)