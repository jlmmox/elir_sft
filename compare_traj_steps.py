"""比较 trajectory 每一步和 HQ 的 PSNR/SSIM。"""
import sys
import torch
from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.metrics import calculate_psnr, calculate_ssim
from PIL import Image
from ELIR.datasets.lol import LOLDataset

yaml_path = sys.argv[1]
lq_path = sys.argv[2]
hq_path = sys.argv[3]
k_steps = int(sys.argv[4]) if len(sys.argv) > 4 else None

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)
model = get_model(conf["model_cfg"]["arch_cfg"])
model.eval()

# 对 LQ 做直方图均衡（对齐 eval 管线）
dummy = LOLDataset.__new__(LOLDataset)
lq_eq = dummy._equalize_low_light(Image.open(lq_path).convert("RGB"))
lq = torch.from_numpy(__import__("numpy").array(lq_eq)).float() / 255.0
lq = lq.permute(2, 0, 1).unsqueeze(0)
hq = torch.from_numpy(__import__("numpy").array(
    Image.open(hq_path).convert("RGB"))).float() / 255.0
hq = hq.permute(2, 0, 1).unsqueeze(0)

if k_steps is not None:
    model.K = k_steps
    model.dt = 1.0 / k_steps

with torch.no_grad():
    trajs = model.trajectories_pixel(lq)

print(f"{'Step':<6} {'PSNR':>8} {'SSIM':>8}")
print("-" * 24)
for i, t in enumerate(trajs):
    psnr = calculate_psnr(t, hq, test_y_channel=True)
    ssim = calculate_ssim(t, hq, test_y_channel=True)
    print(f"{i:<6} {psnr:>8.3f} {ssim:>8.4f}")
