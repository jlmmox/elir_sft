"""对比 forward 和 trajectories_pixel 最后一步的输出差异。"""
import torch
from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from torchvision.utils import save_image
from PIL import Image
import sys

yaml_path = sys.argv[1] if len(sys.argv) > 1 else "configs/elir_eval_e2e_large_lol.yaml"
img_path = sys.argv[2] if len(sys.argv) > 2 else "test.png"
use_tta = "--tta" in sys.argv

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)
model = get_model(conf["model_cfg"]["arch_cfg"])
model.eval()

lq = torch.from_numpy(__import__("numpy").array(Image.open(img_path).convert("RGB"))).float() / 255.0
# 和训练/eval 一样做 YCrCb 直方图均衡
from ELIR.datasets.lol import LOLDataset
dummy = LOLDataset.__new__(LOLDataset)
lq_eq = dummy._equalize_low_light(Image.open(img_path).convert("RGB"))
lq = torch.from_numpy(__import__("numpy").array(lq_eq)).float() / 255.0
lq = lq.permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    y_fwd = model.inference(lq, use_tta=use_tta)
    y_traj = model.trajectories_pixel(lq)

diff = (y_fwd - y_traj[-1]).abs()
print(f"steps: {len(y_traj)}, max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")

save_image(y_fwd, "a_fwd.png")
save_image(y_traj[-1], "b_traj_last.png")
save_image(y_traj[0], "c_traj_first.png")
print("saved a_fwd.png b_traj_last.png c_traj_first.png")
