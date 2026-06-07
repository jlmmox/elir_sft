"""采集时间门控激活曲线并画图：验证流匹配的频率解耦特性。

用法:
  python plot_gate_curve.py -y configs/elir_eval_e2e_large_lol.yaml \
      --img ~/moxt/DiffUIR/Datasets/Restoration/LOL/test/low/1.png \
      --points 10 --out gate_curve.png
"""

import sys
import torch
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from PIL import Image
from torchvision.transforms import v2

# 解析参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml_path", type=str, required=True)
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--points", type=int, default=10)
parser.add_argument("--out", type=str, default="gate_curve.png")
args = parser.parse_args()

# 加载模型
with open(args.yaml_path) as f:
    conf = load_hyperpyyaml(f)
model = get_model(conf["model_cfg"]["arch_cfg"])
model.eval()

# 加载图片并编码
lq = v2.ToTensor()(Image.open(args.img).convert("RGB")).unsqueeze(0)
from ELIR.datasets.lol import LOLDataset
dummy = LOLDataset.__new__(LOLDataset)
lq_eq = dummy._equalize_low_light(Image.open(args.img).convert("RGB"))
lq = v2.ToTensor()(lq_eq).unsqueeze(0)

# 准备 cond（对齐 FMIR 条件路径）
with torch.no_grad():
    z = model._encode_input(lq)
cond = None
if hasattr(model.fmir, "make_condition"):
    cond = model.fmir.make_condition(lq)
elif hasattr(model.fmir, "condition_stem"):
    cond = model.fmir.condition_stem(lq)

# 采集门控曲线
ts, gates = model.fmir.collect_gate_curve(z, cond=cond, num_points=args.points)

# 打印
print(f"{'t':>6}  {'gate':>8}")
print("─" * 16)
for t, g in zip(ts, gates):
    print(f"{t:>6.3f}  {g:>8.4f}")

# 画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
plt.plot(ts, gates, "o-", color="steelblue", linewidth=2, markersize=8)
plt.xlabel("Flow Matching Time t (0 → 1)", fontsize=12)
plt.ylabel("Mean Gate Activation γ(t)", fontsize=12)
plt.title("Time-Gated Dilated Convolution: Learned Frequency Decoupling", fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"\nSaved: {args.out}")
