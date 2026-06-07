"""检查 RESIDE val 目录图片尺寸"""
import os
from PIL import Image

d = os.path.expanduser("~/moxt/DiffUIR/Datasets/Restoration/RESIDE/SOTS/outdoor/test/val/lq")
count = 0
too_small = 0
for f in sorted(os.listdir(d))[:10]:
    im = Image.open(os.path.join(d, f))
    w, h = im.size
    ok = "OK" if min(w, h) >= 128 else "SKIP(<128)"
    print(f"{f}: {w}×{h} {ok}")
    count += 1
    if min(w, h) < 128:
        too_small += 1
print(f"\n{too_small}/{count} images < 128px")
