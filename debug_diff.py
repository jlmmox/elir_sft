"""比较 eval 保存图和 debug 输出图的像素差异"""
from PIL import Image
import numpy as np
import sys

eval_path = sys.argv[1] if len(sys.argv) > 1 else "runs/eval_e2e_large_lol/samples/sample_000.png"
debug_path = sys.argv[2] if len(sys.argv) > 2 else "a_fwd.png"

a = np.array(Image.open(debug_path)).astype(float)
b = np.array(Image.open(eval_path)).astype(float)
diff = np.abs(a - b)
print(f"max diff: {diff.max():.2f}, mean diff: {diff.mean():.4f} (range 0-255)")
