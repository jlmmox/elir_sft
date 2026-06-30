"""MMSE-only sanity test: 冻结所有, 只训 MMSE, 看 decode PSNR 涨不涨。
若 3k iter 后 MMSE decode PSNR < 20, 说明 MMSE 训练链路本身有 bug。
"""

import sys, torch, os
sys.path.insert(0, os.path.dirname(__file__))

from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.training.losses import charbonnier_loss
from ELIR.metrics import calculate_psnr
from PIL import Image
from torchvision.utils import save_image

yaml_path = sys.argv[1]
lq_path = sys.argv[2]
hq_path = sys.argv[3]

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)

model = get_model(conf["model_cfg"]["arch_cfg"])
model.to('cuda')

# 冻结所有除了 MMSE
for n, p in model.named_parameters():
    p.requires_grad = 'mmse' in n

mmse_params = [p for n, p in model.named_parameters() if 'mmse' in n]
print(f"MMSE params: {len(mmse_params)} ({sum(p.numel() for p in mmse_params):,})")
print(f"requires_grad: {all(p.requires_grad for p in mmse_params)}")

lq = torch.from_numpy(__import__("numpy").array(Image.open(lq_path).convert("RGB"))).float()/255.
hq = torch.from_numpy(__import__("numpy").array(Image.open(hq_path).convert("RGB"))).float()/255.
lq, hq = lq.permute(2,0,1).unsqueeze(0), hq.permute(2,0,1).unsqueeze(0)

lq, hq = lq.to('cuda'), hq.to('cuda')
bs = 8
optimizer = torch.optim.AdamW(mmse_params, lr=1e-4)

model.eval()  # MMSE 也 eval 模式 (BN 不影响)

print(f"\n{'iter':>6} {'L_mmse':>10} {'PSNR_mmse':>10} {'PSNR_hq':>10}")
print("-" * 40)

for it in range(1, 3001):
    # 随机采样 batch (从同一张图复制以模拟小 batch)
    idx = torch.randint(0, 1, (bs,))
    lq_batch = lq[idx]
    hq_batch = hq[idx]

    model.eval()
    with torch.no_grad():
        z_hq = model._encode_input(hq_batch)
        z_lq = model._encode_input(lq_batch)

    model.mmse.train()
    z_mmse = model.mmse(z_lq)
    loss = charbonnier_loss(z_hq, z_mmse)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in mmse_params if p.grad is not None)
    optimizer.step()

    if it == 1 or it % 500 == 0:
        with torch.no_grad():
            model.eval()
            z_hq_val = model._encode_input(hq)
            z_lq_val = model._encode_input(lq)
            z_mmse_val = model.mmse(z_lq_val)

            sp_cond = model._build_fmir_condition(lq, wavelet_cond=None)
            wv_cond = model.wavelet_stem(lq) if model.wavelet_stem else None

            dec_hq = model._decode_latent(z_hq_val, cond=sp_cond, x_lq=lq, wavelet_cond=wv_cond).clamp(0,1)
            dec_mmse = model._decode_latent(z_mmse_val, cond=sp_cond, x_lq=lq, wavelet_cond=wv_cond).clamp(0,1)

            psnr_hq = calculate_psnr(dec_hq, hq, test_y_channel=True)
            psnr_mmse = calculate_psnr(dec_mmse, hq, test_y_channel=True)

        print(f"{it:>6} {loss.item():>10.4f} {psnr_mmse:>10.2f} {psnr_hq:>10.2f}  grad={grad_norm:.4f}")

        if it == 3000:
            save_image(dec_mmse.clamp(0,1), "sanity_mmse_final.png")
            save_image(dec_hq.clamp(0,1), "sanity_hq.png")
            print("saved: sanity_mmse_final.png, sanity_hq.png")
