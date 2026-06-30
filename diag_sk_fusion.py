"""诊断 SK Fusion 目标质量 — 跑一个 batch 打印所有关键数值。"""
import sys, torch
from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.models.elir import pos_emb
from ELIR.metrics import calculate_psnr
from ELIR.training.losses import charbonnier_loss
from PIL import Image

yaml_path = sys.argv[1]
lq_path = sys.argv[2]
hq_path = sys.argv[3]

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)

model = get_model(conf["model_cfg"]["arch_cfg"])
model.eval()

lq = torch.from_numpy(__import__("numpy").array(Image.open(lq_path).convert("RGB"))).float()/255.
hq = torch.from_numpy(__import__("numpy").array(Image.open(hq_path).convert("RGB"))).float()/255.
lq, hq = lq.permute(2,0,1).unsqueeze(0), hq.permute(2,0,1).unsqueeze(0)

model.to('cuda')
lq, hq = lq.to('cuda'), hq.to('cuda')

with torch.no_grad():
    z_hq = model._encode_input(hq)
    z_lq = model._encode_input(lq)
    z_mmse = model.mmse(z_lq)

    spatial_cond = model._build_fmir_condition(lq, wavelet_cond=None)
    t_emb_init = pos_emb(torch.zeros(1), 160).to('cuda')
    wavelet_cond = model.wavelet_stem(lq, t_emb=t_emb_init) if model.wavelet_stem else None

    def decode(z):
        return model._decode_latent(z, cond=spatial_cond, x_lq=lq, wavelet_cond=wavelet_cond).clamp(0,1)

    dec_hq = decode(z_hq)
    dec_mmse = decode(z_mmse)

    print(f"\n--- 基本 PSNR 诊断 ---")
    print(f"PSNR(TAESD_dec(X_hq), HQ): {calculate_psnr(dec_hq, hq, test_y_channel=True):.2f}")
    print(f"PSNR(TAESD_dec(X_mmse), HQ): {calculate_psnr(dec_mmse, hq, test_y_channel=True):.2f}")
    print(f"Charb(X_mmse, X_hq): {charbonnier_loss(z_mmse, z_hq).item():.6f}")
    print(f"std(X_hq): {z_hq.std().item():.4f}")
    print(f"std(z_mmse): {z_mmse.std().item():.4f}")
