"""
最小对比：rain vs blur，同一 TAESD3 encoder，看 latent 差异量
"""
import torch, torch.nn.functional as F
import sys, os
from diffusers import AutoencoderTiny

device = "cuda" if torch.cuda.is_available() else "cpu"
taesd3 = AutoencoderTiny.from_pretrained("madebyollin/taesd3").to(device).eval()
enc = taesd3.encoder
dec = taesd3.decoder

# 手动加载一对 rain LQ/HQ
from ELIR.datasets.dataset import get_loader
from hyperpyyaml import load_hyperpyyaml
from args_handler import argument_handler

yaml_path, _ = argument_handler()
if yaml_path is None:
    print("Usage: python diag_rain_vs_blur.py -y configs/derain/eval.yaml")
    sys.exit(1)

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)

val_cfg = dict(conf["dataset_cfg"]["val_dataset"])
val_cfg.update({"batch_size": 1, "num_workers": 0})
loader = get_loader(val_cfg)

print(f"Dataset: {val_cfg['path']}")
print(f"Samples: {len(loader.dataset)}")

from ELIR.metrics import calculate_psnr

for idx, (x_lq, x_hq) in enumerate(loader):
    if idx >= 5:
        break
    x_lq, x_hq = x_lq.to(device), x_hq.to(device)
    _, _, oh, ow = x_hq.shape

    with torch.no_grad():
        # === TAESD roundtrip (upper bound) ===
        z_hq_raw = enc(x_hq)
        y_roundtrip = dec(z_hq_raw).clamp(0, 1)[:, :, :oh, :ow]
        psnr_roundtrip = float(calculate_psnr(y_roundtrip, x_hq, test_y_channel=True))

        # === Latent difference: LQ vs HQ ===
        z_lq_raw = enc(x_lq)
        z_hq_raw = enc(x_hq)
        ne = max(z_hq_raw[0].numel(), 1)**0.5
        delta_norm = (z_lq_raw - z_hq_raw).flatten(1).norm(dim=1).mean().item() / ne
        cos_lq_hq = F.cosine_similarity(z_lq_raw.flatten(1), z_hq_raw.flatten(1), dim=1).mean().item()
        eps = 1e-6
        charb_lq_hq = torch.sqrt((z_lq_raw - z_hq_raw).pow(2) + eps).mean().item()

        # === Simple MSE PSNR (no Y-channel, no BGR) as sanity check ===
        mse_simple = F.mse_loss(x_lq.clamp(0,1), x_hq.clamp(0,1))
        psnr_simple = 10 * torch.log10(1.0 / mse_simple).item()

        # === Input PSNR (DiffUIR method) ===
        psnr_input = float(calculate_psnr(x_lq.clamp(0,1), x_hq.clamp(0,1), test_y_channel=True))

        print(f"\n[{idx}] shape={oh}×{ow}")
        print(f"  TAESD roundtrip PSNR (ceiling):    {psnr_roundtrip:.2f} dB")
        print(f"  Input LQ→HQ PSNR (simple MSE):      {psnr_simple:.2f} dB")
        print(f"  Input LQ→HQ PSNR (DiffUIR Y-chan):  {psnr_input:.2f} dB")
        print(f"  Latent ||LQ-HQ|| RMS:               {delta_norm:.5f}")
        print(f"  Latent LQ↔HQ cosine:                {cos_lq_hq:.4f}")
        print(f"  Latent LQ↔HQ charb:                 {charb_lq_hq:.5f}")
        print(f"  z_hq stats: mean={z_hq_raw.mean().item():.4f}  std={z_hq_raw.std().item():.4f}")
        print(f"  z_lq stats: mean={z_lq_raw.mean().item():.4f}  std={z_lq_raw.std().item():.4f}")

        # === 关键判断 ===
        print(f"  → ", end="")
        if cos_lq_hq > 0.99:
            print("⚠️  LQ≈HQ in latent — rain info LOST in 8× downsampling. FMIR has nothing to learn.")
        elif cos_lq_hq > 0.95:
            print("⚠️  LQ and HQ latents very similar. FMIR target residual is tiny.")
        else:
            print(f"✓ LQ↔HQ latent difference exists (cos={cos_lq_hq:.4f}). FMIR has meaningful target.")
