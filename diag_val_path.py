"""Use real IRSetup.infer path for one image.

Usage:
python diag_val_path_fixed.py CONFIG.yaml LQ.png HQ.png CKPT.ckpt
"""
import os
import sys
import inspect
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from hyperpyyaml import load_hyperpyyaml

from ELIR.irsetup import IRSetup
from ELIR.models.load_model import get_model
from ELIR.metrics import calculate_psnr, calculate_ssim


yaml_path = sys.argv[1]
lq_path = sys.argv[2]
hq_path = sys.argv[3]
ckpt_path = sys.argv[4]


def load_img(path):
    img = Image.open(os.path.expanduser(path)).convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0
    return x.permute(2, 0, 1).unsqueeze(0)



def build_irsetup_from_ckpt(conf, ckpt_path):
    arch_cfg = conf["model_cfg"]["arch_cfg"]
    base_model = get_model(arch_cfg)

    sig = inspect.signature(IRSetup.__init__)
    print("[IRSetup.__init__]", sig)

    fm_cfg = {}
    try:
        fm_cfg = arch_cfg.get("params", {}).get("fm_cfg", {})
    except Exception:
        pass
    fm_cfg.update(conf.get("fm_cfg", {}))

    eval_cfg = conf.get("eval_cfg", {}) or {}
    ema_decay = conf.get("ema_decay", 0.999)

    candidates = [
        {
            "model": base_model,
            "fm_cfg": fm_cfg,
            "eval_cfg": eval_cfg,
            "ema_decay": ema_decay,
            "optimizer": None,
            "scheduler": None,
            "tmodel": None,
            "run_dir": None,
            "save_images": False,
        },
        {
            "model": base_model,
            "fm_cfg": fm_cfg,
            "eval_cfg": {},
            "ema_decay": 0.999,
        },
    ]

    errors = []
    for kwargs in candidates:
        try:
            module = IRSetup.load_from_checkpoint(
                ckpt_path,
                map_location="cuda",
                strict=False,
                **kwargs,
            )
            print("[load] success with kwargs:", list(kwargs.keys()))
            return module
        except Exception as e:
            errors.append((kwargs, repr(e)))

    print("[load] all attempts failed:")
    for kw, err in errors:
        print("  kwargs=", list(kw.keys()), "err=", err[:1000])
    raise SystemExit(1)


with open(yaml_path, encoding="utf-8") as f:
    conf = load_hyperpyyaml(f)

print("[ckpt] loading LightningModule:", ckpt_path)
module = build_irsetup_from_ckpt(conf, ckpt_path)
module = module.cuda().eval()

def strip_model_prefix(sd):
    return {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in sd.items()
    }


def overlay_ema_blocks_into_model(target_model, ckpt):
    # 1. 先加载 raw state_dict，保证未进 EMA 分块的模块也不是随机
    raw = strip_model_prefix(ckpt["state_dict"])
    missing, unexpected = target_model.load_state_dict(raw, strict=False)
    print("[raw -> target] missing:", len(missing), "unexpected:", len(unexpected))

    # 2. 再用 EMA 分块覆盖核心模块
    overlay = {}
    for block, prefix in [
        ("state_dict_mmse", "mmse."),
        ("state_dict_fmir", "fmir."),
        ("state_dict_enc", "enc."),
        ("state_dict_dec", "dec."),
    ]:
        if block not in ckpt:
            print("[ema overlay] missing block:", block)
            continue

        n = 0
        for k, v in ckpt[block].items():
            full_k = k if k.startswith(prefix) else prefix + k
            overlay[full_k] = v
            n += 1
        print(f"[ema overlay] {block}: {n} keys")

    missing, unexpected = target_model.load_state_dict(overlay, strict=False)
    print("[ema overlay] missing:", len(missing), "unexpected:", len(unexpected))

    mmse_keys = [k for k in overlay if k.startswith("mmse.")]
    print("[ema overlay] mmse keys:", len(mmse_keys))


ckpt_raw = torch.load(ckpt_path, map_location="cpu")

if getattr(module, "ema", None) is not None:
    print("[patch] overlay EMA blocks into module.ema.model")
    overlay_ema_blocks_into_model(module.ema.model, ckpt_raw)
else:
    print("[patch] no module.ema, overlay EMA blocks into module.model")
    overlay_ema_blocks_into_model(module.model, ckpt_raw)
has_ema = getattr(module, "ema", None) is not None
print("[module] ema exists:", has_ema)
print("[module] model exists:", getattr(module, "model", None) is not None)

lq = load_img(lq_path).cuda()
hq = load_img(hq_path).cuda()

with torch.no_grad():
    # Official IRSetup path. irsetup.py decides EMA vs raw inside infer().
    y = module.infer(lq).clamp(0, 1)

psnr = calculate_psnr(y, hq, test_y_channel=True)
ssim = calculate_ssim(y, hq, test_y_channel=True)

print("\nIRSetup.infer result")
print("--------------------")
print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.4f}")
save_image(y, "diag_irsetup_infer.png")
print("saved: diag_irsetup_infer.png")

# Compare direct active model call if available.
active_model = module.ema.model if has_ema else module.model
active_model = active_model.cuda().eval()

with torch.no_grad():
    y_direct = active_model.inference(lq, use_tta=False).clamp(0, 1)

p2 = calculate_psnr(y_direct, hq, test_y_channel=True)
s2 = calculate_ssim(y_direct, hq, test_y_channel=True)

print("\nActive model direct inference")
print("-----------------------------")
print(f"PSNR: {p2:.2f}")
print(f"SSIM: {s2:.4f}")
save_image(y_direct, "diag_active_direct.png")
print("saved: diag_active_direct.png")

# Optional MMSE diagnostic on the exact active model used for inference.
with torch.no_grad():
    if all(hasattr(active_model, name) for name in ["_encode_input", "_decode_latent", "mmse"]):
        z_hq = active_model._encode_input(hq)
        z_lq = active_model._encode_input(lq)
        z_mmse = active_model.mmse(z_lq)

        cond = active_model._build_fmir_condition(lq, wavelet_cond=None)
        wavelet_cond = active_model.wavelet_stem(lq) if getattr(active_model, "wavelet_stem", None) else None

        dec_hq = active_model._decode_latent(
            z_hq,
            cond=cond,
            x_lq=lq,
            wavelet_cond=wavelet_cond,
        ).clamp(0, 1)

        dec_mmse = active_model._decode_latent(
            z_mmse,
            cond=cond,
            x_lq=lq,
            wavelet_cond=wavelet_cond,
        ).clamp(0, 1)

        print("\nActive model latent diagnostic")
        print("------------------------------")
        print(f"SFT_dec(HQ): {calculate_psnr(dec_hq, hq, test_y_channel=True):.2f}")
        print(f"SFT_dec(MM): {calculate_psnr(dec_mmse, hq, test_y_channel=True):.2f}")
        save_image(dec_hq, "diag_active_sft_hq.png")
        save_image(dec_mmse, "diag_active_sft_mmse.png")
        print("saved: diag_active_sft_hq.png, diag_active_sft_mmse.png")
    else:
        print("[diag] active_model lacks latent diagnostic methods")