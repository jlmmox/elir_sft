"""LFG-SFG 全验证集消融：A(normal) B(skip_fmir) C(mmse+noise) D(no_noise) E(scaled)。

用法:
python eval_lfg_ablation_fullval.py CONFIG.yaml CKPT.ckpt
"""

import sys, os, csv, torch
sys.path.insert(0, os.path.dirname(__file__))

from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.models.elir import pos_emb
from ELIR.metrics import calculate_psnr, calculate_ssim
from ELIR.datasets.dataset import get_loader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

yaml_path = sys.argv[1]
ckpt_path = sys.argv[2]
device = "cuda"
out_dir = "eval_lfg_ablation"
os.makedirs(out_dir, exist_ok=True)

# ==================================================
# 1. Load model + ckpt
# ==================================================
with open(yaml_path, encoding="utf-8") as f:
    conf = load_hyperpyyaml(f)

arch_cfg = conf["model_cfg"]["arch_cfg"]
model = get_model(arch_cfg).to(device).eval()
ms = model.state_dict()

ckpt = torch.load(ckpt_path, map_location="cpu")
state = {(k[6:] if k.startswith("model.") else k): v for k, v in ckpt.get("state_dict", ckpt).items()}

for block, prefix in [("state_dict_mmse","mmse."), ("state_dict_fmir","fmir."),
                       ("state_dict_enc","enc."), ("state_dict_dec","dec."),
                       ("state_dict_wavelet","wavelet_stem.")]:
    if block in ckpt:
        for k, v in ckpt[block].items():
            state[k if k.startswith(prefix) else prefix + k] = v

loaded = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
model.load_state_dict(loaded, strict=False)

bad = len(ms) - len(loaded)
print(f"[load] loaded {len(loaded)}/{len(ms)}, bad={bad}")
if bad > 0:
    print(f"[FATAL] {bad} keys not loaded")
    sys.exit(1)

# ==================================================
# 2. 构建验证 dataloader
# ==================================================
# 使用和训练 eval 完全一致的 dataset (含 LQ 直方图均衡)
val_ds_params = dict(conf["dataset_cfg"]["val_dataset"])
val_ds_params["training"] = False
val_ds_params["batch_size"] = 1
val_ds_params["num_workers"] = 0
valloader = get_loader(val_ds_params)

# ==================================================
# 3. 评估所有模式
# ==================================================
MODES = [
    "A_normal",
    "B_skip_fmir",
    "C_mmse_noise_only",
    "D_no_noise_ode",
    "E_scale_025",
    "E_scale_050",
    "E_scale_100",
]

results = {m: {"psnr": [], "ssim": [], "images": []} for m in MODES}
fids = {m: FrechetInceptionDistance(normalize=True).to(device) for m in MODES}

with torch.no_grad():
    for batch in tqdm(valloader, desc="eval"):
        x_lq, hq = batch[0], batch[1]
        x_lq, hq = x_lq.to(device), hq.to(device)

        B, C, oh, ow = hq.shape
        pw, ph = (64 - ow % 64) % 64, (64 - oh % 64) % 64
        xs = torch.nn.functional.pad(x_lq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_lq

        t0 = pos_emb(torch.zeros(xs.shape[0]), 160).to(device)
        wv_cond = model.wavelet_stem(xs, t_emb=t0) if model.wavelet_stem else None
        fmir_c = model._build_fmir_condition(xs, wavelet_cond=wv_cond)
        dec_c  = model._build_fmir_condition(xs, wavelet_cond=None)

        z = model._encode_input(xs)
        nz = model.noise.to(device)
        if nz.shape[-2:] != z.shape[-2:]:
            nz = torch.nn.functional.interpolate(nz, size=z.shape[-2:], mode="bilinear", align_corners=False)

        z_mmse = model.mmse(z)

        def _dec(zt):
            return model._decode_latent(zt, cond=dec_c, x_lq=xs, wavelet_cond=None)[:, :, :oh, :ow].clamp(0, 1)

        def _ode(zt, K_use=None, scale=1.0):
            K_run = K_use if K_use is not None else model.K
            _dt = 1.0 / K_run
            for k in range(K_run):
                t_k = k * _dt
                tt_k = torch.full((zt.shape[0], 1, 1, 1), t_k, device=device, dtype=zt.dtype)
                v_k = model.fmir(zt, pos_emb(t_k, model.t_emb_dim).to(device), cond=fmir_c, t=tt_k)
                zt = zt + scale * _dt * v_k
            return zt

        # A: normal
        y_a = _dec(_ode(z_mmse + nz))
        results["A_normal"]["psnr"].append(calculate_psnr(y_a, hq, test_y_channel=True))
        results["A_normal"]["ssim"].append(calculate_ssim(y_a, hq, test_y_channel=True))
        fids["A_normal"].update(y_a, real=False)
        fids["A_normal"].update(hq, real=True)

        # B: skip_fmir
        y_b = _dec(z_mmse)
        results["B_skip_fmir"]["psnr"].append(calculate_psnr(y_b, hq, test_y_channel=True))
        results["B_skip_fmir"]["ssim"].append(calculate_ssim(y_b, hq, test_y_channel=True))
        fids["B_skip_fmir"].update(y_b, real=False)
        fids["B_skip_fmir"].update(hq, real=True)

        # C: mmse_noise_only
        y_c = _dec(z_mmse + nz)
        results["C_mmse_noise_only"]["psnr"].append(calculate_psnr(y_c, hq, test_y_channel=True))
        results["C_mmse_noise_only"]["ssim"].append(calculate_ssim(y_c, hq, test_y_channel=True))
        fids["C_mmse_noise_only"].update(y_c, real=False)
        fids["C_mmse_noise_only"].update(hq, real=True)

        # D: no_noise_ode
        y_d = _dec(_ode(z_mmse))
        results["D_no_noise_ode"]["psnr"].append(calculate_psnr(y_d, hq, test_y_channel=True))
        results["D_no_noise_ode"]["ssim"].append(calculate_ssim(y_d, hq, test_y_channel=True))
        fids["D_no_noise_ode"].update(y_d, real=False)
        fids["D_no_noise_ode"].update(hq, real=True)

        # E: scaled_ode
        for scl, mode_name in [(0.25, "E_scale_025"), (0.5, "E_scale_050"), (1.0, "E_scale_100")]:
            y_e = _dec(_ode(z_mmse + nz, scale=scl))
            results[mode_name]["psnr"].append(calculate_psnr(y_e, hq, test_y_channel=True))
            results[mode_name]["ssim"].append(calculate_ssim(y_e, hq, test_y_channel=True))
            fids[mode_name].update(y_e, real=False)
            fids[mode_name].update(hq, real=True)

# ==================================================
# 4. 输出
# ==================================================
print(f"\n{'Mode':<18} {'PSNR':>8} {'SSIM':>8} {'FID':>8}")
print("-" * 46)

# per-image PSNR
csv_path = os.path.join(out_dir, "ablation_per_image.csv")
f_csv = open(csv_path, "w", newline="")
w = csv.writer(f_csv)
w.writerow(["mode"] + [f"img_{i}" for i in range(len(results["A_normal"]["psnr"]))])

best_psnr, best_mode = 0, ""
for mode_name in MODES:
    psnr_vals = results[mode_name]["psnr"]
    ssim_vals = results[mode_name]["ssim"]
    avg_p = sum(psnr_vals) / len(psnr_vals)
    avg_s = sum(ssim_vals) / len(ssim_vals)
    fid_val = fids[mode_name].compute().item()
    print(f"{mode_name:<18} {avg_p:>8.2f} {avg_s:>8.4f} {fid_val:>8.1f}")
    w.writerow([mode_name] + [f"{v:.4f}" for v in psnr_vals])
    if avg_p > best_psnr:
        best_psnr, best_mode = avg_p, mode_name
f_csv.close()

print(f"\nBest: {best_mode} ({best_psnr:.2f})")

# 判定
a_p = sum(results["A_normal"]["psnr"]) / len(results["A_normal"]["psnr"])
b_p = sum(results["B_skip_fmir"]["psnr"]) / len(results["B_skip_fmir"]["psnr"])
d_p = sum(results["D_no_noise_ode"]["psnr"]) / len(results["D_no_noise_ode"]["psnr"])
e025_p = sum(results["E_scale_025"]["psnr"]) / len(results["E_scale_025"]["psnr"])

print("\n--- Verdict ---")
if b_p > a_p + 0.3:
    print(f"B(skip_fmir) > A(normal) by {b_p - a_p:.2f}: FMIR 在全验证集拖后腿")
elif abs(b_p - a_p) < 0.3:
    print(f"B≈A (diff={abs(b_p-a_p):.2f}): FMIR 基本无贡献")
if d_p > a_p + 0.3:
    print(f"D(no_noise) > A(normal) by {d_p - a_p:.2f}: noise 注入有负面影响")
if e025_p > a_p + 0.3:
    print(f"E(scale=0.25) > A(normal) by {e025_p - a_p:.2f}: ODE 更新过强")
if a_p >= max(b_p, d_p, e025_p):
    print("A(normal) 最好: FMIR 仍有正贡献")

print(f"\nPer-image PSNR saved: {csv_path}")
