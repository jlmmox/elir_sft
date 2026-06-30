"""
ELIR 去雨根因诊断：latent丢雨 + 高频旁路重注入 双重验证
==============================================================
用法:
  python diag_rain_rootcause.py -y configs/derain/eval.yaml --ckpt <path> [--out ./runs/diag_rc]
"""
import os, sys, math, csv, argparse, itertools
import torch, torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torchvision.utils import save_image
import cv2

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, default="configs/derain/eval.yaml")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--out", type=str, default="./runs/diag_rc")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--max_samples", type=int, default=10)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ---------------------------------------------------------------------------
from hyperpyyaml import load_hyperpyyaml
with open(args.yaml) as f:
    conf = load_hyperpyyaml(f)

eval_cfg = conf.get("eval_cfg", {})
model_cfg = conf.get("model_cfg", {})
arch_cfg = model_cfg.get("arch_cfg", {})
fm_cfg = conf.get("fm_cfg", {})
dataset_cfg = conf.get("dataset_cfg", {})
ckpt_path = args.ckpt or eval_cfg.get("ckpt_path") or arch_cfg.get("path")
print(f"ckpt={ckpt_path}")

# ---------------------------------------------------------------------------
from ELIR.models.load_model import get_model
from ELIR.irsetup import IRSetup
from ELIR.models.elir import pos_emb
from ELIR.metrics import calculate_psnr, calculate_ssim, _tensor2numpy_single, to_y_channel

# ---- 原始 TAESD3 ----
from diffusers import AutoencoderTiny
taesd3 = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_enc = taesd3.encoder.to(device).eval()
raw_dec = taesd3.decoder.to(device).eval()
for p in itertools.chain(raw_enc.parameters(), raw_dec.parameters()):
    p.requires_grad = False

# ---- 构建模型 ----
arch_cfg_no_path = dict(arch_cfg)
arch_cfg_no_path["path"] = None
model = get_model(arch_cfg_no_path)

# ---- Teacher ----
class _TeacherW:
    def __init__(self, enc): self.encoder = enc
    def eval(self): self.encoder.eval()
    def to(self, d): self.encoder.to(d); return self
    def parameters(self): return self.encoder.parameters()
tmodel = _TeacherW(raw_enc)

# ---- 加载 ckpt ----
if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    print(f"ckpt global_step={ckpt.get('global_step','N/A')}, epoch={ckpt.get('epoch','N/A')}")
    sd = ckpt.get("state_dict", {})
    if sd:
        cleaned = OrderedDict((k[len("model."):] if k.startswith("model.") else k, v) for k, v in sd.items())
        m, u = model.load_state_dict(cleaned, strict=False)
        print(f"[main] missing={len(m)}, unexpected={len(u)}")
    for key, attr in [("state_dict_fmir","fmir"),("state_dict_mmse","mmse"),
                       ("state_dict_enc","enc"),("state_dict_dec","dec"),
                       ("state_dict_wavelet","wavelet_stem")]:
        if key in ckpt and ckpt[key]:
            mm, uu = getattr(model, attr).load_state_dict(ckpt[key], strict=False)
            frac = 1.0-len(mm)/max(len(getattr(model,attr).state_dict()),1)
            print(f"[{key}] → {attr}: missing={len(mm)}, unexpected={len(uu)}, loaded={frac:.1%}")
model.to(device).eval()

# ---- IRSetup ----
setup = IRSetup(model=model, fm_cfg=fm_cfg, eval_cfg=dict(eval_cfg, metrics=["psnr"]),
                run_dir=args.out, save_images=False, optimizer=None, scheduler=None, tmodel=tmodel)
setup.to(device).eval()
active = setup.ema.model if setup.ema else model
print(f"EMA={setup.ema is not None}")

# ---- 数据集 ----
from ELIR.datasets.dataset import get_loader
vcfg = dict(dataset_cfg.get("val_dataset", {}))
vcfg.update({"batch_size": 1, "num_workers": 0})
loader = get_loader(vcfg)
print(f"dataset: {len(loader.dataset)} samples")

# ===================================================================
# 辅助函数
# ===================================================================
def _charb(a, b, eps=1e-6):
    return torch.sqrt((a-b).pow(2)+eps).mean().item()

def _cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()

def _rms(t):
    return t.flatten(1).norm(dim=1).mean().item()/max(t[0].numel(),1)**0.5

def _to_numpy(t, oh, ow):
    """torch [1,C,H,W] RGB [0,1] → numpy HWC BGR uint8, cropped to oh×ow"""
    t = t[:,:,:oh,:ow].clamp(0,1)
    return _tensor2numpy_single(t)

def _psnr_np(a, b):
    """两个 HWC BGR uint8 numpy → Y-channel PSNR"""
    ay = to_y_channel(a.astype(np.float64))[...,0]
    by = to_y_channel(b.astype(np.float64))[...,0]
    mse = np.mean((ay-by)**2)
    return float(20*np.log10(255/np.sqrt(mse))) if mse>0 else 100.0

def _dwt_hf_energy(img_np_bgr):
    """BGR uint8 → 单层 Haar DWT → 返回 LH+HL+HH 总能量"""
    gray = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h2, w2 = h//2, w//2
    ll = gray[0:h2*2:2, 0:w2*2:2] + gray[1:h2*2:2, 0:w2*2:2] + gray[0:h2*2:2, 1:w2*2:2] + gray[1:h2*2:2, 1:w2*2:2]
    lh = gray[0:h2*2:2, 0:w2*2:2] + gray[1:h2*2:2, 0:w2*2:2] - gray[0:h2*2:2, 1:w2*2:2] - gray[1:h2*2:2, 1:w2*2:2]
    hl = gray[0:h2*2:2, 0:w2*2:2] - gray[1:h2*2:2, 0:w2*2:2] + gray[0:h2*2:2, 1:w2*2:2] - gray[1:h2*2:2, 1:w2*2:2]
    hh = gray[0:h2*2:2, 0:w2*2:2] - gray[1:h2*2:2, 0:w2*2:2] - gray[0:h2*2:2, 1:w2*2:2] + gray[1:h2*2:2, 1:w2*2:2]
    lh_e = float(np.mean(lh**2)); hl_e = float(np.mean(hl**2)); hh_e = float(np.mean(hh**2))
    ll_e = float(np.mean(ll**2))
    return ll_e, lh_e, hl_e, hh_e

# ===================================================================
# 全局条件来源审计
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 2: CONDITION SOURCE AUDIT")
print(f"{'='*70}")

# 检查 wavelet_stem 的调用路径
print("""
[Code Audit] Condition data flow in forward() + loss:

1. wavelet_stem(x_lq, t_emb) → wavelet_cond  [来自 LQ ✓]
   elir.py:1044-1049  forward()

2. _build_fmir_condition(x_lq, wavelet_cond=wavelet_cond) → fmir_cond  [来自 LQ ✓]
   elir.py:1052  forward()

3. _build_fmir_condition(x_lq, wavelet_cond=None) → decoder_cond  [来自 LQ ✓]
   elir.py:1055  forward()

4. _prepare_fmir_condition(model, x_lq) → cond  [来自 LQ ✓]
   losses.py:122-138  e2e_gan_loss()

5. _prepare_decoder_condition(model, x_lq, cond) → dec_cond  [来自 LQ ✓]
   losses.py:141-154  e2e_gan_loss()

6. _decode_latent(z, cond=decoder_cond, x_lq=x_lq, wavelet_cond=None)  [latent来自FMIR, cond来自LQ]
   elir.py:1097  forward()

7. x_hq (GT) usage: only as loss target, NEVER as condition input ✓
   losses.py: e2e_gan_loss → X_hq = teacher_enc(x_hq).detach() for flow target
   x_hq used in: charb(pred, x_hq), ssim(pred, x_hq), gan_d(x_hq)

8. wavelet_stem frequency bands: LL, LH, HL, HH (Haar DWT)
   wavelet_stem.py: DWT → per-band processing → cond pyramid
   LL=low-freq base, LH/HL/HH=high-freq detail from LQ image

VERDICT: All conditions come from x_lq (rainy image).
         x_hq is used ONLY as loss target.
         No HQ information leaks into condition path.
         BUT: wavelet_stem's LH/HL/HH bands carry rain-streak HF
              from x_lq → these are injected into both FMIR cond
              AND decoder cond (if wavelet_cond used for decoder).
              In current config: wavelet_cond only for FMIR,
              decoder_cond is pure spatial CNN (wavelet_cond=None).
""")

# Check if wavelet ever leaks into decoder path
has_wavelet = active.wavelet_stem is not None
fmir_use_wavelet = getattr(active, "use_wavelet_fmir_cond", False)
fuse_with_spatial = getattr(active, "wavelet_fuse_with_spatial", False)
decoder_cond_fusions = getattr(active, "decoder_cond_fusions", None)
print(f"wavelet_stem exists: {has_wavelet}")
print(f"fmir_use_wavelet_cond: {fmir_use_wavelet}")
print(f"wavelet_fuse_with_spatial (for decoder): {fuse_with_spatial}")
print(f"decoder_cond_fusions exists: {decoder_cond_fusions is not None}")
print(f"→ Wavelet only enters FMIR cond, NOT decoder spatial cond in current config.")

# ===================================================================
# 收集所有 sample 数据
# ===================================================================
samples = []
for idx, (x_lq, x_hq) in enumerate(loader):
    if idx >= args.max_samples:
        break
    oh, ow = x_hq.shape[2], x_hq.shape[3]
    ph, pw = (64-oh%64)%64, (64-ow%64)%64
    x_lq_dev = x_lq.to(device)
    x_hq_dev = x_hq.to(device)
    x_lq_p = F.pad(x_lq_dev, (0,pw,0,ph), mode="reflect") if ph or pw else x_lq_dev
    x_hq_p = F.pad(x_hq_dev, (0,pw,0,ph), mode="reflect") if ph or pw else x_hq_dev

    with torch.no_grad():
        # ---- Step 1: Teacher vs Student latent ----
        z_teacher_lq = raw_enc(x_lq_p).float()
        z_teacher_hq = raw_enc(x_hq_p).float()
        z_student_lq = active._encode_input(x_lq_p)
        z_student_hq = active._encode_input(x_hq_p)

        # ---- Step 5: latent 距离 ----
        z_mmse = active.mmse(z_student_lq)
        noise = active._inference_noise(z_mmse, device)
        z0 = z_mmse + noise

        # FMIR ODE
        wc_full = None
        if active.wavelet_stem is not None:
            t_emb_init = pos_emb(torch.zeros(x_lq_p.shape[0]), active.t_emb_dim).to(device)
            wc_full = active.wavelet_stem(x_lq_p, t_emb=t_emb_init)
        fcond = active._build_fmir_condition(x_lq_p, wavelet_cond=wc_full)
        fcond = active._augment_cond_with_latent(fcond, z_student_lq, z_mmse)
        K = int(active.K); dt_val = 1.0/K
        z_final = z0.clone()
        for k in range(K):
            tv = torch.full((z_final.shape[0],), k*dt_val, device=device, dtype=z_final.dtype)
            vk = active.fmir(z_final, pos_emb(tv, active.t_emb_dim).to(device),
                            cond=fcond, t=tv[:,None,None,None])
            z_final = z_final + dt_val * vk

        # ---- Step 3: 正常推理 baseline ----
        dec_cond = active._build_fmir_condition(x_lq_p, wavelet_cond=None)
        y_normal = active._decode_latent(z_final, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)
        y_normal = y_normal[:,:,:oh,:ow].clamp(0,1)

        # ---- Step 3: Wavelet ablation variants ----
        def _run_fmir_latent(wc_mod):
            """Run FMIR ODE with given wavelet cond, return refined latent (NOT decoded image)"""
            fc = active._build_fmir_condition(x_lq_p, wavelet_cond=wc_mod)
            fc = active._augment_cond_with_latent(fc, z_student_lq, z_mmse)
            zf = z0.clone()
            for k in range(K):
                tv = torch.full((zf.shape[0],), k*dt_val, device=device, dtype=zf.dtype)
                vk = active.fmir(zf, pos_emb(tv, active.t_emb_dim).to(device),
                                cond=fc, t=tv[:,None,None,None])
                zf = zf + dt_val * vk
            return zf

        def _decode_z(z_lat, cond):
            y = active._decode_latent(z_lat, cond=cond, x_lq=x_lq_p, wavelet_cond=None)
            return y[:,:,:oh,:ow].clamp(0,1)

        # A: Normal (full wavelet FMIR + spatial decoder cond)
        y_A = y_normal

        # C: wavelet_cond = None in FMIR + spatial decoder cond
        z_C = _run_fmir_latent(None)
        y_C = _decode_z(z_C, dec_cond)

        # F: full wavelet FMIR + ZERO decoder cond
        zero_cond = {}
        if dec_cond is not None:
            for k, v in dec_cond.items():
                zero_cond[k] = torch.zeros_like(v)
        y_F = _decode_z(z_final, zero_cond)

        # G: wavelet=None FMIR + ZERO decoder cond  (both conditions removed)
        z_G = _run_fmir_latent(None)
        y_G = _decode_z(z_G, zero_cond)

        # D/E: scaled wavelet cond in FMIR + normal decoder cond
        if wc_full is not None:
            wc_D = {k: v * 0.1 for k, v in wc_full.items()}
            wc_E = {k: v * 0.25 for k, v in wc_full.items()}
            y_D = _decode_z(_run_fmir_latent(wc_D), dec_cond)
            y_E = _decode_z(_run_fmir_latent(wc_E), dec_cond)
        else:
            y_D, y_E = y_C, y_C

        # ---- 计算所有 PSNR ----
        samples.append({
            "idx": idx, "oh": oh, "ow": ow,
            "x_lq": x_lq_dev, "x_hq": x_hq_dev,
            # Step 1
            "t_lq_hq_charb": _charb(z_teacher_lq, z_teacher_hq),
            "t_lq_hq_cos": _cos(z_teacher_lq, z_teacher_hq),
            "t_lq_hq_delta": _rms(z_teacher_lq - z_teacher_hq),
            "s_lq_hq_charb": _charb(z_student_lq, z_student_hq),
            "s_lq_hq_cos": _cos(z_student_lq, z_student_hq),
            "s_lq_hq_delta": _rms(z_student_lq - z_student_hq),
            # Step 5
            "norm_lq_hq": _rms(z_student_lq - z_student_hq),
            "norm_mmse_hq": _rms(z_mmse - z_student_hq),
            "norm_final_hq": _rms(z_final - z_student_hq),
            "norm_final_mmse": _rms(z_final - z_mmse),
            "cos_lq_hq": _cos(z_student_lq, z_student_hq),
            "cos_mmse_hq": _cos(z_mmse, z_student_hq),
            "cos_final_hq": _cos(z_final, z_student_hq),
            # Step 3 results
            "y_A": y_A, "y_C": y_C, "y_D": y_D, "y_E": y_E, "y_F": y_F, "y_G": y_G,
        })

# ===================================================================
# 计算 PSNR & SSIM & DWT
# ===================================================================
print(f"\n{'='*70}")
print(f"RESULTS TABLE")
print(f"{'='*70}")

header = ["idx","A_full","C_noWav","D_wav01","E_wav025","F_dec0","G_noWav_dec0",
          "t_cos","t_delta","s_cos","s_delta","HF_o2L","HF_o2H"]
print("  " + " | ".join(f"{h:>8}" for h in header[:8]))
print("  " + "-"*80)

all_results = []
for s in samples:
    oh, ow = s["oh"], s["ow"]
    x_hq = s["x_hq"]
    x_lq = s["x_lq"]

    # PSNR
    psnr_A = float(calculate_psnr(s["y_A"], x_hq, test_y_channel=True))
    psnr_C = float(calculate_psnr(s["y_C"], x_hq, test_y_channel=True))
    psnr_D = float(calculate_psnr(s["y_D"], x_hq, test_y_channel=True))
    psnr_E = float(calculate_psnr(s["y_E"], x_hq, test_y_channel=True))
    psnr_F = float(calculate_psnr(s["y_F"], x_hq, test_y_channel=True))
    psnr_G = float(calculate_psnr(s["y_G"], x_hq, test_y_channel=True))

    # DWT HF energy
    np_lq = _to_numpy(x_lq, oh, ow)
    np_hq = _to_numpy(x_hq, oh, ow)
    np_out = _to_numpy(s["y_A"], oh, ow)
    _, lh_lq, hl_lq, hh_lq = _dwt_hf_energy(np_lq)
    _, lh_hq, hl_hq, hh_hq = _dwt_hf_energy(np_hq)
    _, lh_out, hl_out, hh_out = _dwt_hf_energy(np_out)
    hf_lq = lh_lq + hl_lq + hh_lq
    hf_hq = lh_hq + hl_hq + hh_hq
    hf_out = lh_out + hl_out + hh_out

    # HF similarity
    # Charb between HF bands
    eps = 1e-6
    charb_hf_out_lq = np.sqrt((np.array([lh_out,hl_out,hh_out]) - np.array([lh_lq,hl_lq,hh_lq]))**2 + eps).mean()
    charb_hf_out_hq = np.sqrt((np.array([lh_out,hl_out,hh_out]) - np.array([lh_hq,hl_hq,hh_hq]))**2 + eps).mean()
    cos_hf_out_lq = np.dot([lh_out,hl_out,hh_out],[lh_lq,hl_lq,hh_lq])/(np.linalg.norm([lh_out,hl_out,hh_out])*np.linalg.norm([lh_lq,hl_lq,hh_lq])+1e-8)
    cos_hf_out_hq = np.dot([lh_out,hl_out,hh_out],[lh_hq,hl_hq,hh_hq])/(np.linalg.norm([lh_out,hl_out,hh_out])*np.linalg.norm([lh_hq,hl_hq,hh_hq])+1e-8)

    all_results.append({
        **s,
        "psnr_A": psnr_A, "psnr_C": psnr_C, "psnr_D": psnr_D, "psnr_E": psnr_E, "psnr_F": psnr_F, "psnr_G": psnr_G,
        "hf_lq": hf_lq, "hf_hq": hf_hq, "hf_out": hf_out,
        "charb_hf_out_lq": charb_hf_out_lq, "charb_hf_out_hq": charb_hf_out_hq,
        "cos_hf_out_lq": cos_hf_out_lq, "cos_hf_out_hq": cos_hf_out_hq,
    })

    print(f"  [{s['idx']:2d}] {psnr_A:6.2f}|{psnr_C:6.2f}|{psnr_D:6.2f}|{psnr_E:6.2f}|{psnr_F:6.2f}|{psnr_G:6.2f}| "
          f"{s['t_lq_hq_cos']:.3f}|{s['t_lq_hq_delta']:.3f}|{s['s_lq_hq_cos']:.3f}|{s['s_lq_hq_delta']:.3f}| "
          f"{cos_hf_out_lq:.3f}|{cos_hf_out_hq:.3f}")

# ===================================================================
# 汇总
# ===================================================================
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

avg = lambda key: np.mean([r[key] for r in all_results])

print(f"\n--- Step 1: Teacher vs Student Latent Rain Info ---")
print(f"  teacher_lq↔teacher_hq: charb={avg('t_lq_hq_charb'):.4f}  cos={avg('t_lq_hq_cos'):.4f}  delta={avg('t_lq_hq_delta'):.4f}")
print(f"  student_lq↔student_hq: charb={avg('s_lq_hq_charb'):.4f}  cos={avg('s_lq_hq_cos'):.4f}  delta={avg('s_lq_hq_delta'):.4f}")
t_diff = avg('t_lq_hq_delta')
s_diff = avg('s_lq_hq_delta')
if t_diff > 2*s_diff:
    print(f"  → Encoder training caused rain info loss (teacher preserves {t_diff/s_diff:.1f}x more)")
elif t_diff < 0.5 and s_diff < 0.5:
    print(f"  → BOTH teacher and student lose rain info → TAESD 8× is the physical bottleneck")
else:
    print(f"  → Both preserve some rain info (teacher={t_diff:.4f}, student={s_diff:.4f})")

print(f"\n--- Step 3: Wavelet/Condition Ablation PSNR ---")
print(f"  A_normal(wavelet):        {avg('psnr_A'):.2f} dB")
print(f"  C_noWavelet(spatial only): {avg('psnr_C'):.2f} dB  Δ={avg('psnr_C')-avg('psnr_A'):+.2f}")
print(f"  D_wavelet×0.1:            {avg('psnr_D'):.2f} dB  Δ={avg('psnr_D')-avg('psnr_A'):+.2f}")
print(f"  E_wavelet×0.25:           {avg('psnr_E'):.2f} dB  Δ={avg('psnr_E')-avg('psnr_A'):+.2f}")
print(f"  F_decZeroCond:            {avg('psnr_F'):.2f} dB  Δ={avg('psnr_F')-avg('psnr_A'):+.2f}")

print(f"\n--- Step 4: HF Correlation ---")
print(f"  HF energy: LQ={avg('hf_lq'):.2f}  HQ={avg('hf_hq'):.2f}  output={avg('hf_out'):.2f}")
print(f"  charb(HF_out, HF_LQ)={avg('charb_hf_out_lq'):.4f}   charb(HF_out, HF_HQ)={avg('charb_hf_out_hq'):.4f}")
print(f"  cos(HF_out, HF_LQ)={avg('cos_hf_out_lq'):.4f}      cos(HF_out, HF_HQ)={avg('cos_hf_out_hq'):.4f}")
if avg('cos_hf_out_lq') > avg('cos_hf_out_hq') + 0.1:
    print(f"  → ⚠️  Output HF is CLOSER to LQ HF than HQ HF: rain streaks RE-INJECTED!")
else:
    print(f"  → Output HF closer to HQ — rain removal in HF band is working")

print(f"\n--- Step 5: Latent Trunk Identity Check ---")
print(f"  ||LQ-HQ|| ={avg('norm_lq_hq'):.4f}   cos(LQ,HQ)={avg('cos_lq_hq'):.4f}")
print(f"  ||MMSE-HQ||={avg('norm_mmse_hq'):.4f}   cos(MMSE,HQ)={avg('cos_mmse_hq'):.4f}")
print(f"  ||final-HQ||={avg('norm_final_hq'):.4f}   cos(final,HQ)={avg('cos_final_hq'):.4f}")
print(f"  ||final-MMSE||={avg('norm_final_mmse'):.4f}")

# ===================================================================
# 保存可视化
# ===================================================================
for r in all_results[:args.max_samples]:
    idx = r["idx"]
    oh, ow = r["oh"], r["ow"]
    x_hq = r["x_hq"]
    x_lq = r["x_lq"]

    # 横向拼图：LQ | HQ | output_A | output_C | output_F | residual_A
    np_lq = _to_numpy(x_lq, oh, ow)
    np_hq = _to_numpy(x_hq, oh, ow)
    np_A   = _to_numpy(r["y_A"], oh, ow)
    np_C   = _to_numpy(r["y_C"], oh, ow)
    np_F   = _to_numpy(r["y_F"], oh, ow)

    # residual = |output - HQ| enhanced
    res_A = cv2.absdiff(np_A, np_hq)
    res_A = cv2.convertScaleAbs(res_A, alpha=3.0)  # amplify for visibility

    row = np.concatenate([np_lq, np_hq, np_A, np_C, np_F, res_A], axis=1)
    cv2.imwrite(os.path.join(args.out, f"compare_{idx:03d}.png"), row)

    # 单独保存 residual
    cv2.imwrite(os.path.join(args.out, f"residual_A_{idx:03d}.png"), res_A)

print(f"\nSaved comparisons to {args.out}/compare_*.png")
print(f"Saved residuals to {args.out}/residual_A_*.png")

# ===================================================================
# 最终结论
# ===================================================================
print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")

t_preserves = avg('t_lq_hq_delta') > 0.3
s_preserves = avg('s_lq_hq_delta') > 0.3
hf_is_lq = avg('cos_hf_out_lq') > avg('cos_hf_out_hq') + 0.05
latent_near_identity = avg('norm_final_mmse') < 0.05 and avg('cos_final_hq') > 0.99

print(f"  1. TAESD (teacher) preserves rain in latent:  {'YES' if t_preserves else 'NO'} (delta={avg('t_lq_hq_delta'):.4f})")
print(f"  2. Student encoder preserves rain in latent:   {'YES' if s_preserves else 'NO'} (delta={avg('s_lq_hq_delta'):.4f})")
print(f"  3. Output HF closer to LQ than HQ:             {'YES (re-injection!)' if hf_is_lq else 'NO'} (cos_lq={avg('cos_hf_out_lq'):.3f}, cos_hq={avg('cos_hf_out_hq'):.3f})")
print(f"  4. Latent trunk near identity (FMIR useless):   {'YES' if latent_near_identity else 'NO'}")

print(f"\n  Root Cause Classification:")
if not t_preserves and hf_is_lq:
    print(f"  → C: BOTH latent rain loss + HF re-injection (latent bottleneck + SFT condition carries rain to output)")
elif not t_preserves:
    print(f"  → A: Latent rain loss dominates (TAESD 8× bottleneck)")
elif hf_is_lq:
    print(f"  → B: HF re-injection dominates (SFT/wavelet condition carries rain back)")
else:
    print(f"  → D: Neither — check decoder or data issues")

print(f"\n{'='*70}")
print(f"DONE")
print(f"{'='*70}")
