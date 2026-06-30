"""
ELIR 去雨诊断脚本 v2 — 补充 D5/D6、latent距离、raw TAESD配对检查、修复overfit

用法：
  python diag_derain_v2.py -y configs/derain/eval.yaml
"""

import os, sys, math, csv, argparse, itertools, copy
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from collections import OrderedDict

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, default="configs/derain/eval.yaml")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--out", type=str, default="./runs/diag_derain_v2")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--max_samples", type=int, default=10)
parser.add_argument("--overfit_steps", type=int, default=300)
parser.add_argument("--overfit_lr", type=float, default=1e-4)
parser.add_argument("--skip_overfit", action="store_true")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[diag_v2] device={device}")

# ---------------------------------------------------------------------------
from hyperpyyaml import load_hyperpyyaml
with open(args.yaml) as f:
    conf = load_hyperpyyaml(f)

eval_cfg = conf.get("eval_cfg", {})
model_cfg = conf.get("model_cfg", {})
arch_cfg = model_cfg.get("arch_cfg", {})
fm_cfg = conf.get("fm_cfg", {})
dataset_cfg = conf.get("dataset_cfg", {})
teacher_cfg = model_cfg.get("teacher_cfg", {})

ckpt_path = args.ckpt or eval_cfg.get("ckpt_path") or arch_cfg.get("path")
print(f"ckpt_path: {ckpt_path}")

# ---------------------------------------------------------------------------
from ELIR.models.load_model import get_model
from ELIR.irsetup import IRSetup
from ELIR.models.elir import pos_emb
from ELIR.training.tparmas import get_opt_sched
from ELIR.metrics import calculate_psnr

# ---- 构建 model ----
arch_cfg_no_path = dict(arch_cfg)
arch_cfg_no_path["path"] = None
model = get_model(arch_cfg_no_path)
print(f"model built: {type(model).__name__}")

# ---- 构建 teacher（严格对准 TAESD3 原始 encoder） ----
from diffusers import AutoencoderTiny
taesd3_full = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_taesd_enc = taesd3_full.encoder.to(device)
raw_taesd_dec = taesd3_full.decoder.to(device)
raw_taesd_enc.eval()
raw_taesd_dec.eval()
for p in itertools.chain(raw_taesd_enc.parameters(), raw_taesd_dec.parameters()):
    p.requires_grad = False

# 用 raw_taesd_enc 作为 teacher（确保与 raw_taesd_dec 来自同一权重文件）
class _TeacherWrapper:
    def __init__(self, enc):
        self.encoder = enc
    def eval(self):
        self.encoder.eval()
    def to(self, dev):
        self.encoder.to(dev)
        return self
    def parameters(self):
        return self.encoder.parameters()
tmodel = _TeacherWrapper(raw_taesd_enc)

# ---- 加载 checkpoint ----
if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    print(f"ckpt keys: {list(ckpt.keys())}")
    print(f"ckpt global_step: {ckpt.get('global_step', 'N/A')}, epoch: {ckpt.get('epoch', 'N/A')}")

    state_dict = ckpt.get("state_dict", {})
    if state_dict:
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            cleaned[k[len("model."):] if k.startswith("model.") else k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[main] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  missing: {missing[:5]}")
    else:
        print("[main] NO state_dict in ckpt!")

    for key, attr in [
        ("state_dict_fmir", "fmir"), ("state_dict_mmse", "mmse"),
        ("state_dict_enc", "enc"), ("state_dict_dec", "dec"),
        ("state_dict_wavelet", "wavelet_stem"),
    ]:
        if key in ckpt and ckpt[key]:
            m, u = getattr(model, attr).load_state_dict(ckpt[key], strict=False)
            frac = 1.0 - len(m)/max(len(getattr(model, attr).state_dict()), 1)
            print(f"[{key}] → {attr}: missing={len(m)}, unexpected={len(u)}, loaded={frac:.1%}")
            if m and frac < 0.9:
                print(f"  missing: {m[:5]}")
        elif key in ckpt:
            print(f"[{key}] → {attr}: EMPTY or absent (random init!)")
        else:
            print(f"[{key}] → {attr}: NOT IN CKPT (random init!)")

model.to(device)
model.eval()
tmodel.to(device)

# ---- IRSetup ----
setup = IRSetup(model=model, fm_cfg=fm_cfg, eval_cfg=dict(eval_cfg, metrics=["psnr"]),
                run_dir=args.out, save_images=False, optimizer=None, scheduler=None, tmodel=tmodel)
setup.to(device)
setup.eval()
active = setup.ema.model if setup.ema else model
print(f"EMA: {setup.ema is not None}, active.training={active.training}")

# ---- 数据集 ----
from ELIR.datasets.dataset import get_loader
val_cfg_diag = dict(dataset_cfg.get("val_dataset", {}))
val_cfg_diag.update({"batch_size": 1, "num_workers": 0})
valloader = get_loader(val_cfg_diag)
print(f"dataset: {len(valloader.dataset)} samples")

# ===================================================================
# C2. RAW TAESD 编码器/解码器配对检查
# ===================================================================
print(f"\n{'='*70}")
print(f"[C2] RAW TAESD ENCODER/DECODER PAIRING CHECK")
print(f"{'='*70}")
print(f"  raw_taesd_enc type: {type(raw_taesd_enc).__name__}")
print(f"  raw_taesd_dec type: {type(raw_taesd_dec).__name__}")
print(f"  model.enc type:     {type(model.enc).__name__}")
print(f"  model.dec type:     {type(model.dec).__name__}")

# 验证 raw TAESD 的 enc→dec 往返是否正常
pairing_psnr = []
for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_hq = batch[1].to(device)
    with torch.no_grad():
        z_raw = raw_taesd_enc(x_hq)
        y_raw = raw_taesd_dec(z_raw).clamp(0, 1)
    psnr = float(calculate_psnr(y_raw, x_hq, test_y_channel=True))
    pairing_psnr.append(psnr)
    print(f"  [{idx}] raw TAESD enc→dec roundtrip PSNR: {psnr:.4f} dB  (latent mean={z_raw.mean().item():.4f} std={z_raw.std().item():.4f})")

avg_pair = sum(pairing_psnr)/len(pairing_psnr)
print(f"\n  → Average raw TAESD roundtrip PSNR: {avg_pair:.2f} dB")
if avg_pair < 30:
    print(f"  ⚠️  WARNING: raw TAESD roundtrip < 30dB — encoder/decoder pairing or tanh scaling issue!")
else:
    print(f"  ✓ raw TAESD encoder/decoder pairing is correct")

# ===================================================================
# C3. Model encoder vs teacher encoder 参数对比
# ===================================================================
print(f"\n{'='*70}")
print(f"[C3] ENCODER WEIGHT COMPARISON (model.enc vs raw TAESD enc)")
print(f"{'='*70}")

model_enc_sd = model.enc.state_dict()
raw_enc_sd = raw_taesd_enc.state_dict()
common_keys = set(model_enc_sd.keys()) & set(raw_enc_sd.keys())
print(f"  common keys: {len(common_keys)}/{len(model_enc_sd)} (model) vs {len(raw_enc_sd)} (raw)")

max_diff_key = None
max_diff_val = 0.0
for k in sorted(common_keys):
    diff = (model_enc_sd[k] - raw_enc_sd[k]).abs().max().item()
    if diff > max_diff_val:
        max_diff_val = diff
        max_diff_key = k

print(f"  max |weight_diff|: {max_diff_val:.6f} at {max_diff_key}")

# ===================================================================
# C4. Latent 对齐 + 距离矩阵
# ===================================================================
print(f"\n{'='*70}")
print(f"[C4] LATENT ALIGNMENT & DISTANCE MATRIX")
print(f"{'='*70}")

for idx, batch in enumerate(valloader):
    if idx >= min(5, args.max_samples):
        break
    x_lq, x_hq = batch[0].to(device), batch[1].to(device)
    oh, ow = x_hq.shape[2], x_hq.shape[3]
    ph, pw = (64 - oh % 64) % 64, (64 - ow % 64) % 64
    x_lq_p = F.pad(x_lq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_lq
    x_hq_p = F.pad(x_hq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_hq

    with torch.no_grad():
        # 三路 latent
        z_teacher = raw_taesd_enc(x_hq_p).float()       # raw TAESD enc(HQ)
        z_student_hq = active._encode_input(x_hq_p)      # model.enc(HQ) — drifted
        z_student_lq = active._encode_input(x_lq_p)      # model.enc(LQ)
        z_mmse = active.mmse(z_student_lq)               # MMSE(LQ_latent)

        eps = 1e-6
        def _charb(a, b):
            return torch.sqrt((a-b).pow(2)+eps).mean().item()
        def _cos(a, b):
            return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()
        def _rms(t):
            return t.flatten(1).norm(dim=1).mean().item() / max(t[0].numel(),1)**0.5

        print(f"  [{idx}]")
        # A. teacher↔student_hq (encoder drift)
        print(f"    teacher↔student_hq: charb={_charb(z_teacher, z_student_hq):.4f}  "
              f"cos={_cos(z_teacher, z_student_hq):.4f}  "
              f"rms_t={_rms(z_teacher):.4f}  rms_s={_rms(z_student_hq):.4f}")

        # B. z_mmse vs teacher/student_hq (关键!)
        print(f"    z_mmse↔teacher:     charb={_charb(z_mmse, z_teacher):.4f}  cos={_cos(z_mmse, z_teacher):.4f}")
        print(f"    z_mmse↔student_hq:  charb={_charb(z_mmse, z_student_hq):.4f}  cos={_cos(z_mmse, z_student_hq):.4f}")

        # C. z_student_lq vs z_student_hq
        print(f"    student_lq↔student_hq: charb={_charb(z_student_lq, z_student_hq):.4f}  "
              f"cos={_cos(z_student_lq, z_student_hq):.4f}")

# ===================================================================
# D. 完整消融（含 D5, D6）
# ===================================================================
print(f"\n{'='*70}")
print(f"[D] FULL DECODE ABLATION (D1~D6)")
print(f"{'='*70}")

diag_results = []
for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_lq, x_hq = batch[0].to(device), batch[1].to(device)
    oh, ow = x_hq.shape[2], x_hq.shape[3]
    ph, pw = (64 - oh % 64) % 64, (64 - ow % 64) % 64
    x_lq_p = F.pad(x_lq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_lq
    x_hq_p = F.pad(x_hq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_hq

    with torch.no_grad():
        z_student_hq = active._encode_input(x_hq_p)
        z_student_lq = active._encode_input(x_lq_p)
        z_mmse = active.mmse(z_student_lq)
        cond_spatial = active._build_fmir_condition(x_lq_p, wavelet_cond=None)

        # 构建 zero cond
        zero_cond = {}
        if cond_spatial is not None:
            for k, v in cond_spatial.items():
                zero_cond[k] = torch.zeros_like(v)

        def _sft_psnr(z, cond, x_lq=None):
            y = active._decode_latent(z, cond=cond, x_lq=x_lq, wavelet_cond=None)
            y = y[:, :, :oh, :ow].clamp(0, 1)
            return float(calculate_psnr(y, x_hq, test_y_channel=True))

        # D1: raw TAESD decoder (no SFT) — 直接用 raw_taesd_dec
        y_d1 = raw_taesd_dec(z_student_hq).clamp(0, 1)[:, :, :oh, :ow]
        d1 = float(calculate_psnr(y_d1, x_hq, test_y_channel=True))

        d2 = _sft_psnr(z_student_hq, zero_cond)                 # SFT(z_hq, zero cond)
        d3 = _sft_psnr(z_student_hq, cond_spatial, x_lq_p)      # SFT(z_hq, LQ cond)
        d4 = _sft_psnr(z_mmse, cond_spatial, x_lq_p)            # SFT(z_mmse, LQ cond)
        d5 = _sft_psnr(z_student_hq, zero_cond, x_lq_p)         # SFT(z_hq, zero cond, with x_lq)
        d6 = _sft_psnr(z_student_hq, cond_spatial, x_lq_p)      # same as D3

        diag_results.append((idx, d1, d2, d3, d4, d5, d6))
        print(f"  [{idx}] D1={d1:.2f} D2={d2:.2f} D3={d3:.2f} D4={d4:.2f} D5={d5:.2f} D6={d6:.2f}")

# 汇总
avg = [sum(r[i] for r in diag_results)/len(diag_results) for i in range(1,7)]
print(f"\n  → Average:")
print(f"    D1 (raw TAESD dec, z_student_hq):          {avg[0]:.2f} dB")
print(f"    D2 (SFT z_student_hq, ZERO cond):          {avg[1]:.2f} dB")
print(f"    D3 (SFT z_student_hq, LQ cond):            {avg[2]:.2f} dB  ← diag_sft_hq_psnr")
print(f"    D4 (SFT z_mmse, LQ cond):                  {avg[3]:.2f} dB  ← diag_sft_mmse_psnr")
print(f"    D5 (SFT z_student_hq, ZERO cond, x_lq):    {avg[4]:.2f} dB")
print(f"    D6 (SFT z_student_hq, LQ cond, x_lq):      {avg[5]:.2f} dB")

print(f"\n  → Gaps:")
print(f"    D1→D2: {avg[1]-avg[0]:+.2f} dB  (raw TAESD vs SFT zero-cond — SFT包装器对drifted latent的改善)")
print(f"    D2→D3: {avg[2]-avg[1]:+.2f} dB  (zero cond → LQ cond — LQ条件对HQ latent的贡献)")
print(f"    D3→D4: {avg[3]-avg[2]:+.2f} dB  (z_hq → z_mmse, same LQ cond)")
print(f"    D2→D5: {avg[4]-avg[1]:+.2f} dB  (zero w/o x_lq → zero w/ x_lq — wavelet/target_size影响)")

# ===================================================================
# E. 修正 Flow 诊断（复用 z0 起点）
# ===================================================================
print(f"\n{'='*70}")
print(f"[E] CORRECTED FLOW DIAGNOSTICS")
print(f"{'='*70}")

flow_metrics = {k: [] for k in [
    "charb_mmse_to_hq", "charb_fmir_to_hq", "cos_pred", "cos_old",
    "pred_norm", "target_norm", "noise_norm", "norm_ratio",
    "gain_abs", "gain_rel",
]}
for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_lq, x_hq = batch[0].to(device), batch[1].to(device)
    oh, ow = x_hq.shape[2], x_hq.shape[3]
    ph, pw = (64 - oh % 64) % 64, (64 - ow % 64) % 64
    x_lq_p = F.pad(x_lq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_lq
    x_hq_p = F.pad(x_hq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_hq

    with torch.no_grad():
        z_hq = active._encode_input(x_hq_p)
        z_lq = active._encode_input(x_lq_p)
        z_mmse = active.mmse(z_lq)
        noise = active._inference_noise(z_mmse, device)
        z0 = z_mmse + noise

        # FMIR ODE
        wc = None
        if active.wavelet_stem is not None:
            wc = active.wavelet_stem(x_lq_p, t_emb=pos_emb(torch.zeros(x_lq_p.shape[0]), active.t_emb_dim).to(device))
        fcond = active._build_fmir_condition(x_lq_p, wavelet_cond=wc)
        fcond = active._augment_cond_with_latent(fcond, z_lq, z_mmse)
        K = int(active.K)
        dt_val = 1.0 / K
        z_final = z0.clone()
        for k in range(K):
            t_val = k * dt_val
            t_vec = torch.full((z_final.shape[0],), t_val, device=device, dtype=z_final.dtype)
            v_step = active.fmir(z_final, pos_emb(t_vec, active.t_emb_dim).to(device),
                                cond=fcond, t=t_vec[:,None,None,None])
            z_final = z_final + dt_val * v_step

        delta_pred = z_final - z0
        delta_target = z_hq - z0
        delta_old = z_final - z_mmse
        delta_target_old = z_hq - z_mmse
        ne = max(z0[0].numel(), 1)**0.5
        eps_c = 1e-6

        flow_metrics["charb_mmse_to_hq"].append(torch.sqrt((z_mmse-z_hq).pow(2)+eps_c).mean().item())
        flow_metrics["charb_fmir_to_hq"].append(torch.sqrt((z_final-z_hq).pow(2)+eps_c).mean().item())
        flow_metrics["cos_pred"].append(F.cosine_similarity(delta_pred.flatten(1), delta_target.flatten(1), dim=1).mean().item())
        flow_metrics["cos_old"].append(F.cosine_similarity(delta_old.flatten(1), delta_target_old.flatten(1), dim=1).mean().item())
        flow_metrics["pred_norm"].append(delta_pred.flatten(1).norm(dim=1).mean().item()/ne)
        flow_metrics["target_norm"].append(delta_target.flatten(1).norm(dim=1).mean().item()/ne)
        flow_metrics["noise_norm"].append(noise.flatten(1).norm(dim=1).mean().item()/ne)
        flow_metrics["norm_ratio"].append(flow_metrics["pred_norm"][-1]/(flow_metrics["target_norm"][-1]+1e-8))
        flow_metrics["gain_abs"].append(flow_metrics["charb_mmse_to_hq"][-1]-flow_metrics["charb_fmir_to_hq"][-1])
        flow_metrics["gain_rel"].append(flow_metrics["gain_abs"][-1]/(flow_metrics["charb_mmse_to_hq"][-1]+1e-8))

        print(f"  [{idx}] noise={flow_metrics['noise_norm'][-1]:.4f}  pred={flow_metrics['pred_norm'][-1]:.4f}  "
              f"target={flow_metrics['target_norm'][-1]:.4f}  ratio={flow_metrics['norm_ratio'][-1]:.3f}  "
              f"cos_pred={flow_metrics['cos_pred'][-1]:.3f}  gain_rel={flow_metrics['gain_rel'][-1]:.1%}")

print(f"\n  → Averages:")
for k in flow_metrics:
    print(f"    {k}: {sum(flow_metrics[k])/len(flow_metrics[k]):.6f}")

# ===================================================================
# F. 修复后的 Overfit Test
# ===================================================================
if not args.skip_overfit:
    print(f"\n{'='*70}")
    print(f"[F] OVERFIT TEST ({args.overfit_steps} steps, lr={args.overfit_lr})")
    print(f"{'='*70}")

    # 收集4对数据
    overfit_batches = []
    for idx, batch in enumerate(valloader):
        if idx >= 4:
            break
        overfit_batches.append(batch)

    # 辅助函数：prepare batch（64-align）
    def _prep(batch):
        x_lq, x_hq = batch[0].to(device), batch[1].to(device)
        oh, ow = x_hq.shape[2], x_hq.shape[3]
        ph, pw = (64-oh%64)%64, (64-ow%64)%64
        x_lq_p = F.pad(x_lq, (0,pw,0,ph), mode="reflect") if ph or pw else x_lq
        x_hq_p = F.pad(x_hq, (0,pw,0,ph), mode="reflect") if ph or pw else x_hq
        return x_lq, x_hq, x_lq_p, x_hq_p

    def _eval_overfit(ov_model):
        """评估 overfit 模型在前4张图上的 PSNR + diag_sft_hq PSNR"""
        ov_model.eval()
        psnr_vals, hq_psnr_vals = [], []
        with torch.no_grad():
            for batch in overfit_batches:
                x_lq, x_hq, x_lq_p, x_hq_p = _prep(batch)
                oh, ow = x_hq.shape[2], x_hq.shape[3]

                z_lq = ov_model._encode_input(x_lq_p)
                z_hq = ov_model._encode_input(x_hq_p)
                z_mmse = ov_model.mmse(z_lq)
                noise = ov_model._inference_noise(z_mmse, device)
                z0 = z_mmse + noise
                fm_cond = ov_model._build_fmir_condition(x_lq_p, wavelet_cond=None)
                fm_cond = ov_model._augment_cond_with_latent(fm_cond, z_lq, z_mmse)
                z_final = z0.clone()
                K_ov = int(ov_model.K)
                dt_ov = 1.0/K_ov
                for k in range(K_ov):
                    tv = torch.full((z_final.shape[0],), k*dt_ov, device=device, dtype=z_final.dtype)
                    vk = ov_model.fmir(z_final, pos_emb(tv, ov_model.t_emb_dim).to(device),
                                      cond=fm_cond, t=tv[:,None,None,None])
                    z_final = z_final + dt_ov * vk

                dec_cond = ov_model._build_fmir_condition(x_lq_p, wavelet_cond=None)
                y_pred = ov_model._decode_latent(z_final, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)
                y_pred = y_pred[:,:,:oh,:ow].clamp(0,1)
                psnr_vals.append(float(calculate_psnr(y_pred, x_hq, test_y_channel=True)))

                y_hq_dec = ov_model._decode_latent(z_hq, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)
                y_hq_dec = y_hq_dec[:,:,:oh,:ow].clamp(0,1)
                hq_psnr_vals.append(float(calculate_psnr(y_hq_dec, x_hq, test_y_channel=True)))

        return sum(psnr_vals)/len(psnr_vals), sum(hq_psnr_vals)/len(hq_psnr_vals)

    # ---- Overfit 实验 1: FMIR head only (final_proj + final_block + last 2 up_blocks) ----
    print(f"\n--- Exp 1: FMIR Head-Only ---")
    model_ov1 = copy.deepcopy(model)
    model_ov1.train()
    # 冻结 FMIR 大部分，仅解冻 head
    head_prefixes = ("latent_cond_proj.", "first_proj.", "up_blocks.3.", "up_blocks.2.",
                     "sft_up.3.", "sft_up.2.", "final_block.", "final_proj.")
    for n, p in model_ov1.named_parameters():
        p.requires_grad = n.startswith(head_prefixes) and any(
            n.startswith(pr) for pr in ("fmir.",))
    # 确保 MMSE 可训练（pixel loss需要）
    for n, p in model_ov1.named_parameters():
        if n.startswith("mmse."):
            p.requires_grad = True
    opt1 = torch.optim.AdamW([p for p in model_ov1.parameters() if p.requires_grad], lr=args.overfit_lr)
    n_params1 = sum(p.numel() for p in model_ov1.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params1:,}")
    log1 = []
    for step in range(args.overfit_steps):
        total_loss = 0.0
        for batch in overfit_batches:
            _, _, x_lq_p, x_hq_p = _prep(batch)
            z_lq = model_ov1._encode_input(x_lq_p)
            z_hq = model_ov1._encode_input(x_hq_p).detach()
            z_mmse = model_ov1.mmse(z_lq)
            fm_cond = model_ov1._build_fmir_condition(x_lq_p, wavelet_cond=None)
            fm_cond = model_ov1._augment_cond_with_latent(fm_cond, z_lq, z_mmse)
            t0 = torch.zeros(x_lq_p.shape[0], device=device)
            v_pred = model_ov1.fmir(z_mmse, pos_emb(t0, model_ov1.t_emb_dim).to(device),
                                   cond=fm_cond, t=t0[:,None,None,None])
            loss = F.mse_loss(z_mmse, z_hq) + 0.1 * F.mse_loss(v_pred, z_hq - z_mmse)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
            total_loss += loss.item()
        if step % 50 == 0 or step == args.overfit_steps - 1:
            p, h = _eval_overfit(model_ov1)
            log1.append((step, total_loss/len(overfit_batches), p, h))
            print(f"  step {step:4d}: loss={total_loss/len(overfit_batches):.6f}  PSNR={p:.4f}  diag_hq={h:.4f}")

    # ---- Overfit 实验 2: FMIR SFT + head ----
    print(f"\n--- Exp 2: FMIR SFT + Head (add sft_up, sft, sft_down) ---")
    model_ov2 = copy.deepcopy(model)
    model_ov2.train()
    sft_prefixes = head_prefixes + ("sft_up.", "sft.", "sft_down.", "cross_band_attn.")
    for n, p in model_ov2.named_parameters():
        p.requires_grad = n.startswith(sft_prefixes) and any(
            n.startswith(pr) for pr in ("fmir.",))
    for n, p in model_ov2.named_parameters():
        if n.startswith("mmse."):
            p.requires_grad = True
    opt2 = torch.optim.AdamW([p for p in model_ov2.parameters() if p.requires_grad], lr=args.overfit_lr)
    n_params2 = sum(p.numel() for p in model_ov2.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params2:,}")
    log2 = []
    for step in range(args.overfit_steps):
        total_loss = 0.0
        for batch in overfit_batches:
            _, _, x_lq_p, x_hq_p = _prep(batch)
            z_lq = model_ov2._encode_input(x_lq_p)
            z_hq = model_ov2._encode_input(x_hq_p).detach()
            z_mmse = model_ov2.mmse(z_lq)
            fm_cond = model_ov2._build_fmir_condition(x_lq_p, wavelet_cond=None)
            fm_cond = model_ov2._augment_cond_with_latent(fm_cond, z_lq, z_mmse)
            t0 = torch.zeros(x_lq_p.shape[0], device=device)
            v_pred = model_ov2.fmir(z_mmse, pos_emb(t0, model_ov2.t_emb_dim).to(device),
                                   cond=fm_cond, t=t0[:,None,None,None])
            loss = F.mse_loss(z_mmse, z_hq) + 0.1 * F.mse_loss(v_pred, z_hq - z_mmse)
            opt2.zero_grad()
            loss.backward()
            opt2.step()
            total_loss += loss.item()
        if step % 50 == 0 or step == args.overfit_steps - 1:
            p, h = _eval_overfit(model_ov2)
            log2.append((step, total_loss/len(overfit_batches), p, h))
            print(f"  step {step:4d}: loss={total_loss/len(overfit_batches):.6f}  PSNR={p:.4f}  diag_hq={h:.4f}")

    # ---- Overfit 实验 3: Full FMIR ----
    print(f"\n--- Exp 3: Full FMIR ---")
    model_ov3 = copy.deepcopy(model)
    model_ov3.train()
    for n, p in model_ov3.named_parameters():
        p.requires_grad = n.startswith("fmir.") or n.startswith("mmse.")
    opt3 = torch.optim.AdamW([p for p in model_ov3.parameters() if p.requires_grad], lr=args.overfit_lr)
    n_params3 = sum(p.numel() for p in model_ov3.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params3:,}")
    log3 = []
    for step in range(args.overfit_steps):
        total_loss = 0.0
        for batch in overfit_batches:
            _, _, x_lq_p, x_hq_p = _prep(batch)
            z_lq = model_ov3._encode_input(x_lq_p)
            z_hq = model_ov3._encode_input(x_hq_p).detach()
            z_mmse = model_ov3.mmse(z_lq)
            fm_cond = model_ov3._build_fmir_condition(x_lq_p, wavelet_cond=None)
            fm_cond = model_ov3._augment_cond_with_latent(fm_cond, z_lq, z_mmse)
            t0 = torch.zeros(x_lq_p.shape[0], device=device)
            v_pred = model_ov3.fmir(z_mmse, pos_emb(t0, model_ov3.t_emb_dim).to(device),
                                   cond=fm_cond, t=t0[:,None,None,None])
            loss = F.mse_loss(z_mmse, z_hq) + 0.1 * F.mse_loss(v_pred, z_hq - z_mmse)
            opt3.zero_grad()
            loss.backward()
            opt3.step()
            total_loss += loss.item()
        if step % 50 == 0 or step == args.overfit_steps - 1:
            p, h = _eval_overfit(model_ov3)
            log3.append((step, total_loss/len(overfit_batches), p, h))
            print(f"  step {step:4d}: loss={total_loss/len(overfit_batches):.6f}  PSNR={p:.4f}  diag_hq={h:.4f}")

    # ---- 汇总 ----
    print(f"\n  → Overfit Summary:")
    for label, log in [("Head-Only", log1), ("SFT+Head", log2), ("Full FMIR", log3)]:
        if log:
            p0, pf = log[0][2], log[-1][2]
            h0, hf = log[0][3], log[-1][3]
            print(f"    {label:12s}: PSNR {p0:.2f}→{pf:.2f} (Δ{pf-p0:+.2f})  "
                  f"diag_hq {h0:.2f}→{hf:.2f} (Δ{hf-h0:+.2f})")

    # 保存 log
    for fname, log in [("overfit_head.csv", log1), ("overfit_sft_head.csv", log2), ("overfit_full.csv", log3)]:
        with open(os.path.join(args.out, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step","loss","psnr","diag_sft_hq_psnr"])
            w.writerows(log)

print(f"\n{'='*70}")
print(f"DONE — outputs in {args.out}/")
print(f"{'='*70}")
