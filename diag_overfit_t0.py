"""
ELIR 去雨 FMIR t=0 校准过拟合诊断
===================================
核查结论：
1. CFM velocity 参数化：z_endpoint0 = z_mmse + v0 公式正确
2. lambda_flow_img 是已有功能，不是新实现
3. Xt_low at t=0 是 CLEAN 态，与推理的 NOISY 起点有轻微 mismatch（不致命）
4. cond 在 low_t 分支未 detach → 冻结 wavelet_stem/condition_stem 来隔离
5. Decoder forward 梯度可以穿透 → 无需 no_grad

用法:
  python diag_overfit_t0.py -y configs/derain/eval.yaml \
    --ckpt /home/a27/moxt/elir_sftunet_ED/runs/elir_e2e_large_derain/epoch=64-step=211640.ckpt
"""
import os, sys, math, csv, argparse, itertools
import torch, torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-y","--yaml",type=str,default="configs/derain/eval.yaml")
parser.add_argument("--ckpt",type=str,default=None)
parser.add_argument("--out",type=str,default="./runs/diag_overfit_t0")
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--steps",type=int,default=500)
parser.add_argument("--lr",type=float,default=2e-5)
parser.add_argument("--n_images",type=int,default=4)
args = parser.parse_args()
os.makedirs(args.out,exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ===================================================================
# 加载
# ===================================================================
from hyperpyyaml import load_hyperpyyaml
with open(args.yaml) as f: conf = load_hyperpyyaml(f)

model_cfg = conf.get("model_cfg",{})
arch_cfg = dict(model_cfg.get("arch_cfg",{}))
fm_cfg = dict(conf.get("fm_cfg",{}))
dataset_cfg = conf.get("dataset_cfg",{})
eval_cfg = conf.get("eval_cfg",{})

ckpt_path = args.ckpt or eval_cfg.get("ckpt_path") or arch_cfg.get("path")
if not ckpt_path:
    raise RuntimeError("Need --ckpt")
ckpt_path = os.path.expanduser(ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"ckpt: step={ckpt.get('global_step')}, epoch={ckpt.get('epoch')}")

from ELIR.models.load_model import get_model
from ELIR.models.elir import pos_emb

arch_cfg["path"] = None
model = get_model(deepcopy(arch_cfg))

# 加载 main state_dict
sd = ckpt.get("state_dict",{})
if sd:
    cleaned = OrderedDict((k[len("model."):] if k.startswith("model.") else k, v) for k,v in sd.items())
    model.load_state_dict(cleaned, strict=False)
for key,attr in [("state_dict_fmir","fmir"),("state_dict_mmse","mmse"),
                  ("state_dict_enc","enc"),("state_dict_dec","dec"),
                  ("state_dict_wavelet","wavelet_stem")]:
    if key in ckpt and ckpt[key]:
        getattr(model,attr).load_state_dict(ckpt[key], strict=False)

model.to(device)

# ===================================================================
# 冻结策略
# ===================================================================
# 全部冻结
for p in model.parameters():
    p.requires_grad = False

# 只解冻 FMIR（含 condition_stem）
for n, p in model.fmir.named_parameters():
    p.requires_grad = True

# 可选：冻结 condition_stem（隔离条件生成）
FREEZE_COND_STEM = True
if FREEZE_COND_STEM:
    cond_prefixes = ("condition_stem.","cond_to_256.","cond_down_128.","cond_down_64.","cond_down_32.")
    for n, p in model.fmir.named_parameters():
        if any(n.startswith(pr) for pr in cond_prefixes):
            p.requires_grad = False

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_trainable:,}")
print(f"enc trainable: {any(p.requires_grad for p in model.enc.parameters())}")
print(f"mmse trainable: {any(p.requires_grad for p in model.mmse.parameters())}")
print(f"dec trainable: {any(p.requires_grad for p in model.dec.parameters())}")
print(f"wavelet trainable: {any(p.requires_grad for p in model.wavelet_stem.parameters())}")
print(f"fmir trainable: {any(p.requires_grad for p in model.fmir.parameters())}")
print(f"cond_stem frozen: {FREEZE_COND_STEM}")

model.eval()
model.enc.eval()
model.wavelet_stem.eval()

# ===================================================================
# 数据
# ===================================================================
from ELIR.datasets.dataset import get_loader
vcfg = dict(dataset_cfg.get("val_dataset",{}))
vcfg.update({"batch_size":1,"num_workers":0})
loader = get_loader(vcfg)

batches = []
for idx,(x_lq,x_hq) in enumerate(loader):
    if idx >= args.n_images: break
    batches.append((x_lq.to(device), x_hq.to(device)))
print(f"overfit on {len(batches)} pairs")

# ===================================================================
# FM cfg overrides
# ===================================================================
fm_cfg["lambda_low_t_endpoint"] = 0.05
fm_cfg["low_t_endpoint_t"] = 0.0
fm_cfg["lambda_flow_img"] = 0.10
fm_cfg["lambda_ssim_calib"] = 0.02
fm_cfg["lambda_mmse_guard"] = 0.0  # 关闭 mmse guard
fm_cfg["lambda_fm"] = 1.0
fm_cfg["lambda_mmse_char"] = 0.1
fm_cfg["lambda_charb"] = 1.0
fm_cfg["lambda_ssim"] = 0.5
fm_cfg["lambda_color"] = 0.05
fm_cfg["lambda_blur"] = 0.0
fm_cfg["lambda_perc"] = 0.0
fm_cfg["lambda_gan"] = 0.0
fm_cfg["lambda_pix_max"] = 0.5
fm_cfg["lambda_pix_warmup_steps"] = 0  # no warmup in overfit
fm_cfg["t_sampling"] = "uniform"
fm_cfg["use_residual_amplification"] = False

fm_cfg.setdefault("k_steps",5)
fm_cfg.setdefault("t_emb_dim",160)
fm_cfg.setdefault("sigma_min",0.00001)
fm_cfg.setdefault("sigma_s",0.1)
fm_cfg.setdefault("alpha",0.001)
fm_cfg.setdefault("beta",0.001)
fm_cfg.setdefault("dt",0.05)

print(f"fm_cfg keys: {list(fm_cfg.keys())}")
print(f"lambda_low_t_endpoint={fm_cfg['lambda_low_t_endpoint']}")
print(f"lambda_flow_img={fm_cfg['lambda_flow_img']}")

# ===================================================================
# 训练
# ===================================================================
from ELIR.training.losses import e2e_gan_loss, charbonnier_loss
from ELIR.metrics import calculate_psnr

opt_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(opt_params, lr=args.lr)

# Teacher wrapper
from diffusers import AutoencoderTiny
taesd3 = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_enc = taesd3.encoder.to(device).eval()
for p in raw_enc.parameters(): p.requires_grad = False
class _TW:
    def __init__(self,e): self.encoder = e
    def eval(self): pass
    def to(self,d): self.encoder.to(d); return self
tw = _TW(raw_enc)

log = []
for step in range(args.steps):
    model.train()
    total_g_loss = 0.0
    total_d_loss = 0.0
    metrics_sum = {}

    for x_lq, x_hq in batches:
        oh,ow = x_hq.shape[2], x_hq.shape[3]
        ph,pw = (64-oh%64)%64, (64-ow%64)%64
        x_lq_p = F.pad(x_lq,(0,pw,0,ph),mode="reflect") if ph or pw else x_lq
        x_hq_p = F.pad(x_hq,(0,pw,0,ph),mode="reflect") if ph or pw else x_hq

        fm_cfg_runtime = dict(fm_cfg)
        fm_cfg_runtime["global_step"] = step

        g_loss, d_loss, metrics = e2e_gan_loss(
            model, x_hq_p, x_lq_p, fm_cfg_runtime,
            discriminator=None, perceptual_fn=None,
            dino_encoder=None, dino_spatial_proj=None,
            sk_fusion=None, step=step, tmodel=tw)

        optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()

        total_g_loss += g_loss.item()
        if isinstance(d_loss, torch.Tensor) and d_loss.item() != 0:
            total_d_loss += d_loss.item()
        for k,v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics_sum[k] = metrics_sum.get(k,0.0) + v.item()

    n = len(batches)

    if step % 50 == 0 or step == args.steps - 1:
        # 评估
        model.eval()
        with torch.no_grad():
            eval_psnr_mmse = []
            eval_psnr_t0 = []
            eval_psnr_final = []
            dist_mmse_hq = []
            dist_t0_hq = []
            dist_final_hq = []
            cos_mmse_hq = []
            cos_t0_hq = []
            cos_final_hq = []

            for x_lq, x_hq in batches:
                oh,ow = x_hq.shape[2], x_hq.shape[3]
                ph,pw = (64-oh%64)%64, (64-ow%64)%64
                x_lq_p = F.pad(x_lq,(0,pw,0,ph),mode="reflect") if ph or pw else x_lq
                x_hq_p = F.pad(x_hq,(0,pw,0,ph),mode="reflect") if ph or pw else x_hq

                z_lq = model._encode_input(x_lq_p)
                z_hq = model._encode_input(x_hq_p)
                z_mmse = model.mmse(z_lq)
                noise = model._inference_noise(z_mmse, device)
                z0 = z_mmse + noise

                # FMIR cond
                wc = None
                if model.wavelet_stem is not None:
                    wc = model.wavelet_stem(x_lq_p, t_emb=pos_emb(torch.zeros(x_lq_p.shape[0]), model.t_emb_dim).to(device))
                fcond = model._build_fmir_condition(x_lq_p, wavelet_cond=wc)
                fcond = model._augment_cond_with_latent(fcond, z_lq, z_mmse)

                # t=0 velocity
                t0 = torch.zeros(x_lq_p.shape[0],device=device,dtype=z_mmse.dtype)
                v0 = model.fmir(z_mmse, pos_emb(t0,model.t_emb_dim).to(device), cond=fcond, t=t0[:,None,None,None])
                z_t0_endpoint = z_mmse + v0

                # K-step ODE
                K = int(model.K); dt = 1.0/K
                z_final = z0.clone()
                for k in range(K):
                    tv = torch.full((z_final.shape[0],), k*dt, device=device, dtype=z_final.dtype)
                    vk = model.fmir(z_final, pos_emb(tv,model.t_emb_dim).to(device), cond=fcond, t=tv[:,None,None,None])
                    z_final = z_final + dt * vk

                dec_cond = model._build_fmir_condition(x_lq_p, wavelet_cond=None)

                y_mmse = model._decode_latent(z_mmse, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
                y_t0 = model._decode_latent(z_t0_endpoint, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
                y_final = model._decode_latent(z_final, cond=dec_cond, x_lq=x_lq_p, wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)

                eval_psnr_mmse.append(float(calculate_psnr(y_mmse, x_hq, test_y_channel=True)))
                eval_psnr_t0.append(float(calculate_psnr(y_t0, x_hq, test_y_channel=True)))
                eval_psnr_final.append(float(calculate_psnr(y_final, x_hq, test_y_channel=True)))

                ne = max(z_hq[0].numel(),1)**0.5
                dist_mmse_hq.append((z_mmse-z_hq).flatten(1).norm(dim=1).mean().item()/ne)
                dist_t0_hq.append((z_t0_endpoint-z_hq).flatten(1).norm(dim=1).mean().item()/ne)
                dist_final_hq.append((z_final-z_hq).flatten(1).norm(dim=1).mean().item()/ne)
                cos_mmse_hq.append(F.cosine_similarity(z_mmse.flatten(1), z_hq.flatten(1), dim=1).mean().item())
                cos_t0_hq.append(F.cosine_similarity(z_t0_endpoint.flatten(1), z_hq.flatten(1), dim=1).mean().item())
                cos_final_hq.append(F.cosine_similarity(z_final.flatten(1), z_hq.flatten(1), dim=1).mean().item())

        avg_psnr_m = np.mean(eval_psnr_mmse); avg_psnr_t0 = np.mean(eval_psnr_t0); avg_psnr_f = np.mean(eval_psnr_final)
        avg_dist_m = np.mean(dist_mmse_hq); avg_dist_t0 = np.mean(dist_t0_hq); avg_dist_f = np.mean(dist_final_hq)
        avg_cos_m = np.mean(cos_mmse_hq); avg_cos_t0 = np.mean(cos_t0_hq); avg_cos_f = np.mean(cos_final_hq)

        # Grad norms
        fm_grads = {}
        for n,p in model.fmir.named_parameters():
            if p.grad is not None:
                fm_grads[n] = p.grad.norm().item()

        log.append({
            "step": step,
            "loss_g": total_g_loss/n,
            "psnr_mmse": avg_psnr_m, "psnr_t0": avg_psnr_t0, "psnr_final": avg_psnr_f,
            "dist_mmse_hq": avg_dist_m, "dist_t0_hq": avg_dist_t0, "dist_final_hq": avg_dist_f,
            "cos_mmse_hq": avg_cos_m, "cos_t0_hq": avg_cos_t0, "cos_final_hq": avg_cos_f,
            "fm_grad_max": max(fm_grads.values()) if fm_grads else 0,
            "fm_grad_mean": np.mean(list(fm_grads.values())) if fm_grads else 0,
            **{f"raw_{k}": metrics_sum.get(k,0)/n for k in [
                "loss_fm","loss_low_t_endpoint","loss_flow_img","loss_mmse_char","loss_charb","loss_ssim"]},
        })

        print(f"  step {step:4d}: loss={total_g_loss/n:.4f}  "
              f"PSNR mmse={avg_psnr_m:.2f} t0={avg_psnr_t0:.2f} final={avg_psnr_f:.2f}  "
              f"dist mmse={avg_dist_m:.4f} t0={avg_dist_t0:.4f} final={avg_dist_f:.4f}  "
              f"cos mmse={avg_cos_m:.4f} t0={avg_cos_t0:.4f} final={avg_cos_f:.4f}  "
              f"fm_grad_max={log[-1]['fm_grad_max']:.4f}")

# ===================================================================
# 最终汇总
# ===================================================================
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")

init = log[0]; final = log[-1]

print(f"\n  PSNR:")
print(f"    MMSE:        {init['psnr_mmse']:.2f} → {final['psnr_mmse']:.2f} (Δ{final['psnr_mmse']-init['psnr_mmse']:+.2f})")
print(f"    t0 endpoint: {init['psnr_t0']:.2f} → {final['psnr_t0']:.2f} (Δ{final['psnr_t0']-init['psnr_t0']:+.2f})")
print(f"    final (K=5): {init['psnr_final']:.2f} → {final['psnr_final']:.2f} (Δ{final['psnr_final']-init['psnr_final']:+.2f})")

print(f"\n  Latent dist to HQ:")
print(f"    MMSE:        {init['dist_mmse_hq']:.4f} → {final['dist_mmse_hq']:.4f}")
print(f"    t0 endpoint: {init['dist_t0_hq']:.4f} → {final['dist_t0_hq']:.4f}")
print(f"    final:       {init['dist_final_hq']:.4f} → {final['dist_final_hq']:.4f}")

print(f"\n  Latent cosine to HQ:")
print(f"    MMSE:        {init['cos_mmse_hq']:.4f} → {final['cos_mmse_hq']:.4f}")
print(f"    t0 endpoint: {init['cos_t0_hq']:.4f} → {final['cos_t0_hq']:.4f}")
print(f"    final:       {init['cos_final_hq']:.4f} → {final['cos_final_hq']:.4f}")

print(f"\n  Raw losses:")
for k in ["loss_fm","loss_low_t_endpoint","loss_flow_img","loss_charb","loss_ssim"]:
    rk = f"raw_{k}"
    if rk in init:
        print(f"    {k}: {init[rk]:.6f} → {final[rk]:.6f}")

print(f"\n  FMIR grad max: {init['fm_grad_max']:.4f} → {final['fm_grad_max']:.4f}")

# 判决
psnr_t0_gain = final['psnr_t0'] - init['psnr_t0']
psnr_final_gain = final['psnr_final'] - init['psnr_final']
dist_t0_change = final['dist_t0_hq'] - init['dist_t0_hq']

if psnr_t0_gain > 0.5 and dist_t0_change < -0.01:
    print(f"\n  → 情况 A: t=0 校准有效 (PSNR_t0 +{psnr_t0_gain:.2f}dB, dist↓)")
elif psnr_t0_gain > 0.5 and psnr_final_gain < -0.3:
    print(f"\n  → 情况 B: 起点改善但 ODE 破坏 (PSNR_t0 +{psnr_t0_gain:.2f}dB, PSNR_final {psnr_final_gain:+.2f}dB)")
elif dist_t0_change < -0.01 and psnr_t0_gain < 0:
    print(f"\n  → 情况 C: latent 靠近但 pixel 变差 — lambda_low_t_endpoint 需继续降低")
elif abs(psnr_t0_gain) < 0.3:
    print(f"\n  → 情况 D: 无明显改善 — 检查梯度链")
else:
    print(f"\n  → 混合情况，见上表")

# 保存 log
with open(os.path.join(args.out,"log.csv"),"w",newline="") as f:
    if log:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)

print(f"\nLog saved to {args.out}/log.csv")
