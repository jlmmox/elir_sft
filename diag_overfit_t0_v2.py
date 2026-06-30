"""
ELIR 去雨 FMIR t=0 校准诊断 v2
================================
确认发现：
- elir.py:1070-1072 推理 ODE 从 z0 = z_mmse + noise 出发 (NOISY)
- low_t 分支从 X_mmse.detach() 出发 (CLEAN)
- noise 是 sigma_s=0.1 的固定张量，RMS≈0.081
- 二者互补：low_t 给 clean 锚点，ODE 从 noisy 出发后头几步消化噪声

同时测试 clean/noisy 两种推理起点。

用法:
  python diag_overfit_t0_v2.py -y configs/derain/eval.yaml \
    --ckpt /home/a27/moxt/elir_sftunet_ED/runs/elir_e2e_large_derain/epoch=64-step=211640.ckpt
"""
import os, sys, csv, argparse, itertools
import torch, torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-y","--yaml",type=str,default="configs/derain/eval.yaml")
parser.add_argument("--ckpt",type=str,default=None)
parser.add_argument("--out",type=str,default="./runs/diag_overfit_t0_v2")
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

ckpt_path = args.ckpt or arch_cfg.get("path")
if not ckpt_path: raise RuntimeError("Need --ckpt")
ckpt_path = os.path.expanduser(ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"ckpt: step={ckpt.get('global_step')}, epoch={ckpt.get('epoch')}")

from ELIR.models.load_model import get_model
from ELIR.models.elir import pos_emb
from ELIR.metrics import calculate_psnr, calculate_ssim

arch_cfg["path"] = None
model = get_model(deepcopy(arch_cfg))

sd = ckpt.get("state_dict",{})
if sd:
    cleaned = OrderedDict((k[len("model."):] if k.startswith("model.") else k, v) for k,v in sd.items())
    model.load_state_dict(cleaned, strict=False)
for key,attr in [("state_dict_fmir","fmir"),("state_dict_mmse","mmse"),
                  ("state_dict_enc","enc"),("state_dict_dec","dec"),
                  ("state_dict_wavelet","wavelet_stem")]:
    if key in ckpt and ckpt[key]:
        getattr(model,attr).load_state_dict(ckpt[key], strict=False)
        frac = 1.0-len(getattr(model,attr).load_state_dict(ckpt[key], strict=False)[0])/max(len(getattr(model,attr).state_dict()),1)
        print(f"[{key}] loaded: {frac:.1%}")

model.to(device)

# ===================================================================
# 关键参数
# ===================================================================
sigma_min = float(fm_cfg.get("sigma_min",0.00001))
sigma_s   = float(fm_cfg.get("sigma_s",0.1))
K         = int(model.K)
print(f"sigma_min={sigma_min}, sigma_s={sigma_s}, K={K}, dt={1.0/K:.4f}")
print(f"noise_mode={getattr(model,'noise_mode','legacy_fixed')}")

# ===================================================================
# 实际噪声验证
# ===================================================================
from ELIR.datasets.dataset import get_loader
vcfg = dict(dataset_cfg.get("val_dataset",{}))
vcfg.update({"batch_size":1,"num_workers":0})
loader = get_loader(vcfg)
x_test = next(iter(loader))[0].to(device)

model.eval()
with torch.no_grad():
    z_test = model._encode_input(x_test)
    noise_test = model._inference_noise(z_test, device)
    noise_rms = noise_test.flatten(1).norm(dim=1).mean().item()/max(noise_test[0].numel(),1)**0.5
    z_mmse_test = model.mmse(z_test)
    z_rms = z_mmse_test.flatten(1).norm(dim=1).mean().item()/max(z_mmse_test[0].numel(),1)**0.5

print(f"\n=== NOISE VERIFICATION ===")
print(f"inference noise RMS:      {noise_rms:.6f}  (= sigma_s * randn per-element std)")
print(f"z_mmse signal RMS:        {z_rms:.6f}")
print(f"noise/signal ratio:       {noise_rms/(z_rms+1e-8):.4f}")
print(f"sigma_min:                {sigma_min:.8f}")
print(f"sigma_min==0 effectively: {sigma_min < 1e-5}")
print(f"→ inference z0 = z_mmse + noise  (NOISY, code at elir.py:1072/1086)")
print(f"→ low_t branch   = X_mmse.detach() (CLEAN, code at losses.py:903)")

# ===================================================================
# 冻结策略
# ===================================================================
for p in model.parameters():
    p.requires_grad = False

for n, p in model.fmir.named_parameters():
    p.requires_grad = True

FREEZE_COND_STEM = True
if FREEZE_COND_STEM:
    cond_prefixes = ("condition_stem.","cond_to_256.","cond_down_128.","cond_down_64.","cond_down_32.")
    for n, p in model.fmir.named_parameters():
        if any(n.startswith(pr) for pr in cond_prefixes):
            p.requires_grad = False

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\ntrainable params: {n_trainable:,}")
print(f"enc_trainable={any(p.requires_grad for p in model.enc.parameters())}")
print(f"mmse_trainable={any(p.requires_grad for p in model.mmse.parameters())}")
print(f"dec_trainable={any(p.requires_grad for p in model.dec.parameters())}")
print(f"wavelet_trainable={any(p.requires_grad for p in model.wavelet_stem.parameters())}")
print(f"fmir_trainable={any(p.requires_grad for p in model.fmir.parameters())}")
print(f"cond_stem_frozen={FREEZE_COND_STEM}")

# ===================================================================
# 数据 + 固定噪声
# ===================================================================
batches = []
for idx,(x_lq,x_hq) in enumerate(loader):
    if idx >= args.n_images: break
    batches.append((x_lq.to(device), x_hq.to(device)))
print(f"overfit on {len(batches)} pairs")

# 预缓存所有 latent、cond
cached = []
with torch.no_grad():
    for x_lq, x_hq in batches:
        oh,ow=x_hq.shape[2],x_hq.shape[3]
        ph,pw=(64-oh%64)%64,(64-ow%64)%64
        x_lq_p=F.pad(x_lq,(0,pw,0,ph),mode="reflect") if ph or pw else x_lq
        x_hq_p=F.pad(x_hq,(0,pw,0,ph),mode="reflect") if ph or pw else x_hq

        z_lq = model._encode_input(x_lq_p)
        z_hq = model._encode_input(x_hq_p)
        z_mmse = model.mmse(z_lq)
        noise = model._inference_noise(z_mmse, device)  # fixed noise tensor

        wc = None
        if model.wavelet_stem is not None:
            wc = model.wavelet_stem(x_lq_p, t_emb=pos_emb(torch.zeros(x_lq_p.shape[0]),model.t_emb_dim).to(device))

        cond = model._build_fmir_condition(x_lq_p, wavelet_cond=wc)
        cond = model._augment_cond_with_latent(cond, z_lq, z_mmse)
        dec_cond = model._build_fmir_condition(x_lq_p, wavelet_cond=None)

        cached.append({
            "x_lq_p": x_lq_p, "x_hq_p": x_hq_p, "x_hq": x_hq,
            "oh": oh, "ow": ow,
            "z_lq": z_lq, "z_hq": z_hq, "z_mmse": z_mmse,
            "noise": noise,
            "cond": cond.detach(),
            "dec_cond": dec_cond.detach(),
        })

# 固定 CFM 训练噪声（模式A：每次都同一份）
fixed_cfm_noise = {}
for i, c in enumerate(cached):
    fixed_cfm_noise[i] = sigma_s * torch.randn_like(c["z_mmse"])

# ===================================================================
# FM cfg
# ===================================================================
fm_cfg["lambda_low_t_endpoint"] = 0.05
fm_cfg["low_t_endpoint_t"] = 0.0
fm_cfg["lambda_flow_img"] = 0.10
fm_cfg["lambda_ssim_calib"] = 0.02
fm_cfg["lambda_mmse_guard"] = 0.0
fm_cfg["lambda_fm"] = 1.0
fm_cfg["lambda_mmse_char"] = 0.1
fm_cfg["lambda_charb"] = 1.0
fm_cfg["lambda_ssim"] = 0.5
fm_cfg["lambda_color"] = 0.05
fm_cfg["lambda_blur"] = 0.0
fm_cfg["lambda_perc"] = 0.0
fm_cfg["lambda_gan"] = 0.0
fm_cfg["lambda_pix_max"] = 0.5
fm_cfg["lambda_pix_warmup_steps"] = 0
fm_cfg["t_sampling"] = "uniform"
fm_cfg["use_residual_amplification"] = False
fm_cfg.setdefault("k_steps",K)
fm_cfg.setdefault("t_emb_dim",160)
fm_cfg.setdefault("sigma_min",sigma_min)
fm_cfg.setdefault("sigma_s",sigma_s)
fm_cfg.setdefault("alpha",0.001)
fm_cfg.setdefault("beta",0.001)

# ===================================================================
# Teacher
# ===================================================================
from diffusers import AutoencoderTiny
taesd3 = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_enc = taesd3.encoder.to(device).eval()
for p in raw_enc.parameters(): p.requires_grad=False
class _TW:
    def __init__(self,e): self.encoder=e
    def eval(self): pass
    def to(self,d): self.encoder.to(d); return self
tw = _TW(raw_enc)

# ===================================================================
# 评估函数
# ===================================================================
from ELIR.training.losses import e2e_gan_loss, charbonnier_loss

@torch.no_grad()
def evaluate(model, start_from_clean=True):
    """评估所有路径的 PSNR 和 latent 指标。
    start_from_clean=True: z0 = z_mmse (low_t 对齐)
    start_from_clean=False: z0 = z_mmse + noise (当前 ODE 推理)
    """
    results = {}
    for i, c in enumerate(cached):
        z_mmse = c["z_mmse"]
        z_hq = c["z_hq"]
        cond = c["cond"]
        dec_cond = c["dec_cond"]
        x_hq_p = c["x_hq_p"]
        x_hq = c["x_hq"]
        oh, ow = c["oh"], c["ow"]

        ne = max(z_hq[0].numel(),1)**0.5

        # ---- 1. MMSE baseline ----
        y_mmse = model._decode_latent(z_mmse, cond=dec_cond, x_lq=c["x_lq_p"], wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
        psnr_mmse = float(calculate_psnr(y_mmse, x_hq, test_y_channel=True))

        # ---- 2. Clean t=0 endpoint ----
        t0 = torch.zeros(z_mmse.shape[0],device=device,dtype=z_mmse.dtype)
        v0 = model.fmir(z_mmse, pos_emb(t0,model.t_emb_dim).to(device), cond=cond, t=t0[:,None,None,None])
        z_t0 = z_mmse + v0
        y_t0 = model._decode_latent(z_t0, cond=dec_cond, x_lq=c["x_lq_p"], wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
        psnr_t0 = float(calculate_psnr(y_t0, x_hq, test_y_channel=True))

        # ---- 3. K-step ODE (clean or noisy start) ----
        if start_from_clean:
            z_start = z_mmse
        else:
            z_start = z_mmse + c["noise"]
        dt_val = 1.0/K
        z_curr = z_start.clone()
        traj = []
        for k in range(K):
            tv = torch.full((z_curr.shape[0],), k*dt_val, device=device, dtype=z_curr.dtype)
            vk = model.fmir(z_curr, pos_emb(tv,model.t_emb_dim).to(device), cond=cond, t=tv[:,None,None,None])
            z_curr = z_curr + dt_val * vk
            # Record at each step
            y_step = model._decode_latent(z_curr, cond=dec_cond, x_lq=c["x_lq_p"], wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
            dist_hq = (z_curr-z_hq).flatten(1).norm(dim=1).mean().item()/ne
            cos_hq = F.cosine_similarity(z_curr.flatten(1), z_hq.flatten(1), dim=1).mean().item()
            v_norm = vk.flatten(1).norm(dim=1).mean().item()/ne
            step_delta = (dt_val*vk).flatten(1).norm(dim=1).mean().item()/ne
            traj.append({
                "step": k, "t": k*dt_val,
                "psnr": float(calculate_psnr(y_step, x_hq, test_y_channel=True)),
                "dist_hq": dist_hq, "cos_hq": cos_hq,
                "v_norm": v_norm, "step_delta": step_delta,
            })
        y_final = model._decode_latent(z_curr, cond=dec_cond, x_lq=c["x_lq_p"], wavelet_cond=None)[:,:,:oh,:ow].clamp(0,1)
        psnr_final = float(calculate_psnr(y_final, x_hq, test_y_channel=True))
        dist_final = (z_curr-z_hq).flatten(1).norm(dim=1).mean().item()/ne
        cos_final = F.cosine_similarity(z_curr.flatten(1), z_hq.flatten(1), dim=1).mean().item()

        dist_mmse_hq = (z_mmse-z_hq).flatten(1).norm(dim=1).mean().item()/ne
        cos_mmse_hq = F.cosine_similarity(z_mmse.flatten(1), z_hq.flatten(1), dim=1).mean().item()
        dist_t0_hq = (z_t0-z_hq).flatten(1).norm(dim=1).mean().item()/ne
        cos_t0_hq = F.cosine_similarity(z_t0.flatten(1), z_hq.flatten(1), dim=1).mean().item()
        v0_norm = v0.flatten(1).norm(dim=1).mean().item()/ne

        results[i] = {
            "psnr_mmse": psnr_mmse, "psnr_t0": psnr_t0, "psnr_final": psnr_final,
            "dist_mmse_hq": dist_mmse_hq, "dist_t0_hq": dist_t0_hq, "dist_final_hq": dist_final,
            "cos_mmse_hq": cos_mmse_hq, "cos_t0_hq": cos_t0_hq, "cos_final_hq": cos_final,
            "v0_norm": v0_norm, "traj": traj,
        }
    return results

# ===================================================================
# 初始评估
# ===================================================================
print(f"\n=== INITIAL EVALUATION ===")
model.eval()
init_clean = evaluate(model, start_from_clean=True)
init_noisy = evaluate(model, start_from_clean=False)

def _avg(d, key):
    return np.mean([d[i][key] for i in d])

print(f"  Clean start:  PSNR mmse={_avg(init_clean,'psnr_mmse'):.2f} t0={_avg(init_clean,'psnr_t0'):.2f} final={_avg(init_clean,'psnr_final'):.2f}")
print(f"  Noisy start:  PSNR mmse={_avg(init_noisy,'psnr_mmse'):.2f} t0={_avg(init_noisy,'psnr_t0'):.2f} final={_avg(init_noisy,'psnr_final'):.2f}")
print(f"  Clean: dist mmse={_avg(init_clean,'dist_mmse_hq'):.4f} t0={_avg(init_clean,'dist_t0_hq'):.4f} final={_avg(init_clean,'dist_final_hq'):.4f}")
print(f"  Noisy: dist mmse={_avg(init_noisy,'dist_mmse_hq'):.4f} t0={_avg(init_noisy,'dist_t0_hq'):.4f} final={_avg(init_noisy,'dist_final_hq'):.4f}")

# ===================================================================
# 训练
# ===================================================================
opt_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(opt_params, lr=args.lr)
print(f"\n=== TRAINING {args.steps} steps ===")

log = []
for step in range(args.steps):
    model.train()
    total = {"g_loss":0,"d_loss":0}
    raw_losses = {}

    for i, c in enumerate(cached):
        x_lq_p, x_hq_p = c["x_lq_p"], c["x_hq_p"]

        fm_cfg_runtime = dict(fm_cfg)
        fm_cfg_runtime["global_step"] = step

        # 用固定 CFM 噪声 + 真实 sigma_s（与推理对齐）
        X_mmse_noisy_fixed = c["z_mmse"].detach() + fixed_cfm_noise[i]

        # 临时替换 X_mmse_noisy（需要 monkey-patch e2e_gan_loss 的噪声来源）
        # 实际上 e2e_gan_loss 内部生成噪声，这里我们用固定噪声重写
        # 简化：直接用 e2e_gan_loss 但传入固定噪声的 fm_cfg
        # ---- 手动构建 loss（绕过 e2e_gan_loss 的噪声生成）----
        t_dim = fm_cfg.get("t_emb_dim",160)
        sigma_min_val = float(fm_cfg.get("sigma_min",0.00001))
        sigma_s_val = float(fm_cfg.get("sigma_s",0.1))
        alpha_cfm = float(fm_cfg.get("alpha",0.001))
        beta_cfm = float(fm_cfg.get("beta",0.001))
        bs = x_lq_p.shape[0]

        X_lq = model._encode_input(x_lq_p)
        X_hq = model._encode_input(x_hq_p).detach()

        cond_i = model._build_fmir_condition(x_lq_p, wavelet_cond=None)
        # detach cond to prevent grad to non-FMIR modules
        cond_i = {k: v.detach() if torch.is_tensor(v) else v for k,v in cond_i.items()} if cond_i else None
        cond_i = model._augment_cond_with_latent(cond_i, X_lq, c["z_mmse"].detach())

        dec_cond_i = {k: v.detach() if torch.is_tensor(v) else v for k,v in c["dec_cond"].items()}

        X_mmse = model.mmse(X_lq)
        X_mmse_detached = X_mmse.detach()

        # ---- 1. CFM loss (保留原始随机采样 + 固定噪声) ----
        eps = fixed_cfm_noise[i] / sigma_s_val  # recover unit noise
        X_mmse_noisy = X_mmse_detached + sigma_s_val * eps

        t_cfm = (1 - fm_cfg.get("dt",0.05)) * torch.rand([bs,1,1,1], device=device, dtype=X_hq.dtype)
        segments = torch.linspace(0,1,K+1,device=device,dtype=X_hq.dtype)
        seg_indices = torch.searchsorted(segments, t_cfm, side="left").clamp(min=1)
        seg_ends = segments[seg_indices]

        Xt = (1-(1-sigma_min_val)*t_cfm)*X_mmse_noisy + t_cfm*X_hq
        v0_cfm = model.fmir(Xt, pos_emb(t_cfm.squeeze(-1).squeeze(-1).squeeze(-1), t_dim).to(device), cond=cond_i, t=t_cfm)

        r = t_cfm + fm_cfg.get("dt",0.05)
        Xr = (1-(1-sigma_min_val)*r)*X_mmse_noisy + r*X_hq
        with torch.no_grad():
            v0_cfm_ = model.fmir(Xr, pos_emb(r.squeeze(-1).squeeze(-1).squeeze(-1), t_dim).to(device), cond=cond_i, t=r)

        X_ends = (1-(1-sigma_min_val)*seg_ends)*X_mmse_noisy + seg_ends*X_hq
        f0 = Xt + (seg_ends - t_cfm) * v0_cfm
        r_less = r < seg_ends
        f0_ = r_less*(Xr + (seg_ends - r)*v0_cfm_) + (~r_less)*X_ends

        loss_fm_val = (1-beta_cfm)*(charbonnier_loss(f0, f0_) + alpha_cfm*charbonnier_loss(v0_cfm, v0_cfm_))
        f1 = f0.detach()
        v1 = model.fmir(f1, pos_emb(seg_ends.squeeze(-1).squeeze(-1).squeeze(-1), t_dim).to(device), cond=cond_i, t=seg_ends)
        X1 = f1 + (1-seg_ends)*v1
        loss_fm_val += beta_cfm*charbonnier_loss(X_hq, X1)

        # ---- Pixel loss through X1 decode ----
        pred_pix = model._decode_latent(X1, cond=dec_cond_i, x_lq=x_lq_p, wavelet_cond=None)
        pred_pix = pred_pix[:,:,:c["oh"],:c["ow"]].clamp(0,1)
        x_hq_crop = x_hq_p[:,:,:c["oh"],:c["ow"]].clamp(0,1)
        loss_pix = charbonnier_loss(pred_pix, x_hq_crop) + 0.5*(1.0-F.cosine_similarity(
            pred_pix.view(bs,-1), x_hq_crop.view(bs,-1), dim=1).mean())

        # ---- 2. Clean t=0 latent endpoint ----
        t_low = torch.zeros(bs, device=device, dtype=X_hq.dtype)
        Xt_low = X_mmse_detached  # clean, as design
        v_low = model.fmir(Xt_low, pos_emb(t_low, t_dim).to(device), cond=cond_i, t=t_low[:,None,None,None])

        # CFM 路径终点 target: X_hq + sigma_min * X0_low
        z_terminal_pred = Xt_low + v_low
        z_terminal_target = X_hq.detach() + sigma_min_val * Xt_low

        loss_low_t = charbonnier_loss(z_terminal_pred, z_terminal_target)

        # ---- 3. Clean t=0 pixel endpoint ----
        z_calib = X_mmse_detached + v_low
        pred_calib = model._decode_latent(z_calib, cond=dec_cond_i, x_lq=x_lq_p, wavelet_cond=None)
        pred_calib = pred_calib[:,:,:c["oh"],:c["ow"]].clamp(0,1)
        loss_flow_img_charb = charbonnier_loss(pred_calib, x_hq_crop)
        loss_flow_img_ssim = 0.02 * (1.0 - F.cosine_similarity(
            pred_calib.view(bs,-1), x_hq_crop.view(bs,-1), dim=1).mean())

        # ---- 4. MMSE charb loss ----
        loss_mmse_char_val = charbonnier_loss(X_mmse, X_hq.detach())

        # ---- Total ----
        g_loss = (
            1.0 * loss_fm_val
            + 0.1 * loss_mmse_char_val
            + 0.05 * loss_low_t
            + 0.10 * loss_flow_img_charb
            + loss_flow_img_ssim
            + 0.5 * loss_pix
        )

        optimizer.zero_grad()
        g_loss.backward()

        # 梯度检查（第一步）
        if step == 0 and i == 0:
            # Enc/MMSE/Dec grad check
            enc_grad = sum(1 for _,p in model.enc.named_parameters() if p.grad is not None)
            mmse_grad = sum(1 for _,p in model.mmse.named_parameters() if p.grad is not None)
            dec_grad = sum(1 for _,p in model.dec.named_parameters() if p.grad is not None)
            ws_grad = sum(1 for _,p in model.wavelet_stem.named_parameters() if p.grad is not None) if model.wavelet_stem else 0
            fmir_grad = sum(1 for _,p in model.fmir.named_parameters() if p.grad is not None and p.requires_grad)
            fmir_grad_norms = [p.grad.norm().item() for _,p in model.fmir.named_parameters() if p.grad is not None and p.requires_grad]

            print(f"\n  [step0] GRAD CHECK:")
            print(f"    enc grad!=None: {enc_grad} (expect 0)")
            print(f"    mmse grad!=None: {mmse_grad} (expect 0)")
            print(f"    dec grad!=None: {dec_grad} (expect 0)")
            print(f"    wavelet grad!=None: {ws_grad} (expect 0)")
            print(f"    fmir grad!=None: {fmir_grad} (expect >0)")
            if fmir_grad_norms:
                print(f"    fmir grad_norm max={max(fmir_grad_norms):.4f} mean={np.mean(fmir_grad_norms):.4f}")

            # Assertions
            assert enc_grad == 0, f"Encoder got gradient! ({enc_grad} params)"
            assert mmse_grad == 0, f"MMSE got gradient! ({mmse_grad} params)"
            assert dec_grad == 0, f"Decoder got gradient! ({dec_grad} params)"
            assert fmir_grad > 0, "FMIR got NO gradient!"
            print(f"    ✓ All assertions passed")

        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()

        total["g_loss"] += g_loss.item()
        raw_losses["loss_fm"] = raw_losses.get("loss_fm",0) + loss_fm_val.item()
        raw_losses["loss_low_t_endpoint"] = raw_losses.get("loss_low_t_endpoint",0) + loss_low_t.item()
        raw_losses["loss_flow_img_charb"] = raw_losses.get("loss_flow_img_charb",0) + loss_flow_img_charb.item()
        raw_losses["loss_flow_img_ssim"] = raw_losses.get("loss_flow_img_ssim",0) + loss_flow_img_ssim.item()
        raw_losses["loss_mmse_char"] = raw_losses.get("loss_mmse_char",0) + loss_mmse_char_val.item()
        raw_losses["loss_pix"] = raw_losses.get("loss_pix",0) + loss_pix.item()

    n = len(cached)

    if step % 50 == 0 or step == args.steps - 1:
        model.eval()
        ev_clean = evaluate(model, start_from_clean=True)
        ev_noisy = evaluate(model, start_from_clean=False)

        # ODE trajectory for first image
        traj0_clean = ev_clean[0]["traj"]
        traj0_noisy = ev_noisy[0]["traj"]

        entry = {
            "step": step,
            "loss_g": total["g_loss"]/n,
            "psnr_mmse": _avg(ev_clean,"psnr_mmse"),
            "psnr_t0_clean": _avg(ev_clean,"psnr_t0"),
            "psnr_final_clean": _avg(ev_clean,"psnr_final"),
            "psnr_final_noisy": _avg(ev_noisy,"psnr_final"),
            "dist_mmse": _avg(ev_clean,"dist_mmse_hq"),
            "dist_t0_clean": _avg(ev_clean,"dist_t0_hq"),
            "dist_final_clean": _avg(ev_clean,"dist_final_hq"),
            "dist_final_noisy": _avg(ev_noisy,"dist_final_hq"),
            "cos_mmse": _avg(ev_clean,"cos_mmse_hq"),
            "cos_t0_clean": _avg(ev_clean,"cos_t0_hq"),
            "cos_final_clean": _avg(ev_clean,"cos_final_hq"),
            "cos_final_noisy": _avg(ev_noisy,"cos_final_hq"),
            "v0_norm": _avg(ev_clean,"v0_norm"),
            **{f"raw_{k}": v/n for k,v in raw_losses.items()},
            # ODE trajectory PSNR for image 0, clean start
            **{f"traj_clean_psnr_k{k}": traj0_clean[k]["psnr"] for k in range(min(K,len(traj0_clean)))},
            **{f"traj_noisy_psnr_k{k}": traj0_noisy[k]["psnr"] for k in range(min(K,len(traj0_noisy)))},
        }
        log.append(entry)

        print(f"  step{step:4d}: g={total['g_loss']/n:.4f} "
              f"PSNR(mmse={_avg(ev_clean,'psnr_mmse'):.2f} t0={_avg(ev_clean,'psnr_t0'):.2f} "
              f"final_c={_avg(ev_clean,'psnr_final'):.2f} final_n={_avg(ev_noisy,'psnr_final'):.2f}) "
              f"dist(mmse={_avg(ev_clean,'dist_mmse_hq'):.3f} t0={_avg(ev_clean,'dist_t0_hq'):.3f} "
              f"final_c={_avg(ev_clean,'dist_final_hq'):.3f} final_n={_avg(ev_noisy,'dist_final_hq'):.3f}) "
              f"flow_gain_c={_avg(ev_clean,'psnr_final')-_avg(ev_clean,'psnr_mmse'):+.2f} "
              f"flow_gain_n={_avg(ev_noisy,'psnr_final')-_avg(ev_noisy,'psnr_mmse'):+.2f}")

# ===================================================================
# FINAL
# ===================================================================
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")

init = log[0]; fin = log[-1]
print(f"\n  PSNR:")
print(f"    MMSE:               {init['psnr_mmse']:.2f} → {fin['psnr_mmse']:.2f}")
print(f"    t0 endpoint (clean): {init['psnr_t0_clean']:.2f} → {fin['psnr_t0_clean']:.2f}  Δ{fin['psnr_t0_clean']-init['psnr_t0_clean']:+.2f}")
print(f"    final clean:         {init['psnr_final_clean']:.2f} → {fin['psnr_final_clean']:.2f}  Δ{fin['psnr_final_clean']-init['psnr_final_clean']:+.2f}")
print(f"    final noisy:         {init['psnr_final_noisy']:.2f} → {fin['psnr_final_noisy']:.2f}  Δ{fin['psnr_final_noisy']-init['psnr_final_noisy']:+.2f}")

print(f"\n  Latent dist to HQ:")
print(f"    MMSE:    {init['dist_mmse']:.4f} → {fin['dist_mmse']:.4f}")
print(f"    t0:      {init['dist_t0_clean']:.4f} → {fin['dist_t0_clean']:.4f}")
print(f"    final_c: {init['dist_final_clean']:.4f} → {fin['dist_final_clean']:.4f}")
print(f"    final_n: {init['dist_final_noisy']:.4f} → {fin['dist_final_noisy']:.4f}")

print(f"\n  flow_gain_psnr:")
print(f"    clean: {init['psnr_final_clean']-init['psnr_mmse']:+.2f} → {fin['psnr_final_clean']-fin['psnr_mmse']:+.2f}")
print(f"    noisy: {init['psnr_final_noisy']-init['psnr_mmse']:+.2f} → {fin['psnr_final_noisy']-fin['psnr_mmse']:+.2f}")

# ODE trajectory
print(f"\n  ODE trajectory (clean, img0):")
for k in range(K):
    pk_c = f"traj_clean_psnr_k{k}"
    pk_n = f"traj_noisy_psnr_k{k}"
    if pk_c in init:
        print(f"    k={k}: clean={init[pk_c]:.2f}→{fin[pk_c]:.2f}  noisy={init[pk_n]:.2f}→{fin[pk_n]:.2f}")

print(f"\n  Raw losses:")
for k in ["loss_fm","loss_low_t_endpoint","loss_flow_img_charb","loss_flow_img_ssim","loss_pix"]:
    rk = f"raw_{k}"
    if rk in init and rk in fin:
        print(f"    {k}: {init[rk]:.6f} → {fin[rk]:.6f}")

# Verdict
psnr_t0_gain = fin['psnr_t0_clean'] - init['psnr_t0_clean']
psnr_final_gain = fin['psnr_final_clean'] - init['psnr_final_clean']
dist_t0_change = fin['dist_t0_clean'] - init['dist_t0_clean']

print(f"\n  → VERDICT:")
if psnr_t0_gain > 0.5 and psnr_final_gain > 0.3:
    print(f"    情况A: t=0校准有效，ODE未破坏")
elif psnr_t0_gain > 0.5 and psnr_final_gain < -0.3:
    print(f"    情况B: 起点改善但ODE破坏")
elif dist_t0_change < -0.01 and psnr_t0_gain < 0:
    print(f"    情况C: latent靠近但pixel变差")
elif abs(psnr_t0_gain) < 0.3:
    print(f"    情况D: 无明显改善")
else:
    print(f"    混合情况")

# Save
with open(os.path.join(args.out,"log.csv"),"w",newline="") as f:
    if log:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
print(f"\nLog: {args.out}/log.csv")
