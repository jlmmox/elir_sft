"""
ELIR 去雨系统性诊断脚本
按 A→F 顺序排查 diag_sft_hq_psnr_full=22.9 / latent_delta_cosine=0.195 的根因。

用法：
  python diag_derain.py -y configs/derain/eval.yaml
  或在脚本内修改 CKPT_PATH / EVAL_YAML 后直接运行。
"""

import os, sys, math, csv, argparse, itertools
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, default="configs/derain/eval.yaml")
parser.add_argument("--ckpt", type=str, default=None,
                    help="覆盖 eval yaml 中的 checkpoint 路径")
parser.add_argument("--out", type=str, default="./runs/diag_derain")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--max_samples", type=int, default=10,
                    help="数据配对诊断用的最大样本数")
parser.add_argument("--overfit_steps", type=int, default=200,
                    help="过拟合测试的训练步数")
parser.add_argument("--overfit_lr", type=float, default=1e-4)
parser.add_argument("--skip_overfit", action="store_true")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[diag] device={device}")
print(f"[diag] output dir={args.out}")

# ---------------------------------------------------------------------------
# 加载配置
# ---------------------------------------------------------------------------
from hyperpyyaml import load_hyperpyyaml

with open(args.yaml) as f:
    conf = load_hyperpyyaml(f)

# 解析 checkpoint 路径
eval_cfg = conf.get("eval_cfg", {})
model_cfg = conf.get("model_cfg", {})
arch_cfg = model_cfg.get("arch_cfg", {})
fm_cfg = conf.get("fm_cfg", {})
dataset_cfg = conf.get("dataset_cfg", {})
teacher_cfg = model_cfg.get("teacher_cfg", {})

ckpt_path = args.ckpt or eval_cfg.get("ckpt_path") or arch_cfg.get("path")
print(f"\n{'='*70}")
print(f"[A] CHECKPOINT & MODEL LOADING")
print(f"{'='*70}")
print(f"  config yaml: {args.yaml}")
print(f"  ckpt_path (resolved): {ckpt_path}")

# ---------------------------------------------------------------------------
# A. 判断 global_step=0 来源 + 加载 checkpoint
# ---------------------------------------------------------------------------
from ELIR.models.load_model import get_model
from ELIR.irsetup import IRSetup
from ELIR.training.tparmas import get_opt_sched
from ELIR.models.elir import pos_emb

# 先构建模型（不加载 checkpoint 权重 → get_model 会加载 arch_cfg.path 的权重）
# 我们手动控制：先用 arch_cfg 构建模型，再显式加载 checkpoint
arch_cfg_no_path = dict(arch_cfg)
arch_cfg_no_path["path"] = None  # 不让 get_model 自动加载
model = get_model(arch_cfg_no_path)
print(f"  model built: {type(model).__name__}")

# 构建 teacher（eval yaml 可能 teacher_cfg=null，手动构造）
tmodel = None
if teacher_cfg is None:
    # eval yaml 没有 teacher，手动创建 TAESD encoder 作为 teacher
    from diffusers import AutoencoderTiny
    taesd_teacher = AutoencoderTiny.from_pretrained("madebyollin/taesd3").to(device)
    tmodel = taesd_teacher.encoder  # 直接用 encoder，但 get_model 返回的可能需要 .encoder 属性
    # 包装成有 .encoder 属性的对象以兼容 _to_teacher_latent
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
    tmodel = _TeacherWrapper(taesd_teacher.encoder)
    tmodel.eval()
    for p in tmodel.parameters():
        p.requires_grad = False
    print(f"  teacher built (auto from TAESD3): encoder (frozen)")
elif isinstance(teacher_cfg, dict):
    teacher_cfg_no_path = dict(teacher_cfg)
    teacher_cfg_no_path["path"] = None
    tmodel = get_model(teacher_cfg_no_path)
    tmodel.eval()
    for p in tmodel.parameters():
        p.requires_grad = False
    print(f"  teacher built: {type(tmodel).__name__} (frozen)")
else:
    print(f"  teacher: skipping (unknown type {type(teacher_cfg)})")

# 解析 EMSS decay
ema_decay = eval_cfg.get("ema_decay", conf.get("train_cfg", {}).get("ema_decay", 0.999))

if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt_path_expanded = os.path.expanduser(ckpt_path)
    print(f"  loading checkpoint: {ckpt_path_expanded}")
    ckpt = torch.load(ckpt_path_expanded, map_location="cpu", weights_only=False)

    # 检查 checkpoint 结构
    print(f"  ckpt keys ({len(ckpt)}): {list(ckpt.keys())}")

    # 判断是否为 weights-only ckpt（无 trainer state）
    has_trainer_state = any(k.startswith("optimizer") or k.startswith("lr_scheduler")
                           for k in ckpt.keys())
    print(f"  has_trainer_state (optimizer/lr_scheduler): {has_trainer_state}")

    # global_step 记录
    if "global_step" in ckpt:
        print(f"  ckpt global_step: {ckpt['global_step']}")
    else:
        print(f"  ckpt global_step: NOT FOUND (weights-only checkpoint?)")

    if "epoch" in ckpt:
        print(f"  ckpt epoch: {ckpt['epoch']}")

    # 手动加载每个子模块
    state_dict = ckpt.get("state_dict", {})
    if state_dict:
        # PyTorch Lightning 格式：去掉 "model." 前缀
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model."):]] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"  [main state_dict] missing keys: {len(missing)}")
        print(f"  [main state_dict] unexpected keys: {len(unexpected)}")
        if missing:
            print(f"    missing sample (first 10):")
            for m in missing[:10]:
                print(f"      - {m}")
        if unexpected:
            print(f"    unexpected sample (first 10):")
            for u in unexpected[:10]:
                print(f"      - {u}")
    else:
        print(f"  [main state_dict] NOT FOUND in checkpoint!")

    # 加载子模块
    for key, attr in [
        ("state_dict_fmir", "fmir"),
        ("state_dict_mmse", "mmse"),
        ("state_dict_enc", "enc"),
        ("state_dict_dec", "dec"),
        ("state_dict_wavelet", "wavelet_stem"),
        ("state_dict_sft", "sft_refiner"),
        ("state_dict_condition", "condition_stem"),
    ]:
        if key in ckpt and ckpt[key]:
            module = getattr(model, attr, None)
            if module is not None:
                m, u = module.load_state_dict(ckpt[key], strict=False)
                loaded_frac = 1.0 - len(m) / max(len(module.state_dict()), 1)
                print(f"  [{key}] → model.{attr}: missing={len(m)}, unexpected={len(u)}, "
                      f"loaded={loaded_frac:.1%}")
                if len(m) > 0 and loaded_frac < 0.9:
                    print(f"    missing sample: {m[:5]}")
            else:
                print(f"  [{key}] → model.{attr}: MODULE IS None, SKIPPED")
        elif key in ckpt and (not ckpt[key] or len(ckpt[key]) == 0):
            print(f"  [{key}] → model.{attr}: EMPTY in ckpt!")
        else:
            print(f"  [{key}] → model.{attr}: NOT FOUND in ckpt (using random init)")
else:
    print(f"  WARNING: checkpoint NOT FOUND at {ckpt_path}")
    print(f"  Model will use random initialization!")

model.to(device)
model.eval()
if tmodel is not None:
    tmodel.to(device)

# 构建 IRSetup（用于推理）
# 手动设置 eval_cfg 以免触发不需要的日志
setup_eval_cfg = dict(eval_cfg)
setup_eval_cfg.setdefault("metrics", ["psnr"])

setup = IRSetup(
    model=model,
    fm_cfg=fm_cfg,
    eval_cfg=setup_eval_cfg,
    run_dir=args.out,
    save_images=False,
    ema_decay=ema_decay,
    optimizer=None,
    scheduler=None,
    tmodel=tmodel,
)
setup.to(device)
setup.eval()

# EMA
active = setup.ema.model if setup.ema else model
print(f"\n  EMA active: {setup.ema is not None}")
print(f"  active.training = {active.training}")

# ---------------------------------------------------------------------------
# 加载验证集
# ---------------------------------------------------------------------------
from ELIR.datasets.dataset import get_loader

val_cfg = dataset_cfg.get("val_dataset", {})
# 用较小的 batch 做诊断
val_cfg_diag = dict(val_cfg)
val_cfg_diag["batch_size"] = 1
val_cfg_diag["num_workers"] = 0
valloader = get_loader(val_cfg_diag)
print(f"\n  val dataset: {val_cfg.get('path')}")
print(f"  lq_subdir={val_cfg.get('lq_subdir')}, hq_subdir={val_cfg.get('hq_subdir')}")
print(f"  dataset length: {len(valloader.dataset)}")

# ---------------------------------------------------------------------------
# B. 数据配对诊断
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"[B] DATA PAIRING DIAGNOSIS")
print(f"{'='*70}")

from ELIR.metrics import calculate_psnr

input_lq_psnr_vals = []
data_samples = []

for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_lq, x_hq = batch[0], batch[1]

    # LQ→HQ raw PSNR
    psnr_raw = calculate_psnr(x_lq.clamp(0, 1), x_hq.clamp(0, 1), test_y_channel=True)
    if isinstance(psnr_raw, list):
        psnr_raw = sum(psnr_raw) / len(psnr_raw)
    input_lq_psnr_vals.append(float(psnr_raw))

    # 拼图保存
    cmp = torch.cat([x_lq.clamp(0, 1), x_hq.clamp(0, 1)], dim=3)  # LQ | HQ horizontal
    data_samples.append((idx, float(psnr_raw), cmp))

    print(f"  [{idx}] input_lq_psnr={psnr_raw:.4f}dB  "
          f"LQ shape={list(x_lq.shape)}  HQ shape={list(x_hq.shape)}  "
          f"LQ range=[{x_lq.min().item():.3f}, {x_lq.max().item():.3f}]  "
          f"HQ range=[{x_hq.min().item():.3f}, {x_hq.max().item():.3f}]")

# 保存拼图
for idx, psnr_val, cmp in data_samples:
    save_image(cmp, os.path.join(args.out, f"B_pair_{idx:03d}_psnr{psnr_val:.1f}.png"))

avg_input_psnr = sum(input_lq_psnr_vals) / len(input_lq_psnr_vals) if input_lq_psnr_vals else 0
print(f"\n  → Average input LQ→HQ PSNR: {avg_input_psnr:.4f} dB")
print(f"  → 已保存 {len(data_samples)} 组拼图到 {args.out}/B_pair_*.png")

# ---------------------------------------------------------------------------
# C. Encoder latent 对齐
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"[C] ENCODER LATENT ALIGNMENT (teacher vs student on same HQ)")
print(f"{'='*70}")

with torch.no_grad():
    for idx, batch in enumerate(valloader):
        if idx >= min(5, args.max_samples):
            break
        x_lq, x_hq = batch[0], batch[1]
        x_hq_dev = x_hq.to(device)

        # Teacher latent
        if tmodel is not None:
            if hasattr(tmodel, "encoder"):
                z_hq_teacher = tmodel.encoder(x_hq_dev).float()
            elif hasattr(tmodel, "encode"):
                z_hq_teacher = tmodel.encode(x_hq_dev).float()
            else:
                z_hq_teacher = tmodel(x_hq_dev).float()
        else:
            z_hq_teacher = None

        # Student latent
        if hasattr(active, "enc") and active.enc is not None:
            if hasattr(active.enc, "encoder"):
                z_hq_student = active.enc.encoder(x_hq_dev).float()
            elif hasattr(active.enc, "encode"):
                z_hq_student = active.enc.encode(x_hq_dev).float()
            else:
                z_hq_student = active.enc(x_hq_dev).float()
        else:
            z_hq_student = None

        if z_hq_teacher is not None and z_hq_student is not None:
            # Charbonnier
            eps = 1e-6
            charb = torch.sqrt((z_hq_teacher - z_hq_student).pow(2) + eps).mean().item()

            # Cosine similarity
            cos = F.cosine_similarity(
                z_hq_teacher.flatten(1), z_hq_student.flatten(1), dim=1
            ).mean().item()

            # Statistics
            t_mean, t_std = z_hq_teacher.mean().item(), z_hq_teacher.std().item()
            s_mean, s_std = z_hq_student.mean().item(), z_hq_student.std().item()

            # Element-wise magnitude ratio
            t_norm = z_hq_teacher.flatten(1).norm(dim=1).mean().item()
            s_norm = z_hq_student.flatten(1).norm(dim=1).mean().item()
            norm_ratio = s_norm / (t_norm + 1e-8)

            print(f"  [{idx}] teacher↔student: charb={charb:.6f}  cosine={cos:.4f}  "
                  f"norm_ratio(s/t)={norm_ratio:.4f}")
            print(f"        teacher mean={t_mean:.4f} std={t_std:.4f}  "
                  f"student mean={s_mean:.4f} std={s_std:.4f}")
        elif z_hq_student is not None:
            s_mean, s_std = z_hq_student.mean().item(), z_hq_student.std().item()
            print(f"  [{idx}] student only: mean={s_mean:.4f} std={s_std:.4f}")

        # 也对比 z_lq
        if hasattr(active, "enc") and active.enc is not None:
            x_lq_dev = x_lq.to(device)
            if hasattr(active.enc, "encoder"):
                z_lq_student = active.enc.encoder(x_lq_dev).float()
            elif hasattr(active.enc, "encode"):
                z_lq_student = active.enc.encode(x_lq_dev).float()
            else:
                z_lq_student = active.enc(x_lq_dev).float()

            if z_hq_student is not None:
                delta_norm = (z_hq_student - z_lq_student).flatten(1).norm(dim=1).mean().item()
                lq_norm = z_hq_student.flatten(1).norm(dim=1).mean().item()
                hq_norm = z_hq_student.flatten(1).norm(dim=1).mean().item()
                print(f"        z_lq norm={z_lq_student.flatten(1).norm(dim=1).mean().item():.4f}  "
                      f"z_hq norm={hq_norm:.4f}  ||z_lq - z_hq||={delta_norm:.4f}")

# ---------------------------------------------------------------------------
# D. HQ latent decode 消融
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"[D] HQ LATENT DECODE ABLATION")
print(f"{'='*70}")

# 获取原始 TAESD decoder（无 SFT），用于对比
from diffusers import AutoencoderTiny
taesd_pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_taesd_dec = taesd_pretrained.decoder.to(device)
raw_taesd_dec.eval()
for p in raw_taesd_dec.parameters():
    p.requires_grad = False

diag_results = []
for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_lq, x_hq = batch[0], batch[1]
    x_lq_dev = x_lq.to(device)
    x_hq_dev = x_hq.to(device)
    ori_h, ori_w = x_hq.shape[2], x_hq.shape[3]

    with torch.no_grad():
        # 64-align
        pad_h = (64 - ori_h % 64) % 64
        pad_w = (64 - ori_w % 64) % 64
        x_lq_p = F.pad(x_lq_dev, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x_lq_dev
        x_hq_p = F.pad(x_hq_dev, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x_hq_dev

        # 编码
        z_hq = active._encode_input(x_hq_p)
        z_lq = active._encode_input(x_lq_p)
        z_mmse = active.mmse(z_lq)

        # SFT 条件
        cond_spatial = active._build_fmir_condition(x_lq_p, wavelet_cond=None)

        # ---- D1: 原始 TAESD decoder (z_hq) ----
        y_d1 = raw_taesd_dec(z_hq).clamp(0, 1)[:, :, :ori_h, :ori_w]
        psnr_d1 = float(calculate_psnr(y_d1, x_hq_dev, test_y_channel=True))

        # ---- D2: SFT decoder with ZERO condition (模拟恒等映射) ----
        # 构建全零 cond dict
        zero_cond = {}
        if cond_spatial is not None:
            for k, v in cond_spatial.items():
                zero_cond[k] = torch.zeros_like(v)
        y_d2 = active._decode_latent(z_hq, cond=zero_cond, x_lq=x_lq_p, wavelet_cond=None).clamp(0, 1)[:, :, :ori_h, :ori_w]
        psnr_d2 = float(calculate_psnr(y_d2, x_hq_dev, test_y_channel=True))

        # ---- D3: SFT decoder (z_hq, spatial_cond) ----
        y_d3 = active._decode_latent(z_hq, cond=cond_spatial, x_lq=x_lq_p, wavelet_cond=None).clamp(0, 1)[:, :, :ori_h, :ori_w]
        psnr_d3 = float(calculate_psnr(y_d3, x_hq_dev, test_y_channel=True))

        # ---- D4: SFT decoder (z_mmse, spatial_cond) ----
        y_d4 = active._decode_latent(z_mmse, cond=cond_spatial, x_lq=x_lq_p, wavelet_cond=None).clamp(0, 1)[:, :, :ori_h, :ori_w]
        psnr_d4 = float(calculate_psnr(y_d4, x_hq_dev, test_y_channel=True))

        # 保存对比图
        cmp_d = torch.cat([y_d1, y_d2, y_d3, y_d4, x_hq_dev[:, :, :ori_h, :ori_w]], dim=3)
        save_image(cmp_d, os.path.join(args.out, f"D_ablation_{idx:03d}.png"))

        diag_results.append((idx, psnr_d1, psnr_d2, psnr_d3, psnr_d4))
        print(f"  [{idx}] D1_rawTAESD(z_hq)={psnr_d1:.4f}  "
              f"D2_SFT_zero(z_hq)={psnr_d2:.4f}  "
              f"D3_SFT_spatial(z_hq)={psnr_d3:.4f}  "
              f"D4_SFT_spatial(z_mmse)={psnr_d4:.4f}")

# 汇总
avg_d1 = sum(r[1] for r in diag_results) / len(diag_results)
avg_d2 = sum(r[2] for r in diag_results) / len(diag_results)
avg_d3 = sum(r[3] for r in diag_results) / len(diag_results)
avg_d4 = sum(r[4] for r in diag_results) / len(diag_results)
print(f"\n  → Average:")
print(f"    D1 (raw TAESD dec z_hq):              {avg_d1:.4f} dB")
print(f"    D2 (SFT dec z_hq, ZERO cond):         {avg_d2:.4f} dB")
print(f"    D3 (SFT dec z_hq, LQ spatial cond):   {avg_d3:.4f} dB  ← diag_sft_hq_psnr_full")
print(f"    D4 (SFT dec z_mmse, LQ spatial cond): {avg_d4:.4f} dB  ← diag_sft_mmse_psnr_full")

# 分析
gap_d1_d2 = avg_d1 - avg_d2
gap_d2_d3 = avg_d2 - avg_d3
gap_d3_d4 = avg_d4 - avg_d3
print(f"\n  → Analysis:")
print(f"    D1→D2 gap ({gap_d1_d2:+.2f}dB): 原始TAESD vs SFT(zero cond) — SFT包装器的固有损失")
print(f"    D2→D3 gap ({gap_d2_d3:+.2f}dB): zero cond → LQ spatial cond — LQ条件对HQ latent的反向干扰")
print(f"    D3→D4 gap ({gap_d3_d4:+.2f}dB): z_hq → z_mmse (same LQ cond) — 为什么MMSE latent解码比HQ好")

# ---------------------------------------------------------------------------
# E. 修正 Flow 诊断
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"[E] CORRECTED FLOW DIAGNOSTICS")
print(f"{'='*70}")

latent_diag = {
    "charb_mmse_to_hq": [],
    "charb_fmir_to_hq": [],
    "cosine_pred": [],        # cos(delta_pred, delta_target) from z0
    "cosine_old": [],         # old metric: cos(z_final-z_mmse, z_hq-z_mmse)
    "norm_pred": [],
    "norm_target": [],
    "norm_ratio": [],
    "gain_abs": [],
    "gain_rel": [],
}

for idx, batch in enumerate(valloader):
    if idx >= args.max_samples:
        break
    x_lq, x_hq = batch[0], batch[1]
    x_lq_dev = x_lq.to(device)
    x_hq_dev = x_hq.to(device)
    ori_h, ori_w = x_hq.shape[2], x_hq.shape[3]

    # 64-align
    pad_h = (64 - ori_h % 64) % 64
    pad_w = (64 - ori_w % 64) % 64
    x_lq_p = F.pad(x_lq_dev, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x_lq_dev
    x_hq_p = F.pad(x_hq_dev, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x_hq_dev

    with torch.no_grad():
        z_hq = active._encode_input(x_hq_p)
        z_lq = active._encode_input(x_lq_p)
        z_mmse = active.mmse(z_lq)

        # 起点 z0 = z_mmse + noise（与 forward() 对齐）
        noise = active._inference_noise(z_mmse, x_lq_dev.device)
        z0 = z_mmse + noise

        # FMIR 条件
        wavelet_cond = None
        if active.wavelet_stem is not None:
            from ELIR.models.elir import pos_emb
            t_emb_init = pos_emb(torch.zeros(x_lq_p.shape[0]), active.t_emb_dim).to(x_lq_dev.device)
            wavelet_cond = active.wavelet_stem(x_lq_p, t_emb=t_emb_init)

        fmir_cond = active._build_fmir_condition(x_lq_p, wavelet_cond=wavelet_cond)
        fmir_cond = active._augment_cond_with_latent(fmir_cond, z_lq, z_mmse)

        # K-step ODE
        K = int(active.K)
        dt_val = 1.0 / K
        z_final = z0.clone()
        for k in range(K):
            t_val = k * dt_val
            t_vec = torch.full((z_final.shape[0],), t_val, device=x_lq_dev.device, dtype=z_final.dtype)
            t_emb = pos_emb(t_vec, active.t_emb_dim).to(x_lq_dev.device)
            t_tensor = t_vec[:, None, None, None]
            v_step = active.fmir(z_final, t_emb, cond=fmir_cond, t=t_tensor)
            z_final = z_final + dt_val * v_step

        # ---- 修正后的 delta 计算 ----
        delta_pred = z_final - z0          # FMIR 真实的位移量（不含 noise，因为 z0 已经含了）
        delta_target = z_hq - z0           # 目标位移量（从 z0 到 z_hq）

        # 旧版 metric（z_final - z_mmse，包含 noise）
        delta_fmir_old = z_final - z_mmse
        delta_target_old = z_hq - z_mmse

        # Per-element RMS normalization
        ne = max(z0[0].numel(), 1) ** 0.5

        pred_norm = delta_pred.flatten(1).norm(dim=1).mean().item() / ne
        target_norm = delta_target.flatten(1).norm(dim=1).mean().item() / ne
        noise_norm = noise.flatten(1).norm(dim=1).mean().item() / ne

        cos_pred = F.cosine_similarity(
            delta_pred.flatten(1), delta_target.flatten(1), dim=1
        ).mean().item()

        cos_old = F.cosine_similarity(
            delta_fmir_old.flatten(1), delta_target_old.flatten(1), dim=1
        ).mean().item()

        norm_ratio = pred_norm / (target_norm + 1e-8)

        # Charbonnier distances
        eps_c = 1e-6
        charb_mmse = torch.sqrt((z_mmse - z_hq).pow(2) + eps_c).mean().item()
        charb_fmir = torch.sqrt((z_final - z_hq).pow(2) + eps_c).mean().item()
        gain_abs = charb_mmse - charb_fmir
        gain_rel = gain_abs / (charb_mmse + 1e-8)

        latent_diag["charb_mmse_to_hq"].append(charb_mmse)
        latent_diag["charb_fmir_to_hq"].append(charb_fmir)
        latent_diag["cosine_pred"].append(cos_pred)
        latent_diag["cosine_old"].append(cos_old)
        latent_diag["norm_pred"].append(pred_norm)
        latent_diag["norm_target"].append(target_norm)
        latent_diag["norm_ratio"].append(norm_ratio)
        latent_diag["gain_abs"].append(gain_abs)
        latent_diag["gain_rel"].append(gain_rel)

        print(f"  [{idx}] noise_norm={noise_norm:.5f}  "
              f"pred_norm={pred_norm:.5f}  target_norm={target_norm:.5f}  "
              f"norm_ratio={norm_ratio:.4f}")
        print(f"        cos_pred(vs z0→z_hq)={cos_pred:.4f}  "
              f"cos_old(vs z_mmse→z_hq)={cos_old:.4f}")
        print(f"        charb: mmse→hq={charb_mmse:.6f}  fmir→hq={charb_fmir:.6f}  "
              f"gain_abs={gain_abs:.6f}  gain_rel={gain_rel:.4%}")

# 汇总
print(f"\n  → Flow Diagnostic Summary:")
for key in latent_diag:
    vals = latent_diag[key]
    if vals:
        avg_v = sum(vals) / len(vals)
        print(f"    avg {key}: {avg_v:.6f}")

# ---------------------------------------------------------------------------
# F. 单图/4图 Overfit
# ---------------------------------------------------------------------------
if not args.skip_overfit:
    print(f"\n{'='*70}")
    print(f"[F] OVERFIT TEST ({args.overfit_steps} steps, lr={args.overfit_lr}, "
          f"max {args.max_samples} pairs)")
    print(f"{'='*70}")

    # 收集 overfit 数据
    overfit_batches = []
    for idx, batch in enumerate(valloader):
        if idx >= min(4, args.max_samples):
            break
        overfit_batches.append(batch)

    if len(overfit_batches) == 0:
        print("  No data for overfit — skipping.")
    else:
        # 复制模型用于 overfit
        import copy
        model_ov = copy.deepcopy(model)
        model_ov.train()
        # 确保 trainable 状态正确
        if hasattr(model_ov, "enc") and model_ov.enc is not None:
            if getattr(model_ov, "enc_trainable", False):
                model_ov.enc.train()
            else:
                model_ov.enc.eval()
        if hasattr(model_ov, "dec") and model_ov.dec is not None:
            if getattr(model_ov, "dec_trainable", False):
                if hasattr(model_ov.dec, "set_trainable_only"):
                    model_ov.dec.set_trainable_only()
                model_ov.dec.train()
        model_ov.to(device)

        # 只优化 FMIR + MMSE（快速测试）
        opt_params = []
        for n, p in model_ov.named_parameters():
            if p.requires_grad and (n.startswith("fmir.") or n.startswith("mmse.")):
                opt_params.append(p)
        if not opt_params:
            # fallback: 所有可训练参数
            opt_params = [p for p in model_ov.parameters() if p.requires_grad]
        optimizer_ov = torch.optim.AdamW(opt_params, lr=args.overfit_lr)

        print(f"  optimizing {sum(p.numel() for p in opt_params):,} params")
        print(f"  overfit on {len(overfit_batches)} image pairs")

        overfit_log = []

        for step in range(args.overfit_steps):
            total_loss = 0.0
            for batch in overfit_batches:
                x_lq, x_hq = batch[0].to(device), batch[1].to(device)

                # 64-align
                _, _, oh, ow = x_hq.shape
                ph = (64 - oh % 64) % 64
                pw = (64 - ow % 64) % 64
                if ph or pw:
                    x_lq_in = F.pad(x_lq, (0, pw, 0, ph), mode="reflect")
                    x_hq_in = F.pad(x_hq, (0, pw, 0, ph), mode="reflect")
                else:
                    x_lq_in, x_hq_in = x_lq, x_hq

                # 简化 loss: CharB(z_mmse, z_hq) + CharB(v_pred, target_v)
                z_lq_ov = model_ov._encode_input(x_lq_in)
                z_hq_ov = model_ov._encode_input(x_hq_in).detach()  # detach target
                z_mmse_ov = model_ov.mmse(z_lq_ov)

                # FMIR condition
                cond_ov = model_ov._build_fmir_condition(x_lq_in, wavelet_cond=None)
                cond_ov = model_ov._augment_cond_with_latent(cond_ov, z_lq_ov, z_mmse_ov)

                # One-step velocity prediction at t=0
                from ELIR.models.elir import pos_emb
                t0 = torch.zeros(x_lq.shape[0], device=device)
                v_pred = model_ov.fmir(z_mmse_ov, pos_emb(t0, model_ov.t_emb_dim).to(device),
                                       cond=cond_ov, t=t0[:, None, None, None])
                target_v = z_hq_ov - z_mmse_ov

                loss_mmse = F.mse_loss(z_mmse_ov, z_hq_ov)
                loss_flow = F.mse_loss(v_pred, target_v)
                loss = loss_mmse + 0.1 * loss_flow

                optimizer_ov.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
                optimizer_ov.step()

                total_loss += loss.item()

            if step % 50 == 0 or step == args.overfit_steps - 1:
                # 评估
                with torch.no_grad():
                    ov_psnr_vals = []
                    ov_hq_psnr_vals = []
                    for batch in overfit_batches:
                        x_lq, x_hq = batch[0].to(device), batch[1].to(device)
                        _, _, oh, ow = x_hq.shape
                        ph = (64 - oh % 64) % 64
                        pw = (64 - ow % 64) % 64
                        x_lq_in = F.pad(x_lq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_lq

                        # 推理
                        cond_ov_test = model_ov._build_fmir_condition(x_lq_in, wavelet_cond=None)
                        z_lq_test = model_ov._encode_input(x_lq_in)
                        z_mmse_test = model_ov.mmse(z_lq_test)
                        cond_ov_test = model_ov._augment_cond_with_latent(cond_ov_test, z_lq_test, z_mmse_test)

                        noise_test = model_ov._inference_noise(z_mmse_test, device)
                        z_flow = z_mmse_test + noise_test
                        K_ov = int(model_ov.K)
                        dt_ov = 1.0 / K_ov
                        for k in range(K_ov):
                            t_val = k * dt_ov
                            t_vec = torch.full((z_flow.shape[0],), t_val, device=device, dtype=z_flow.dtype)
                            t_tensor = t_vec[:, None, None, None]
                            vk = model_ov.fmir(z_flow, pos_emb(t_vec, model_ov.t_emb_dim).to(device),
                                              cond=cond_ov_test, t=t_tensor)
                            z_flow = z_flow + dt_ov * vk

                        dec_cond = model_ov._build_fmir_condition(x_lq_in, wavelet_cond=None)
                        y_pred = model_ov._decode_latent(z_flow, cond=dec_cond, x_lq=x_lq_in, wavelet_cond=None)
                        y_pred = y_pred[:, :, :oh, :ow].clamp(0, 1)
                        ov_psnr_vals.append(float(calculate_psnr(y_pred, x_hq, test_y_channel=True)))

                        # HQ decode diagnostic (SFT decode z_hq with LQ cond)
                        z_hq_test = model_ov._encode_input(
                            F.pad(x_hq, (0, pw, 0, ph), mode="reflect") if ph or pw else x_hq.to(device))
                        y_hq_dec = model_ov._decode_latent(z_hq_test, cond=dec_cond,
                                                           x_lq=x_lq_in, wavelet_cond=None)
                        y_hq_dec = y_hq_dec[:, :, :oh, :ow].clamp(0, 1)
                        ov_hq_psnr_vals.append(float(calculate_psnr(y_hq_dec, x_hq, test_y_channel=True)))

                avg_ov_psnr = sum(ov_psnr_vals) / len(ov_psnr_vals)
                avg_ov_hq_psnr = sum(ov_hq_psnr_vals) / len(ov_hq_psnr_vals)
                overfit_log.append((step, total_loss / len(overfit_batches), avg_ov_psnr, avg_ov_hq_psnr))
                print(f"  step {step:5d}: loss={total_loss/len(overfit_batches):.6f}  "
                      f"PSNR={avg_ov_psnr:.4f}  diag_sft_hq_PSNR={avg_ov_hq_psnr:.4f}")

        # 最终对比
        init_psnr = overfit_log[0][2] if overfit_log else 0
        final_psnr = overfit_log[-1][2] if overfit_log else 0
        init_hq = overfit_log[0][3] if overfit_log else 0
        final_hq = overfit_log[-1][3] if overfit_log else 0
        print(f"\n  → Overfit Summary:")
        print(f"    PSNR:           {init_psnr:.4f} → {final_psnr:.4f}  (Δ={final_psnr-init_psnr:+.2f})")
        print(f"    diag_sft_hq:    {init_hq:.4f} → {final_hq:.4f}  (Δ={final_hq-init_hq:+.2f})")

        if final_psnr - init_psnr > 2.0:
            print(f"    ✓ 过拟合有效提升 PSNR (+{final_psnr-init_psnr:.1f}dB)")
            print(f"      说明 FMIR+MMSE 在当前 latent 分辨率下有能力学习去雨")
            print(f"      → 问题不在 8× 下采样瓶颈，在训练策略/loss权重/收敛")
        elif final_psnr - init_psnr > 0.5:
            print(f"    ~ 过拟合有微弱提升 (+{final_psnr-init_psnr:.1f}dB)")
            print(f"      → 模型有能力但需要更多 steps 或更好的 loss 设计")
        else:
            print(f"    ✗ 过拟合无法提升 PSNR (+{final_psnr-init_psnr:.1f}dB)")
            print(f"      → 可能是架构瓶颈或 latent 分辨率问题")
            if final_hq - init_hq < 0.5:
                print(f"      → diag_sft_hq 也未改善，SFT decoder 可能是硬瓶颈")

        # 保存过拟合曲线
        with open(os.path.join(args.out, "F_overfit_log.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "psnr", "diag_sft_hq_psnr"])
            writer.writerows(overfit_log)

# ---------------------------------------------------------------------------
# 最终汇总
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"DIAGNOSIS COMPLETE — check {args.out}/ for outputs")
print(f"{'='*70}")
