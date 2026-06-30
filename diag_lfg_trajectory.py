"""LFG-SFG 轨迹诊断 + checkpoint 加载验证。

用法:
python diag_lfg_trajectory.py CONFIG.yaml LQ.png HQ.png CKPT.ckpt [K_steps]
"""

import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))

from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.models.elir import pos_emb
from ELIR.metrics import calculate_psnr, calculate_ssim
from PIL import Image
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

yaml_path = sys.argv[1]
lq_path   = sys.argv[2]
hq_path   = sys.argv[3]
ckpt_path = sys.argv[4]
k_steps   = int(sys.argv[5]) if len(sys.argv) > 5 else None
device    = "cuda"

out_dir = "diag_lfg_out"
os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# 1. 加载 YAML + 构建模型
# ============================================================================
with open(yaml_path, encoding="utf-8") as f:
    conf = load_hyperpyyaml(f)
arch_cfg = conf["model_cfg"]["arch_cfg"]
model = get_model(arch_cfg).to(device).eval()
model_state = model.state_dict()

# ============================================================================
# 2. 加载 ckpt 并构建完整 state_dict (raw + EMA overlay)
# ============================================================================
ckpt = torch.load(ckpt_path, map_location="cpu")

# 原始 state_dict (带 model. 前缀, 剥掉)
raw = ckpt.get("state_dict", ckpt)
state = {(k[6:] if k.startswith("model.") else k): v for k, v in raw.items()}

# EMA / submodule overlay
EMAS = [
    ("state_dict_mmse",    "mmse."),
    ("state_dict_fmir",    "fmir."),
    ("state_dict_enc",     "enc."),
    ("state_dict_dec",     "dec."),
    ("state_dict_wavelet", "wavelet_stem."),
]
for block_key, prefix in EMAS:
    if block_key not in ckpt:
        print(f"[ckpt] MISSING block: {block_key}")
        continue
    n = 0
    for k, v in ckpt[block_key].items():
        full_k = k if k.startswith(prefix) else (prefix + k)
        state[full_k] = v
        n += 1
    print(f"[ckpt] EMA overlay: {block_key} -> {n} keys")

# ============================================================================
# 3. 分类统计 missing / shape mismatch
# ============================================================================
BAD = {}  # key -> (model_shape, ckpt_shape or None)
for k, mv in model_state.items():
    if k not in state:
        BAD[k] = (tuple(mv.shape), None)
    elif mv.shape != state[k].shape:
        BAD[k] = (tuple(mv.shape), tuple(state[k].shape))

# 按模块分类
_modules = ["mmse", "fmir", "enc", "dec", "wavelet"]
stats = {m: 0 for m in _modules}
other_bad = 0
for k in BAD:
    found = False
    for m in _modules:
        if k.startswith(m + "."):
            stats[m] += 1
            found = True
            break
    if not found:
        other_bad += 1

print(f"\n[verify] total bad keys: {len(BAD)}")
for m in _modules:
    print(f"  {m}: {stats[m]}")
print(f"  other: {other_bad}")

if BAD:
    print("\n[verify] BAD top 30:")
    for i, (k, (ms, cs)) in enumerate(sorted(BAD.items())[:30]):
        print(f"  {k}: model{ms} vs ckpt{cs}")

# ============================================================================
# 4. 加载有效权重
# ============================================================================
loaded = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
model.load_state_dict(loaded, strict=False)
print(f"\n[load] loaded {len(loaded)} / {len(model_state)} keys")

if k_steps is not None:
    model.K = k_steps
    model.dt = 1.0 / k_steps

# ============================================================================
# 5. 数据加载
# ============================================================================
# 和训练 val 一样的 LQ 直方图均衡预处理
lq_raw = Image.open(os.path.expanduser(lq_path)).convert("RGB")
hq_raw = Image.open(os.path.expanduser(hq_path)).convert("RGB")
from ELIR.datasets.lol import LOLDataset
dummy = LOLDataset.__new__(LOLDataset)
lq_eq = dummy._equalize_low_light(lq_raw)
lq = torch.from_numpy(__import__("numpy").array(lq_eq)).float() / 255.
hq = torch.from_numpy(__import__("numpy").array(hq_raw)).float() / 255.
lq, hq = lq.permute(2, 0, 1).unsqueeze(0), hq.permute(2, 0, 1).unsqueeze(0)
lq, hq = lq.to(device), hq.to(device)

# ============================================================================
# 6. VALIDATION-IDENTICAL step0 诊断 (比对 diag_sft_mmse_psnr)
# ============================================================================
with torch.no_grad():
    z_lq   = model._encode_input(lq)
    z_hq   = model._encode_input(hq)
    z_mmse = model.mmse(z_lq)

    cond_spatial = model._build_fmir_condition(lq, wavelet_cond=None)
    wv = model.wavelet_stem(lq) if model.wavelet_stem else None

    dec_hq   = model._decode_latent(z_hq,   cond=cond_spatial, x_lq=lq, wavelet_cond=wv).clamp(0, 1)
    dec_mmse = model._decode_latent(z_mmse, cond=cond_spatial, x_lq=lq, wavelet_cond=wv).clamp(0, 1)

    ori_h, ori_w = hq.shape[2], hq.shape[3]
    dec_hq   = dec_hq[:, :, :ori_h, :ori_w]
    dec_mmse = dec_mmse[:, :, :ori_h, :ori_w]

psnr_sft_hq   = calculate_psnr(dec_hq,   hq, test_y_channel=True)
psnr_sft_mmse = calculate_psnr(dec_mmse, hq, test_y_channel=True)

print(f"\n{'':<16} {'PSNR':>8} {'SSIM':>8}")
print(f"{'SFT_dec(HQ)':<16} {psnr_sft_hq:>8.2f}")
print(f"{'SFT_dec(MMSE)':<16} {psnr_sft_mmse:>8.2f}")

EXPECTED_MMSE_PSNR = 23.5
if psnr_sft_mmse < EXPECTED_MMSE_PSNR - 1.0:
    print(f"\n[FATAL] SFT_dec(MMSE)={psnr_sft_mmse:.2f} << expected ~{EXPECTED_MMSE_PSNR}")
    print("check: 1) ckpt/yaml 匹配  2) LQ/HQ 是否同一张  3) enc/dec/mmse  missing 数量")
    if stats["mmse"] > 0 or stats["enc"] > 0 or stats["dec"] > 0:
        print(f"[FATAL] mmse_missing={stats['mmse']}, enc_missing={stats['enc']}, dec_missing={stats['dec']}")
    sys.exit(1)

# ============================================================================
# 7. FMIR 消融诊断
# ============================================================================
with torch.no_grad():
    _, _, oh, ow = lq.shape
    ph = (64 - oh % 64) % 64
    pw = (64 - ow % 64) % 64
    xs = torch.nn.functional.pad(lq, (0, pw, 0, ph), mode="reflect") if ph or pw else lq

    t0 = pos_emb(torch.zeros(xs.shape[0]), 160).to(device)
    wv_cond = model.wavelet_stem(xs, t_emb=t0) if model.wavelet_stem else None
    fmir_c = model._build_fmir_condition(xs, wavelet_cond=wv_cond)
    dec_c  = model._build_fmir_condition(xs, wavelet_cond=None)

    z = model._encode_input(xs)
    nz = model.noise.to(device)
    if nz.shape[-2:] != z.shape[-2:]:
        nz = torch.nn.functional.interpolate(nz, size=z.shape[-2:], mode="bilinear", align_corners=False)

    def _dec(_z):
        return model._decode_latent(_z, cond=dec_c, x_lq=xs, wavelet_cond=None)[:, :, :oh, :ow]

    def _ode(_z0, K_val=None, scale=1.0):
        K_use = K_val if K_val is not None else model.K
        _dt = 1.0 / K_use
        _zt = _z0.clone()
        traj = [_dec(_zt)]
        for _k in range(K_use):
            _t = _k * _dt
            _tt = torch.full((_zt.shape[0], 1, 1, 1), _t, device=device, dtype=_zt.dtype)
            _v = model.fmir(_zt, pos_emb(_t, model.t_emb_dim).to(device), cond=fmir_c, t=_tt)
            _zt = _zt + scale * _dt * _v
            traj.append(_dec(_zt.clone()))
        return traj

    z_mmse = model.mmse(z)
    z_noise = z_mmse + nz

    # --- 1. decode(MMSE) ---
    y_mmse = _dec(z_mmse)
    p_mmse = calculate_psnr(y_mmse, hq, test_y_channel=True)
    s_mmse = calculate_ssim(y_mmse, hq, test_y_channel=True)
    save_image(y_mmse.clamp(0, 1), f"{out_dir}/abl1_mmse.png")
    print(f"\--- Ablation ---")
    print(f"1. decode(MMSE)            PSNR={p_mmse:>7.2f}  SSIM={s_mmse:.4f}")

    # --- 2. decode(MMSE+noise) ---
    y_noise = _dec(z_noise)
    p_noise = calculate_psnr(y_noise, hq, test_y_channel=True)
    s_noise = calculate_ssim(y_noise, hq, test_y_channel=True)
    save_image(y_noise.clamp(0, 1), f"{out_dir}/abl2_mmse_noise.png")
    print(f"2. decode(MMSE+noise)      PSNR={p_noise:>7.2f}  SSIM={s_noise:.4f}")

    # --- 3. ODE trajectory (standard) ---
    trajs_std = _ode(z_noise)
    print(f"\n3. Standard ODE (K={model.K}):")
    for i, t in enumerate(trajs_std):
        p = calculate_psnr(t, hq, test_y_channel=True)
        s = calculate_ssim(t, hq, test_y_channel=True)
        note = "<< MMSE+noise" if i == 0 else ("<< Final" if i == len(trajs_std)-1 else "")
        print(f"   step{i}: PSNR={p:>7.2f}  SSIM={s:.4f}  {note}")
        save_image(t.clamp(0, 1), f"{out_dir}/abl3_ode_std_{i:02d}.png")

    # --- 4. No-noise ODE ---
    trajs_nn = _ode(z_mmse)
    print(f"\n4. No-noise ODE (K={model.K}):")
    for i, t in enumerate(trajs_nn):
        p = calculate_psnr(t, hq, test_y_channel=True)
        s = calculate_ssim(t, hq, test_y_channel=True)
        note = "<< MMSE" if i == 0 else ("<< Final" if i == len(trajs_nn)-1 else "")
        print(f"   step{i}: PSNR={p:>7.2f}  SSIM={s:.4f}  {note}")
        save_image(t.clamp(0, 1), f"{out_dir}/abl4_ode_nonoise_{i:02d}.png")

    # --- 5. Small-step ODE ---
    for K_test in [1, 3]:
        tr = _ode(z_noise, K_val=K_test)
        p1 = calculate_psnr(tr[0], hq, test_y_channel=True)
        pk = calculate_psnr(tr[-1], hq, test_y_channel=True)
        print(f"5a. K={K_test} ODE:  step0={p1:.2f}  final={pk:.2f}")

    for scl in [0.25, 0.5]:
        tr = _ode(z_noise, scale=scl)
        pk = calculate_psnr(tr[-1], hq, test_y_channel=True)
        print(f"5b. ode_scale={scl}: final={pk:.2f}")

    # --- 6. 诊断 ---
    print(f"\n--- Diagnostic ---")
    if p_mmse > 21:
        if p_noise < p_mmse - 2:
            print(">>> Noise 破坏了 MMSE latent → sigma_s 或 noise scale 问题")
        if p_noise > 20 and calculate_psnr(trajs_std[-1], hq, test_y_channel=True) < p_noise - 2:
            print(">>> FMIR ODE 越推越差 → FMIR vector field 方向错误(频率条件?)")
        if calculate_psnr(trajs_nn[-1], hq, test_y_channel=True) < p_mmse - 2:
            print(">>> No-noise ODE 也下降 → FMIR 本身有问题, 不是 noise 的锅")
        if p_mmse > calculate_psnr(trajs_std[-1], hq, test_y_channel=True):
            print(">>> BEST = decode(MMSE) → skip FMIR 反而更好")

# ============================================================================
# 8. Gate heatmaps
# ============================================================================
if hasattr(model, "wavelet_stem") and hasattr(model.wavelet_stem, "gate_256"):
    print("\n--- Gate heatmaps ---")
    ws = model.wavelet_stem
    _, debug = ws(lq, return_debug=True, t_emb=t0)
    for key in ["gate256", "gate128", "gate64", "gate32"]:
        g = debug.get(key)
        if g is None:
            continue
        g_np = g.detach()[0, 0].cpu().numpy()
        print(f"  {key}: mean={g_np.mean():.4f}  min={g_np.min():.4f}  max={g_np.max():.4f}")
        plt.figure(figsize=(5, 4))
        plt.imshow(g_np, cmap="hot", vmin=0, vmax=1)
        plt.colorbar(label="gate")
        plt.title(f"{key}  mean={g_np.mean():.4f}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{key}_heatmap.png", dpi=100)
        plt.close()

save_image(lq,  f"{out_dir}/lq.png")
save_image(hq,  f"{out_dir}/hq.png")
print(f"\nSaved: {out_dir}/")
