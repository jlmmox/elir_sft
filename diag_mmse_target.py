"""MMSE 诊断: 加载 ckpt, 打印加载状态, decode 对比 (TAESD vs SFT decoder)"""

import sys, torch, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(__file__))

from hyperpyyaml import load_hyperpyyaml
from ELIR.models.load_model import get_model
from ELIR.metrics import calculate_psnr, calculate_ssim
from PIL import Image

yaml_path = sys.argv[1]
lq_path   = sys.argv[2]
hq_path   = sys.argv[3]
if len(sys.argv) < 5:
    print("用法: python diag_mmse_target.py <yaml> <lq> <hq> <ckpt>")
    sys.exit(1)

ckpt_path = sys.argv[4]
device = 'cuda'

with open(yaml_path) as f:
    conf = load_hyperpyyaml(f)

model = get_model(conf["model_cfg"]["arch_cfg"])
model.to(device)
model.eval()

# ---- 加载 checkpoint (必填) ----
if not os.path.exists(os.path.expanduser(ckpt_path)):
    print(f"[ckpt] ERROR: not found: {ckpt_path}")
    sys.exit(1)

print(f"[ckpt] loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
state = None

# 优先加载 EMA 权重 (训练 val 路径用的是 self.ema.model)
ema_found = False
for k in ['state_dict_ema','ema_state_dict','params_ema']:
    if k in ckpt:
        state = ckpt[k]
        print(f"[ckpt] using EMA: {k}")
        ema_found = True
        break
# 检查 callbacks 里的 EMA
if not ema_found and 'callbacks' in ckpt:
    cb = ckpt['callbacks']
    if isinstance(cb, dict) and 'ModelEMA' in cb:
        state = cb['ModelEMA']
        print("[ckpt] using callback/ModelEMA")
        ema_found = True
    elif hasattr(cb, 'state_dict') and hasattr(cb.state_dict(), 'keys'):
        cb_state = cb.state_dict()
        if cb_state:
            state = cb_state
            print("[ckpt] using callbacks state_dict (EMA)")

# Fallback: regular keys
if state is None:
    for k in ['state_dict','model','params','net_g','state_dict_sft']:
        if k in ckpt:
            state = ckpt[k]
            print(f"[ckpt] fallback: {k}")
            break
if state is None:
    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
        print("[ckpt] using raw checkpoint as state_dict")
    else:
        print(f"[ckpt] ERROR: cannot find state_dict. top keys: {list(ckpt.keys())[:10]}")
        sys.exit(1)

# 剥掉 model. 前缀 (ckpt 由 Lightning 保存时自动加上的)
state = {(k[6:] if k.startswith('model.') else k): v for k, v in state.items()}
print(f"[ckpt] raw state_dict: {len(state)} keys")

# ---- EMA overlay: 用 EMA 分块覆盖 raw state_dict ----
# EMA 分块内 key 不带模块前缀 (如 conv.weight), 需要加前缀
ema_blocks = {
    'state_dict_mmse': 'mmse.',
    'state_dict_fmir': 'fmir.',
    'state_dict_enc':  'enc.',
    'state_dict_dec':  'dec.',
    'state_dict_sft':  '',        # sft 的 key 已经带 sft_refiner. 前缀
}
for block_name, prefix in ema_blocks.items():
    if block_name not in ckpt:
        print(f"[ckpt] EMA block NOT FOUND: {block_name}")
        continue
    ema_raw = ckpt[block_name]
    count = 0
    for k, v in ema_raw.items():
        # 给 key 加模块前缀 (如果还没带)
        full_k = k if k.startswith(prefix) else prefix + k
        if full_k in state:
            state[full_k] = v
            count += 1
    print(f"[ckpt] EMA overlay: {block_name} -> {count}/{len(ema_raw)} keys merged (prefix='{prefix}')")

model_state = model.state_dict()
filtered = {}
for k, v in state.items():
    if k in model_state and model_state[k].shape == v.shape:
        filtered[k] = v

missing = [k for k in model_state if k not in filtered]
unexpected = [k for k in state if k not in model_state]
print(f"[ckpt] loaded {len(filtered)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    m20 = missing[:20]
    print(f"[ckpt] missing: {m20}")

mmse_loaded = any('mmse' in k for k in filtered)
print(f"[ckpt] mmse keys present: {mmse_loaded}")
if mmse_loaded:
    for name, p in model.named_parameters():
        if 'mmse' in name and 'weight' in name and p.ndim >= 4:
            print(f"[mmse] {name}: mean={p.mean().item():.4f} std={p.std().item():.4f}")
            break
else:
    print("[ckpt] WARNING: no mmse keys loaded")

model.load_state_dict(filtered, strict=False)

if len(filtered) == 0 or not any(k.startswith("mmse.") for k in filtered):
    raise RuntimeError(
        f"checkpoint not loaded correctly: {len(filtered)} keys, "
        f"mmse present={any(k.startswith('mmse.') for k in filtered)}. "
        f"State keys sample: {list(state.keys())[:5]}"
    )
print("[ckpt] model weights restored")

# ---- 数据加载 ----
lq = torch.from_numpy(__import__("numpy").array(
    Image.open(os.path.expanduser(lq_path)).convert("RGB"))).float() / 255.
hq = torch.from_numpy(__import__("numpy").array(
    Image.open(os.path.expanduser(hq_path)).convert("RGB"))).float() / 255.
lq, hq = lq.permute(2,0,1).unsqueeze(0), hq.permute(2,0,1).unsqueeze(0)
lq, hq = lq.to(device), hq.to(device)

# ---- 诊断 ----
with torch.no_grad():
    z_hq  = model._encode_input(hq)
    z_lq  = model._encode_input(lq)
    z_mmse = model.mmse(z_lq)

    C_CNN = model._build_fmir_condition(lq, wavelet_cond=None)
    t0 = torch.zeros(1).to(device)
    from ELIR.models.elir import pos_emb
    te = pos_emb(t0, 160).to(device)
    C_wav = model.wavelet_stem(lq, t_emb=te) if model.wavelet_stem else None

    # ---- 1. SFT 解码器 ----
    dec_sft_hq   = model._decode_latent(z_hq,   cond=C_CNN, x_lq=lq, wavelet_cond=C_wav).clamp(0,1)
    dec_sft_mmse = model._decode_latent(z_mmse, cond=C_CNN, x_lq=lq, wavelet_cond=C_wav).clamp(0,1)

    # ---- 2. 纯 TAESD 解码器 (无 SFT) ----
    taesd_dec = getattr(model.dec, 'taesd_decoder', None)
    if taesd_dec is None:
        # sft_taesd_finetuner 包装了 taesd_decoder
        taesd_dec = getattr(model.dec, 'taesd_decoder', model.dec)

    try:
        dec_raw_hq   = taesd_dec(z_hq).clamp(0,1)
        dec_raw_mmse = taesd_dec(z_mmse).clamp(0,1)
        has_raw = True
    except Exception as e:
        print(f"[raw decode] skipping: {e}")
        dec_raw_hq = dec_raw_mmse = None
        has_raw = False

    # ---- 报表 ----
    col = f"{'':<12} {'PSNR':>8} {'SSIM':>8}"
    print(f"\n{col}")
    print("-" * 32)

    def row(label, img):
        if img is None: return
        p = calculate_psnr(img, hq, test_y_channel=True)
        s = calculate_ssim(img, hq, test_y_channel=True)
        print(f"{label:<12} {p:>8.2f} {s:>8.4f}")

    row("SFT_dec(HQ)", dec_sft_hq)
    row("SFT_dec(MM)", dec_sft_mmse)
    if has_raw:
        row("TAESD_dec(HQ)", dec_raw_hq)
        row("TAESD_dec(MM)", dec_raw_mmse)

    # ---- 3. full forward (含 FMIR ODE) ----
    with torch.no_grad():
        y_full = model(lq).clamp(0,1)
    p_full = calculate_psnr(y_full, hq, test_y_channel=True)
    s_full = calculate_ssim(y_full, hq, test_y_channel=True)
    print(f"\nfull forward (FMIR ODE): PSNR={p_full:.2f} SSIM={s_full:.4f}")

    from torchvision.utils import save_image
    save_image(dec_sft_hq,   "diag_sft_hq.png")
    save_image(dec_sft_mmse, "diag_sft_mmse.png")
    save_image(y_full,       "diag_full_fwd.png")
    if has_raw:
        save_image(dec_raw_hq,   "diag_raw_hq.png")
        save_image(dec_raw_mmse, "diag_raw_mmse.png")
    print("saved: diag_*.png")
