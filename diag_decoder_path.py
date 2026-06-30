"""
最小核对：TAESD encoder ⇔ decoder 往返 vs SFT decoder 往返
检查是否存在 tanh / scaling / conv_in bypass 导致的不一致
"""
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载原始 TAESD
taesd3 = AutoencoderTiny.from_pretrained("madebyollin/taesd3").to(device).eval()
raw_enc = taesd3.encoder
raw_dec = taesd3.decoder

print(f"raw_enc type: {type(raw_enc).__name__}")
print(f"raw_dec type: {type(raw_dec).__name__}")

# 检查 decoder 的 forward 是否包含预处理
import inspect
try:
    src = inspect.getsource(raw_dec.forward)
    print(f"\nraw_dec.forward() source:\n{src[:500]}")
except:
    print("\n[cannot get source for raw_dec.forward]")

# 检查关键属性
print(f"\nraw_dec attributes:")
for attr in ["conv_in", "conv_out", "up_blocks", "layers", "tanh", "scale"]:
    if hasattr(raw_dec, attr):
        val = getattr(raw_dec, attr)
        print(f"  .{attr}: {type(val).__name__}", end="")
        if isinstance(val, torch.nn.Module):
            print(f"  (module)")
        else:
            print(f"  = {val}")

# 查看 conv_in 的结构
if hasattr(raw_dec, "conv_in"):
    print(f"\n  conv_in: {raw_dec.conv_in}")

# 创建 SFT decoder（零初始化）
from ELIR.models.sft_taesd_finetuner import SFT_TAESDFineTuner
sft_dec = SFT_TAESDFineTuner(raw_dec, latent_channels=16, gamma_scale=0.1, beta_scale=0.1).to(device).eval()

# 确认 SFT 走哪个 forward 路径
if hasattr(raw_dec, "conv_in") and hasattr(raw_dec, "up_blocks"):
    print(f"\nSFT forward path: _forward_from_up_blocks (conv_in + up_blocks)")
    print(f"  raw_dec.conv_in is used directly, NOT raw_dec.forward()")
elif hasattr(raw_dec, "layers"):
    print(f"\nSFT forward path: _forward_from_layers")
else:
    print(f"\nSFT forward path: UNKNOWN")

# 用一张随机图测试
x = torch.randn(1, 3, 256, 256, device=device).clamp(0, 1)

with torch.no_grad():
    # Encoder: raw vs tiny_enc
    z_raw = raw_enc(x)

    # 同时用 TAESD.py Encoder 产生 latent
    from ELIR.models.load_model import get_model
    tiny_enc_cfg = {"name": "tiny_enc", "trainable": False, "params": {}}
    tiny_enc_model = get_model(tiny_enc_cfg).to(device).eval()
    z_tiny = tiny_enc_model(x)

    print(f"\n--- Encoder comparison ---")
    print(f"z_raw  mean={z_raw.mean().item():.4f}  std={z_raw.std().item():.4f}  "
          f"min={z_raw.min().item():.4f}  max={z_raw.max().item():.4f}")
    print(f"z_tiny mean={z_tiny.mean().item():.4f}  std={z_tiny.std().item():.4f}  "
          f"min={z_tiny.min().item():.4f}  max={z_tiny.max().item():.4f}")
    print(f"diff max={ (z_raw - z_tiny).abs().max().item():.6f}  "
          f"cos={F.cosine_similarity(z_raw.flatten(1), z_tiny.flatten(1)).item():.6f}")

    # --- Decoder 对比 (用 z_raw 即 diffusers encoder 的输出) ---
    # 1. raw_dec(z_raw) — 完整 forward()
    y1 = raw_dec(z_raw).clamp(0, 1)

    # 2. raw_dec.conv_in(z_raw) → 逐层手动跑
    # 先看 conv_in(z_raw) 和 forward 第一步的差异
    y2 = raw_dec.conv_in(z_raw)

    # 3. SFT_dec(z_raw, zero_cond) — SFT 包装器
    zero_cond = {
        "32": torch.zeros(1, 64, z_raw.shape[2]*8, z_raw.shape[3]*8, device=device),
        "64": torch.zeros(1, 64, z_raw.shape[2]*8, z_raw.shape[3]*8, device=device),
        "128": torch.zeros(1, 32, z_raw.shape[2]*8, z_raw.shape[3]*8, device=device),
        "256": torch.zeros(1, 16, z_raw.shape[2]*8, z_raw.shape[3]*8, device=device),
    }
    y3 = sft_dec(z_raw, zero_cond).clamp(0, 1)

    # 4. 如果 forward 有 tanh，手动模拟
    y1_tanh = raw_dec(torch.tanh(z_raw / 3) * 3).clamp(0, 1)

    print(f"\n--- Decoder comparison (using z_raw, same encoder) ---")
    print(f"raw_dec(z_raw)     output mean={y1.mean().item():.4f} std={y1.std().item():.4f}")
    print(f"raw_dec.conv_in(z) output mean={y2.mean().item():.4f} std={y2.std().item():.4f}  (shape={list(y2.shape)})")
    print(f"SFT_dec(z, zero)   output mean={y3.mean().item():.4f} std={y3.std().item():.4f}")

    # Check if tanh is applied inside raw_dec.forward
    # If it is, y1 and y1_tanh should be identical
    diff_1_tanh = (y1 - y1_tanh).abs().max().item()
    print(f"\ny1 vs y1(tanh scaled input) diff: {diff_1_tanh:.6f}  "
          f"{'→ tanh IS in forward()' if diff_1_tanh < 1e-3 else '→ tanh NOT in forward() or differs'}")

    # Compare raw_dec vs SFT_dec with same input
    # If SFT zero-cond is identity, y1 should ≈ y3
    from ELIR.metrics import calculate_psnr
    psnr_1v3 = calculate_psnr(y1, y3, test_y_channel=False)
    print(f"\nraw_dec vs SFT_dec(zero) PSNR: {psnr_1v3:.2f} dB  "
          f"{'≈ same' if psnr_1v3 > 40 else '⚠️ DIFFERENT — code path mismatch!'}")

    if psnr_1v3 < 40:
        print(f"\n!!! CONFIRMED: raw decoder forward() and SFT decoder conv_in path "
              f"produce DIFFERENT outputs for the SAME latent.")
        print(f"!!! This means the pretrained TAESD weights expect tanh/scale preprocessing "
              f"that the SFT path bypasses.")
