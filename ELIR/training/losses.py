import torch
import torch.nn.functional as F
import math
import random

from ELIR.models.gan import hinge_g_loss, hinge_d_loss

try:
    import importlib
    ssim = importlib.import_module("piq").ssim
except Exception:
    def ssim(pred, target, data_range=1.0):
        return differentiable_ssim(pred, target, data_range=data_range)


def _gaussian_kernel(window_size=11, sigma=1.5, channels=3, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum()
    kernel = kernel2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    return kernel


def differentiable_ssim(x, y, window_size=11, sigma=1.5, data_range=1.0, eps=1e-8):
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects same shape for x and y, got {x.shape} vs {y.shape}")
    c = x.shape[1]
    kernel = _gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        channels=c,
        device=x.device,
        dtype=x.dtype,
    )
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=c)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)
    return ssim_map.mean()


def _decode_with_module(module, z, cond_dict=None, x_lq=None, wavelet_cond_dict=None):
    if module is None:
        return z
    if hasattr(module, "_use_encoder_skip_wavelet"):
        if isinstance(cond_dict, dict) and "fmir_spatial" in cond_dict:
            return module(z, cond_dict["fmir_spatial"], cond_dict["wavelet"], x_lq)
        return module(z, cond_dict, wavelet_cond_dict, x_lq)
    if hasattr(module, "_use_encoder_skip"):
        return module(z, cond_dict, x_lq)
    if hasattr(module, "_force_sft_trainable_only"):
        return module(z, cond_dict)
    if hasattr(module, "decode"):
        return module.decode(z)
    if hasattr(module, "decoder"):
        return module.decoder(z)
    return module(z)



def pos_emb(t, t_dim, scale=1000):
    assert t_dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.to(dtype=torch.float32).reshape(-1)
    device = t.device
    half_dim = t_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = scale * t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
    return emb.detach()


def _encode_with_module(module, x):
    if hasattr(module, "encode"):
        return module.encode(x)
    if hasattr(module, "encoder"):
        return module.encoder(x)
    return module(x)


def _to_model_latent(model, x):
    if x.shape[1] in [4, 16]:
        return x.float()
    if hasattr(model, "enc") and model.enc is not None:
        return _encode_with_module(model.enc, x).float()
    # Pure pixel-space mode: no latent encoder is expected, so keep tensors as-is.
    return x.float()


def _to_teacher_latent(tmodel, x):
    if x.shape[1] in [4, 16]:
        return x.float()
    if tmodel is None:
        return x.float()
    if hasattr(tmodel, "encoder"):
        return _encode_with_module(tmodel.encoder, x).float()
    if hasattr(tmodel, "encode"):
        return _encode_with_module(tmodel, x).float()
    # 裸 encoder 模块（如 tiny_enc），直接 __call__
    return _encode_with_module(tmodel, x).float()


def _prepare_fmir_condition(model, x_lq):
    if x_lq is None:
        return None
    if x_lq.ndim != 4 or x_lq.shape[1] != 3:
        return None
    # 通过 model._build_fmir_condition 走完整路径（含 wavelet 频带融合）
    if hasattr(model, "_build_fmir_condition"):
        wavelet_cond = None
        if hasattr(model, "wavelet_stem") and model.wavelet_stem is not None:
            if getattr(model, "use_wavelet_fmir_cond", False) or getattr(model.wavelet_stem, "time_cond", False):
                t_dim = getattr(model, "t_emb_dim", 160)
                t_emb_init = pos_emb(torch.zeros(x_lq.shape[0]), t_dim).to(x_lq.device)
                wavelet_cond = model.wavelet_stem(x_lq, t_emb=t_emb_init)
        return model._build_fmir_condition(x_lq, wavelet_cond=wavelet_cond)
    if hasattr(model, "fmir") and hasattr(model.fmir, "make_condition"):
        return model.fmir.make_condition(x_lq)
    return None


def _prepare_decoder_condition(model, x_lq, spatial_cond):
    # split 模式：Decoder 拿纯 CNN 空间条件，不做 wavelet 融合
    if getattr(model, "use_wavelet_fmir_cond", False):
        return model._build_fmir_condition(x_lq, wavelet_cond=None)
    # 三流互补 decoder：fmir_cond 与 wavelet_cond 分别返回，不融合
    if hasattr(model, "dec") and model.dec is not None and hasattr(model.dec, "_use_encoder_skip_wavelet"):
        wavelet_cond = None
        if hasattr(model, "wavelet_stem") and model.wavelet_stem is not None and x_lq is not None:
            wavelet_cond = model.wavelet_stem(x_lq)
        return {"fmir_spatial": spatial_cond, "wavelet": wavelet_cond}

    if hasattr(model, "_build_decoder_condition"):
        return model._build_decoder_condition(x_lq, spatial_cond=spatial_cond)
    return spatial_cond


def _compute_lambda_pix(fm_cfg):
    lambda_max = float(fm_cfg.get("lambda_pix_max", 0.5))
    warmup_steps = int(fm_cfg.get("lambda_pix_warmup_steps", 10000))
    global_step = int(fm_cfg.get("global_step", warmup_steps))
    if warmup_steps <= 0:
        return lambda_max
    ratio = max(0.0, min(1.0, float(global_step) / float(warmup_steps)))
    return lambda_max * ratio


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def charbonnier_map(pred, target, eps=1e-3):
    """逐像素 Charbonnier，返回与输入同 shape 的张量（不 reduce）。"""
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps)


from contextlib import contextmanager

@contextmanager
def temporarily_freeze_module(module):
    """临时冻结模块参数：梯度可穿过但不更新参数。"""
    states = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, state in zip(module.parameters(), states):
            p.requires_grad_(state)


def _detach_dict(cond):
    if isinstance(cond, dict):
        return {k: (v.detach() if torch.is_tensor(v) else _detach_dict(v))
                for k, v in cond.items()}
    if torch.is_tensor(cond):
        return cond.detach()
    return cond


def _gaussian_blur(x, kernel_size=21, sigma=None):
    """对 [B,C,H,W] 张量做高斯模糊，用于低频颜色一致性损失。"""
    if sigma is None:
        sigma = kernel_size / 6.0
    # 构建 1D 高斯核
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    # 2D 可分离核
    kernel_2d = gauss_1d[:, None] @ gauss_1d[None, :]
    kernel = kernel_2d.expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])


def fm_loss(model, x_hq, x_lq, fm_cfg):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)

    with torch.no_grad():
        X_hq = _to_model_latent(model, x_hq)
        X_lq = _to_model_latent(model, x_lq)
        X_mmse = model.mmse(X_lq)
    cond = _prepare_fmir_condition(model, x_lq)

    b = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse + sigma_s * eps
    t = torch.rand([b, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    loss = F.mse_loss(u, v)
    return loss


def cfm_loss(model, x_hq, x_lq, fm_cfg):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)

    with torch.no_grad():
        X_hq = _to_model_latent(model, x_hq)
        X_lq = _to_model_latent(model, x_lq)
        X_mmse = model.mmse(X_lq)
    cond = _prepare_fmir_condition(model, x_lq)

    # 原 CFM: bridge 活跃时 beta=0 避免和 final_latent 冲突
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    # Flow Loss
    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends

    loss = F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_)

    return loss


def l2_fm_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    bs = x_hq.shape[0]

    # L2 loss
    with torch.no_grad():
        X_hq = _to_teacher_latent(tmodel, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow loss
    eps = torch.randn_like(X_hq)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    loss += F.mse_loss(u, v)
    return loss


def l2_fm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    beta = fm_cfg.get("beta", 0.001)

    # L2 loss
    with torch.no_grad():
        X_hq = _to_teacher_latent(tmodel, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow loss
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_hq)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    u = X_hq - (1 - sigma_min) * X_mmse_noisy
    v = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    loss += (1-beta)*F.mse_loss(u, v)

    # Keep MSE term in latent space to avoid decode during training.
    X1 = Xt + (1 - t) * v
    loss += beta*F.mse_loss(X_hq, X1)

    return loss


def l2_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)

    # L2 loss
    with torch.no_grad():
        X_hq = _to_teacher_latent(tmodel, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # 原 CFM: bridge 活跃时 beta=0 避免和 final_latent 冲突
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    # Flow Loss
    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends

    loss += (F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_))

    return loss


def l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)
    beta = fm_cfg.get("beta", 0.001)

    # L2 loss
    with torch.no_grad():
        X_hq = _to_model_latent(model, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    dec_cond = _prepare_decoder_condition(model, x_lq, cond)
    X_mmse = model.mmse(X_lq)
    loss = F.mse_loss(X_hq, X_mmse)

    # Flow Loss
    # 原 CFM: bridge 活跃时 beta=0 避免和 final_latent 冲突
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends
    loss += (1-beta)*(F.mse_loss(f0, f0_) + alpha * F.mse_loss(v0, v0_))

    # MSE loss
    f1 = f0.detach()
    v1 = model.fmir(f1, pos_emb(seg_ends, t_dim), cond=cond, t=seg_ends)
    X1 = f1 + (1 - seg_ends) * v1
    loss += beta*F.mse_loss(X_hq, X1)
    x_pred_pixel = _decode_with_module(model.dec, X1, cond_dict=dec_cond, x_lq=x_lq)
    loss += _compute_lambda_pix(fm_cfg) * F.mse_loss(x_pred_pixel, x_hq)

    return loss


def charbonnier_ssim_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel):
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)
    beta = fm_cfg.get("beta", 0.001)

    # L2 loss
    with torch.no_grad():
        X_hq = _to_model_latent(model, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    dec_cond = _prepare_decoder_condition(model, x_lq, cond)
    X_mmse = model.mmse(X_lq)
    loss = charbonnier_loss(X_hq, X_mmse)

    # Flow Loss
    # 原 CFM: bridge 活跃时 beta=0 避免和 final_latent 冲突
    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1-dt)*torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    # Split to segments
    segments = torch.linspace(0, 1, K+1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    # Move farward up to segment end line
    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less*(Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends
    loss += (1-beta)*(charbonnier_loss(f0, f0_) + alpha * charbonnier_loss(v0, v0_))

    # MSE loss
    f1 = f0.detach()
    v1 = model.fmir(f1, pos_emb(seg_ends, t_dim), cond=cond, t=seg_ends)
    X1 = f1 + (1 - seg_ends) * v1
    loss += beta*charbonnier_loss(X_hq, X1)
    x_pred_pixel = _decode_with_module(model.dec, X1, cond_dict=dec_cond, x_lq=x_lq)

    pred = torch.clamp(x_pred_pixel, 0.0, 1.0)
    target = torch.clamp(x_hq, 0.0, 1.0)
    pixel_loss = charbonnier_loss(pred, target) + 0.1 * (1.0 - ssim(pred, target, data_range=1.0))
    loss += _compute_lambda_pix(fm_cfg) * pixel_loss

    return loss


def get_loss(model, x_hq, x_lq, fm_cfg, tmodel=None):
    method = fm_cfg.get("method")
    if method == "fm_loss":
        loss = fm_loss(model, x_hq, x_lq, fm_cfg)
    elif method == "cfm_loss":
        loss = cfm_loss(model, x_hq, x_lq, fm_cfg)
    elif method == "l2_fm_loss":
        loss = l2_fm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_fm_mse_loss":
        loss = l2_fm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_loss":
        loss = l2_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_mse_loss":
        loss = l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "pixel_space_l2_cfm_loss":
        # Alias used by training yaml files.
        loss = l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "charbonnier_ssim_cfm_loss":
        loss = charbonnier_ssim_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_mse_loss_pixel_ablation":
        # Keep formula unchanged; this is a name alias for config compatibility.
        loss = l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    else:
        assert False, "Error: Unknown training method!"
    return loss


# ---------------------------------------------------------------------------
# 端到端 GAN 损失：FM 潜在空间流匹配 + 像素空间 GAN + 感知损失
# ---------------------------------------------------------------------------

def e2e_gan_loss(model, x_hq, x_lq, fm_cfg, discriminator, perceptual_fn, step,
                  dino_encoder=None, dino_spatial_proj=None, sk_fusion=None, tmodel=None,
                  global_step=0):
    """端到端训练损失：FMIR + Decoder 联合优化，含对抗和感知损失。

    Args:
        global_step: 真实训练步数（由 Lightning training_step 传入），用于 ODE warmup。
    Returns:
        (g_loss, d_loss): generator loss and discriminator loss.
        调用方负责分别 backward。
    """
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_s = fm_cfg.get("sigma_s", 0.1)  # 保留兼容
    train_noise_scale = float(fm_cfg.get("train_noise_scale", sigma_s))
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    alpha_cfm = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)
    beta_cfm = fm_cfg.get("beta", 0.001)

    # ---- 训练分布控制：low-t 采样 / noise / 辅助损失 ----
    t_sampling = str(fm_cfg.get("t_sampling", "uniform"))
    t_min = float(fm_cfg.get("t_min", 0.0))
    t_max = float(fm_cfg.get("t_max", 0.3))
    t_choices = list(fm_cfg.get("t_choices", [0.0, 0.05, 0.1, 0.2, 0.3]))
    lambda_fm = float(fm_cfg.get("lambda_fm", 1.0))
    lambda_mmse_char = float(fm_cfg.get("lambda_mmse_char", 0.0))
    lambda_low_t_endpoint = float(fm_cfg.get("lambda_low_t_endpoint", 0.0))
    low_t_endpoint_t = float(fm_cfg.get("low_t_endpoint_t", 0.0))
    lambda_mmse_guard = float(fm_cfg.get("lambda_mmse_guard", 0.0))
    lambda_calib_latent = float(fm_cfg.get("lambda_calib_latent", 0.0))
    lambda_flow_img = float(fm_cfg.get("lambda_flow_img", 0.0))
    lambda_ssim_calib = float(fm_cfg.get("lambda_ssim_calib", 0.05))
    calib_scale = float(fm_cfg.get("flow_infer_scale", 1.0))
    # K-step ODE 终点 pixel 监督 + 逐样本 improvement 约束
    lambda_ode_pixel = float(fm_cfg.get("lambda_ode_pixel", 0.0))
    lambda_ode_ssim = float(fm_cfg.get("lambda_ode_ssim", 0.0))
    lambda_flow_improve = float(fm_cfg.get("lambda_flow_improve", 0.0))
    lambda_latent_improve = float(fm_cfg.get("lambda_latent_improve", 0.0))
    flow_improve_margin = float(fm_cfg.get("flow_improve_margin", 0.0))
    latent_improve_margin = float(fm_cfg.get("latent_improve_margin", 0.0))
    use_latent_cond = bool(fm_cfg.get("use_latent_cond", False))

    # Loss weights (from config, with defaults)
    w_gan = float(fm_cfg.get("lambda_gan", 0.05))
    w_perc = float(fm_cfg.get("lambda_perc", 0.1))
    w_charb = float(fm_cfg.get("lambda_charb", 1.0))
    w_ssim = float(fm_cfg.get("lambda_ssim", 0.5))
    lambda_pix = _compute_lambda_pix(fm_cfg)

    # Warmup: gradually increase GAN and perceptual weights
    warmup_steps_gan = int(fm_cfg.get("gan_perc_warmup_steps", 5000))
    warmup_ratio = min(1.0, float(step) / max(1, warmup_steps_gan))
    w_gan = w_gan * warmup_ratio
    w_perc = w_perc * warmup_ratio

    # ---- 潜在空间 FM 损失 ----
    # HQ 潜编码必须用冻结的教师模型, 否则 Encoder 可训练时 ODE 终点会漂移
    with torch.no_grad():
        X_hq = _to_teacher_latent(tmodel, x_hq) if tmodel is not None else _to_model_latent(model, x_hq)
    X_lq = _to_model_latent(model, x_lq)
    cond = _prepare_fmir_condition(model, x_lq)
    dec_cond = _prepare_decoder_condition(model, x_lq, cond)
    X_mmse = model.mmse(X_lq)

    # ---- MMSE → HQ 对齐损失（仅优化 MMSE，不影响 FMIR） ----
    loss_mmse_char = charbonnier_loss(X_hq, X_mmse) if lambda_mmse_char > 0 else torch.tensor(0.0, device=X_lq.device)

    # ---- CFM velocity 损失（仅优化 FMIR，X_mmse 在 CFM 段内 detach） ----
    loss_fm = torch.tensor(0.0, device=X_lq.device)

    # ---- 潜在空间条件增强：让 FMIR 感知 latent 空间中 MMSE 的修正 ----
    # 将 z_lq 和 z_mmse - z_lq 注入 cond dict，LUNet 内部通过 zero-init
    # latent_cond_proj 融合，初始等价于无 latent cond 的标准行为。
    if use_latent_cond:
        if cond is None:
            cond = {}
        else:
            cond = dict(cond)
        cond["latent_lq"] = X_lq.detach()
        cond["latent_mmse_delta"] = (X_mmse - X_lq).detach()

    # ---- DINOv2 表示对齐：MMSE → 对齐目标 ----
    w_dino = float(fm_cfg.get("lambda_dino", 0.0))
    use_dino_perceptual = bool(fm_cfg.get("dino_perceptual", False))
    if w_dino > 0 and dino_encoder is not None:
        if use_dino_perceptual:
            # DINO 感知损失: decode MMSE → 像素 → DINO 空间比较
            # detach 条件防止 DINO 梯度更新 WaveletStem/condition_stem
            dec_cond_dino = _detach_dict(dec_cond) if dec_cond is not None else None
            img_mmse = _decode_with_module(model.dec, X_mmse, cond_dict=dec_cond_dino, x_lq=x_lq).clamp(0, 1)
            dino_hq = dino_encoder.encode_no_grad(x_hq)
            dino_mmse = dino_encoder.encode_with_grad(img_mmse)
            loss_dino = charbonnier_loss(dino_mmse, dino_hq.detach())

        elif model.dino_projector is not None:
            # 旧版: DINOv2 patch token 对齐 (潜空间)
            dino_hq = dino_encoder(x_hq)
            dino_mmse = model.dino_projector(X_mmse)
            loss_dino = F.mse_loss(dino_mmse, dino_hq.detach())
        else:
            loss_dino = torch.tensor(0.0, device=X_lq.device)
    else:
        loss_dino = torch.tensor(0.0, device=X_lq.device)

    bs = x_hq.shape[0]

    # =====================================================================
    # One-Step Residual 诊断训练模式
    # 验证 FMIR 能否在单步监督下学到 z_mmse → z_hq 的方向。
    # t 固定为 0，不跑 ODE，不加 noise，只用 residual + endpoint loss。
    # =====================================================================
    bridge_mode = str(fm_cfg.get("bridge_mode", "")).lower()

    if bridge_mode == "one_step_residual":
        lambda_one_step_residual = float(fm_cfg.get("lambda_one_step_residual", 1.0))
        lambda_one_step_endpoint = float(fm_cfg.get("lambda_one_step_endpoint", 1.0))
        bridge_detach_mmse = bool(fm_cfg.get("bridge_detach_mmse", True))
        bridge_detach_cond = bool(fm_cfg.get("bridge_detach_cond", True))
        one_step_t = float(fm_cfg.get("one_step_t", 0.0))
        one_step_use_cond = bool(fm_cfg.get("one_step_use_cond", True))

        X0 = X_mmse.detach() if bridge_detach_mmse else X_mmse
        X1 = X_hq  # z_hq 目标

        if one_step_use_cond:
            one_cond = _detach_dict(cond) if bridge_detach_cond else cond
        else:
            one_cond = None

        t0 = torch.full((bs,), one_step_t, device=X_lq.device, dtype=X0.dtype)
        t0_tensor = t0[:, None, None, None]
        t0_emb = pos_emb(t0, t_dim).to(X_lq.device)

        target_v = X1 - X0

        v_pred = model.fmir(
            X0,
            t0_emb,
            cond=one_cond,
            t=t0_tensor,
        )

        z1 = X0 + v_pred  # 单步预测的 refined latent

        loss_one_step_residual_raw = charbonnier_loss(v_pred, target_v)
        loss_one_step_endpoint_raw = charbonnier_loss(z1, X1)

        # 总 loss 仅由 one-step 两项构成（无 CFM / pixel / GAN）
        g_loss = (
            lambda_one_step_residual * loss_one_step_residual_raw
            + lambda_one_step_endpoint * loss_one_step_endpoint_raw
        )

        # ---- 诊断 metrics ----
        with torch.no_grad():
            charb_mmse = charbonnier_loss(X0.detach(), X1.detach())
            charb_z1 = charbonnier_loss(z1.detach(), X1.detach())
            v_pred_n = v_pred.detach().flatten(1).norm(dim=1).mean() / max(v_pred[0].numel(), 1) ** 0.5
            target_v_n = target_v.detach().flatten(1).norm(dim=1).mean() / max(target_v[0].numel(), 1) ** 0.5

            vp_flat = v_pred.detach()
            tv_flat = target_v.detach()

            one_step_metrics = {
                "one_step_residual_raw": loss_one_step_residual_raw.detach(),
                "one_step_residual_weighted": (lambda_one_step_residual * loss_one_step_residual_raw).detach(),
                "one_step_endpoint_raw": loss_one_step_endpoint_raw.detach(),
                "one_step_endpoint_weighted": (lambda_one_step_endpoint * loss_one_step_endpoint_raw).detach(),
                "one_step_charb_mmse_to_hq": charb_mmse,
                "one_step_charb_z1_to_hq": charb_z1,
                "one_step_gain_charb": charb_mmse - charb_z1,
                "one_step_v_pred_norm": v_pred_n,
                "one_step_target_v_norm": target_v_n,
                "one_step_norm_ratio": v_pred_n / (target_v_n + 1e-8),
                "one_step_delta_cosine": F.cosine_similarity(
                    v_pred.detach().flatten(1), target_v.detach().flatten(1), dim=1
                ).mean(),
                "one_step_z1_requires_grad": torch.tensor(float(z1.requires_grad), device=X0.device),
                "one_step_z1_grad_fn_exists": torch.tensor(float(z1.grad_fn is not None), device=X0.device),
                # 幅值统计
                "one_step_t_value": one_step_t,
                "one_step_use_cond_value": float(one_step_use_cond),
                "one_step_v_pred_mean": vp_flat.mean(),
                "one_step_v_pred_std": vp_flat.std(),
                "one_step_target_v_mean": tv_flat.mean(),
                "one_step_target_v_std": tv_flat.std(),
                "one_step_v_pred_absmean": vp_flat.abs().mean(),
                "one_step_target_v_absmean": tv_flat.abs().mean(),
                "one_step_v_pred_maxabs": vp_flat.abs().max(),
                "one_step_target_v_maxabs": tv_flat.abs().max(),
                # 总 loss 占比
                "total_g_loss_value": g_loss.detach(),
                "total_contains_final_latent_debug": torch.tensor(0.0, device=g_loss.device),
            }

        ret = dict(one_step_metrics)
        ret.update({
            "loss_fm": torch.tensor(0.0, device=X_lq.device),
            "loss_fm_weighted": torch.tensor(0.0, device=X_lq.device),
            "loss_mmse_char": torch.tensor(0.0, device=X_lq.device),
            "loss_mmse_char_weighted": torch.tensor(0.0, device=X_lq.device),
            "loss_charb": torch.tensor(0.0, device=X_lq.device),
            "loss_ssim": torch.tensor(0.0, device=X_lq.device),
            "loss_color": torch.tensor(0.0, device=X_lq.device),
            "loss_blur": torch.tensor(0.0, device=X_lq.device),
            "loss_dino": torch.tensor(0.0, device=X_lq.device),
            "loss_perc": torch.tensor(0.0, device=X_lq.device),
            "loss_gan_g": torch.tensor(0.0, device=X_lq.device),
            "loss_gan_d": torch.tensor(0.0, device=X_lq.device),
            "warmup_ratio": warmup_ratio,
        })

        loss_gan_d = torch.tensor(0.0, device=X_lq.device)
        return g_loss, loss_gan_d, ret

    # ---- Conditional Flow-based Latent Refinement Bridge ----
    # X0 = MMSE(LQ)  →  X1 = TAESD_enc(HQ)  →  FMIR 学 refinement 向量场
    lambda_bridge = float(fm_cfg.get("lambda_bridge", 0.0))
    lambda_final_latent = float(fm_cfg.get("lambda_final_latent", 0.0))

    loss_bridge = torch.tensor(0.0, device=X_lq.device)
    loss_final_latent = torch.tensor(0.0, device=X_lq.device)
    bridge_metrics = {}

    if lambda_bridge > 0 or lambda_final_latent > 0:
        bridge_sigma_s = float(fm_cfg.get("bridge_sigma_s", 0.0))
        bridge_random_k = fm_cfg.get("bridge_random_k", [3, 5])
        bridge_detach_mmse = bool(fm_cfg.get("bridge_detach_mmse", True))
        bridge_detach_cond = bool(fm_cfg.get("bridge_detach_cond", True))
        X0 = X_mmse.detach() if bridge_detach_mmse else X_mmse
        if bridge_sigma_s > 0:
            X0 = X0 + bridge_sigma_s * torch.randn_like(X0)
        X_bridge_1 = X_hq

        # Bridge 条件 (detach to avoid perturbing WaveletStem/MMSE)
        bridge_cond = _detach_dict(cond) if bridge_detach_cond else cond

        # Bridge loss: FMIR 直接学 X0 → X1 的向量场
        bt = torch.rand(bs, device=X_lq.device)  # t ~ U(0,1)
        Xt = (1 - bt[:, None, None, None]) * X0 + bt[:, None, None, None] * X_bridge_1
        v_target = X_bridge_1 - X0
        v_pred = model.fmir(Xt, pos_emb(bt, t_dim), cond=bridge_cond, t=bt[:, None, None, None])
        loss_bridge = charbonnier_loss(v_pred, v_target) if lambda_bridge > 0 else loss_bridge
        bridge_metrics["loss_bridge_raw"] = loss_bridge.detach()
        bridge_metrics["loss_bridge_weighted"] = (lambda_bridge * loss_bridge).detach()

        v_pred_n = v_pred.flatten(1).norm(dim=1).mean()
        v_target_n = v_target.flatten(1).norm(dim=1).mean()
        cos = F.cosine_similarity(v_pred.flatten(1), v_target.detach().flatten(1)).mean()

        bridge_metrics.update({
            "loss_bridge": loss_bridge.detach(),
            "v_pred_norm": v_pred_n.detach(),
            "v_target_norm": v_target_n.detach(),
            "norm_ratio": (v_pred_n / (v_target_n + 1e-8)).detach(),
            "bridge_cosine": cos.detach(),
        })

        # Final latent loss: Euler unroll from X0
        if lambda_final_latent > 0:
            K_bridge = random.choice(list(bridge_random_k))
            bridge_metrics["bridge_K"] = torch.tensor(float(K_bridge), device=X_lq.device).detach()
            _dt = 1.0 / K_bridge
            z_unroll = X0
            for _k in range(K_bridge):
                _tk = _k * _dt
                _t_vec = torch.full((bs,), _tk, device=X_lq.device, dtype=X0.dtype)
                _v = model.fmir(z_unroll, pos_emb(_t_vec, t_dim), cond=bridge_cond, t=_t_vec[:, None, None, None])
                z_unroll = z_unroll + _dt * _v
            loss_final_latent = charbonnier_loss(z_unroll, X_bridge_1)
            bridge_metrics["loss_final_latent_raw"] = loss_final_latent.detach()
            bridge_metrics["loss_final_latent_weighted"] = (lambda_final_latent * loss_final_latent).detach()

            # decode PSNR 对比 + latent 空间诊断
            with torch.no_grad():
                dec_mmse_test = _decode_with_module(model.dec, X_mmse, cond_dict=dec_cond, x_lq=x_lq).clamp(0, 1)
                dec_final_test = _decode_with_module(model.dec, z_unroll, cond_dict=dec_cond, x_lq=x_lq).clamp(0, 1)
                v_mmse = charbonnier_loss(dec_mmse_test, x_hq)  # rough proxy
                v_final = charbonnier_loss(dec_final_test, x_hq)
                bridge_metrics["pix_charb_mmse"] = v_mmse.detach()
                bridge_metrics["pix_charb_final"] = v_final.detach()
                bridge_metrics["pix_charb_gain"] = (v_mmse - v_final).detach()

                # ---- 训练期 latent 空间诊断 ----
                charb_mmse_to_hq = charbonnier_loss(X0.detach(), X_bridge_1.detach())
                charb_unroll_to_hq = charbonnier_loss(z_unroll.detach(), X_bridge_1.detach())
                bridge_metrics["train_latent_charb_mmse_to_hq"] = charb_mmse_to_hq.detach()
                bridge_metrics["train_latent_charb_unroll_to_hq"] = charb_unroll_to_hq.detach()
                bridge_metrics["train_latent_gain_charb"] = (charb_mmse_to_hq - charb_unroll_to_hq).detach()
                bridge_metrics["train_z_unroll_requires_grad"] = torch.tensor(
                    float(z_unroll.requires_grad), device=X0.device)
                bridge_metrics["train_z_unroll_grad_fn_exists"] = torch.tensor(
                    float(z_unroll.grad_fn is not None), device=X0.device)

                delta_fmir_train = z_unroll.detach() - X0.detach()
                delta_target_train = X_bridge_1.detach() - X0.detach()
                fmir_dn = delta_fmir_train.flatten(1).norm(dim=1).mean() / max(delta_fmir_train[0].numel(), 1) ** 0.5
                target_dn = delta_target_train.flatten(1).norm(dim=1).mean() / max(delta_target_train[0].numel(), 1) ** 0.5
                bridge_metrics["train_fmir_delta_norm"] = fmir_dn.detach()
                bridge_metrics["train_target_delta_norm"] = target_dn.detach()
                bridge_metrics["train_delta_norm_ratio"] = (fmir_dn / (target_dn + 1e-8)).detach()
                bridge_metrics["train_delta_cosine"] = F.cosine_similarity(
                    delta_fmir_train.flatten(1), delta_target_train.flatten(1), dim=1
                ).mean().detach()

    # 原 CFM: bridge 活跃时 beta=0 避免和 final_latent 冲突
    if lambda_final_latent > 0:
        beta_cfm = 0.0
    bs = x_hq.shape[0]
    if train_noise_scale > 0:
        eps = torch.randn_like(X_mmse)
        X_mmse_noisy = X_mmse.detach() + train_noise_scale * eps
    else:
        X_mmse_noisy = X_mmse.detach()

    # ---- Residual Target Amplification ----
    # 放大 z_hq - z_mmse residual 以增强 Flow Matching 监督信号。
    # z0 保持 z_mmse 不变，仅将 z1 从 z_hq 替换为 z_mmse + gamma*(z_hq - z_mmse)。
    # 推理/验证时用 residual_infer_scale (=1/gamma) 缩回。
    use_residual_amp = bool(fm_cfg.get("use_residual_amplification", False))
    residual_target_scale = float(fm_cfg.get("residual_target_scale", 1.0))
    residual_infer_scale = float(fm_cfg.get("residual_infer_scale", 1.0))

    if use_residual_amp:
        # noisy path 上的 amplified Z1
        Z1_cfm = X_mmse_noisy + residual_target_scale * (X_hq - X_mmse_noisy)
        # clean path 上的 amplified Z1（用于 low-t endpoint / guard）
        Z1_clean_amp = X_mmse.detach() + residual_target_scale * (X_hq - X_mmse.detach())
    else:
        Z1_cfm = X_hq
        Z1_clean_amp = X_hq

    # ---- t 采样策略 ----
    if t_sampling == "low_t":
        t = t_min + (t_max - t_min) * torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)
    elif t_sampling == "discrete_low_t":
        idx = torch.randint(0, len(t_choices), (bs,), device=X_hq.device)
        t_vals_tensor = torch.tensor(t_choices, device=X_hq.device, dtype=X_hq.dtype)
        t = t_vals_tensor[idx][:, None, None, None]
    else:  # uniform
        t = (1 - dt) * torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    segments = torch.linspace(0, 1, K + 1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * Z1_cfm

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * Z1_cfm
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    # ---- CFM 训练期 velocity 方向诊断 ----
    with torch.no_grad():
        target_v_raw = X_hq - X_mmse_noisy                    # 原始 residual
        target_v_amp = Z1_cfm - X_mmse_noisy                   # 放大后的 residual
        cfm_cos = F.cosine_similarity(
            v0.detach().flatten(1), target_v_amp.detach().flatten(1), dim=1
        ).mean()
        cfm_cos_raw = F.cosine_similarity(
            v0.detach().flatten(1), target_v_raw.detach().flatten(1), dim=1
        ).mean()
        # 幅值统计 (per-element RMS)
        _ne = max(v0[0].numel(), 1) ** 0.5
        cfm_target_v_norm_raw = target_v_raw.detach().flatten(1).norm(dim=1).mean() / _ne
        cfm_target_v_norm_amp = target_v_amp.detach().flatten(1).norm(dim=1).mean() / _ne
        cfm_v_pred_norm_raw = v0.detach().flatten(1).norm(dim=1).mean() / _ne
        cfm_v_pred_norm_infer = (residual_infer_scale * v0.detach()).flatten(1).norm(dim=1).mean() / _ne

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * Z1_cfm
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less * (Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends
    loss_fm += (1 - beta_cfm) * (charbonnier_loss(f0, f0_) + alpha_cfm * charbonnier_loss(v0, v0_))

    f1 = f0.detach()
    v1 = model.fmir(f1, pos_emb(seg_ends, t_dim), cond=cond, t=seg_ends)
    X1 = f1 + (1 - seg_ends) * v1
    loss_fm += beta_cfm * charbonnier_loss(Z1_cfm, X1)  # target uses Z1_cfm (amplified or original)

    # ---- 低 t endpoint 辅助 + PMRF MMSE fidelity guard ----
    # 两者共享 v_low 计算：在 t=low_t 处预测 velocity，避免重复 FMIR forward
    loss_low_t_endpoint = torch.tensor(0.0, device=X_lq.device)
    loss_mmse_guard = torch.tensor(0.0, device=X_lq.device)
    low_t_metrics = {}
    low_t_active = (lambda_low_t_endpoint > 0 or lambda_mmse_guard > 0
                    or lambda_calib_latent > 0 or lambda_flow_img > 0)

    if low_t_active:
        # 推理起点：clean X_mmse，与 forward() clean start 对齐
        X0_low = X_mmse.detach()
        t_low = torch.full((bs,), low_t_endpoint_t, device=X_lq.device, dtype=X_hq.dtype)
        t_low_tensor = t_low[:, None, None, None]
        t_low_emb = pos_emb(t_low, t_dim).to(X_lq.device)
        # CFM 插值: Xt = (1-(1-σ_min)*t)*X0 + t*Z1
        Xt_low = (1 - (1 - sigma_min) * t_low_tensor) * X0_low + t_low_tensor * Z1_clean_amp
        v_low = model.fmir(Xt_low, t_low_emb, cond=_detach_dict(cond), t=t_low_tensor)
        # 从 low_t 到 t=1 的终点预测（linear path: z1 = z_t + (1-t) * v）
        z_low_endpoint = Xt_low + (1 - t_low_tensor) * v_low

        # ---- low-t endpoint loss：约束 velocity → CFM path terminal target ----
        # CFM 参数化: Xt = (1-(1-σ_min)*t)*X0 + t*X_hq
        # t=0 endpoint: z_endpoint ≈ X0 + v ≈ X_hq + σ_min*X0  (NOT σ_min*Xt)
        if lambda_low_t_endpoint > 0:
            if use_residual_amp:
                z_amp_target = Z1_clean_amp + sigma_min * X0_low
                loss_low_t_endpoint = charbonnier_loss(z_low_endpoint, z_amp_target)
            else:
                z_terminal_target = X_hq.detach() + sigma_min * X0_low
                loss_low_t_endpoint = charbonnier_loss(z_low_endpoint, z_terminal_target)

        # ---- PMRF MMSE fidelity guard：z_mmse 是 posterior mean，不能被 FMIR 拉坏 ----
        if lambda_mmse_guard > 0:
            charb_mmse_to_hq = charbonnier_loss(X_mmse.detach(), X_hq.detach())
            if use_residual_amp:
                # 用缩回后的输出判断，不要直接用放大的 v_low
                v_low_infer = residual_infer_scale * v_low
                z_low_infer = X_mmse.detach() + v_low_infer
                charb_low_to_hq = charbonnier_loss(z_low_infer, X_hq.detach())
            else:
                charb_low_to_hq = charbonnier_loss(z_low_endpoint, X_hq.detach())
            # 仅当 FMIR 输出比 z_mmse 更差时产生 penalty
            loss_mmse_guard = F.relu(charb_low_to_hq - charb_mmse_to_hq)

        with torch.no_grad():
            target_v_low_amp = Z1_clean_amp - Xt_low       # amplified target velocity
            target_v_low_raw = X_hq - Xt_low                # original target velocity
            low_t_cos_amp = F.cosine_similarity(
                v_low.detach().flatten(1), target_v_low_amp.detach().flatten(1), dim=1
            ).mean()
            low_t_cos_raw = F.cosine_similarity(
                v_low.detach().flatten(1), target_v_low_raw.detach().flatten(1), dim=1
            ).mean()
            _ne_l = max(v_low[0].numel(), 1) ** 0.5
            low_t_v_pred_norm_raw = v_low.detach().flatten(1).norm(dim=1).mean() / _ne_l
            low_t_v_pred_norm_infer = (residual_infer_scale * v_low.detach()).flatten(1).norm(dim=1).mean() / _ne_l
            low_t_metrics = {
                "low_t_endpoint_loss_raw": loss_low_t_endpoint.detach(),
                "low_t_endpoint_loss_weighted": (lambda_low_t_endpoint * loss_low_t_endpoint).detach(),
                "low_t_cosine": low_t_cos_amp.detach(),
                "low_t_cosine_raw": low_t_cos_raw.detach(),
                "low_t_value": low_t_endpoint_t,
                "loss_mmse_guard_raw": loss_mmse_guard.detach(),
                "loss_mmse_guard_weighted": (lambda_mmse_guard * loss_mmse_guard).detach(),
                "low_t_v_pred_norm_raw": low_t_v_pred_norm_raw.detach(),
                "low_t_v_pred_norm_infer": low_t_v_pred_norm_infer.detach(),
            }

    # ---- Calibrated one-step training loss ----
    # z_calib = z_mmse + calib_scale * v_low
    # loss_calib_latent = CharB(z_calib, z_hq)
    # loss_flow_img = CharB(decoder(z_calib), x_hq) + ssim_weight * (1 - SSIM)
    # 梯度链: loss → decoder(z_calib) → z_calib → v_low → FMIR
    loss_calib_latent = torch.tensor(0.0, device=X_lq.device)
    loss_flow_img = torch.tensor(0.0, device=X_lq.device)
    calib_metrics = {}
    if (lambda_calib_latent > 0 or lambda_flow_img > 0) and low_t_active:
        z_calib = X_mmse.detach() + calib_scale * v_low
        if lambda_calib_latent > 0:
            loss_calib_latent = charbonnier_loss(z_calib, X_hq)
        if lambda_flow_img > 0:
            dec_cond_calib = _detach_dict(dec_cond)
            pred_calib = _decode_with_module(model.dec, z_calib, cond_dict=dec_cond_calib, x_lq=x_lq)
            pred_calib_c = torch.clamp(pred_calib, 0.0, 1.0)
            target_calib_c = torch.clamp(x_hq, 0.0, 1.0)
            loss_flow_img = charbonnier_loss(pred_calib_c, target_calib_c)
            if lambda_ssim_calib > 0:
                loss_flow_img = loss_flow_img + lambda_ssim_calib * (1.0 - ssim(pred_calib_c, target_calib_c, data_range=1.0))
        with torch.no_grad():
            calib_metrics["loss_calib_latent_raw"] = loss_calib_latent.detach()
            calib_metrics["loss_calib_latent_weighted"] = (lambda_calib_latent * loss_calib_latent).detach()
            calib_metrics["loss_flow_img_raw"] = loss_flow_img.detach()
            calib_metrics["loss_flow_img_weighted"] = (lambda_flow_img * loss_flow_img).detach()
            calib_metrics["calib_z_requires_grad"] = torch.tensor(
                float(z_calib.requires_grad), device=X_lq.device)
            calib_metrics["calib_scale_value"] = torch.tensor(calib_scale, device=X_lq.device)

    # ---- 像素空间解码（仅在 pixel loss 需要时执行） ----
    pixel_loss_active = (lambda_pix > 0 and (w_charb > 0 or w_ssim > 0 or w_perc > 0 or w_color > 0 or w_blur > 0))
    if pixel_loss_active or (discriminator is not None):
        if use_residual_amp and pixel_loss_active:
            X1_decode = X_mmse_noisy + residual_infer_scale * (X1 - X_mmse_noisy)
            x_pred_pixel = _decode_with_module(model.dec, X1_decode, cond_dict=dec_cond, x_lq=x_lq)
        else:
            x_pred_pixel = _decode_with_module(model.dec, X1, cond_dict=dec_cond, x_lq=x_lq)
        pred = torch.clamp(x_pred_pixel, 0.0, 1.0)
        target = torch.clamp(x_hq, 0.0, 1.0)
    else:
        pred = torch.zeros(1, 3, 64, 64, device=X_lq.device)  # dummy 4D
        target = torch.zeros(1, 3, 64, 64, device=X_lq.device)  # dummy 4D

    # ---- 像素损失 ----
    if pixel_loss_active:
        loss_charb = charbonnier_loss(pred, target)
        loss_ssim = 1.0 - ssim(pred, target, data_range=1.0)
    else:
        loss_charb = torch.tensor(0.0, device=X_lq.device)
        loss_ssim = torch.tensor(0.0, device=X_lq.device)

    # ---- 余弦颜色损失（逐像素 RGB 向量夹角，惩罚偏色） ----
    w_color = float(fm_cfg.get("lambda_color", 0.0))
    if w_color > 0:
        cos_sim = F.cosine_similarity(pred, target, dim=1)  # [B, H, W]
        loss_color = (1.0 - cos_sim).mean()
    else:
        loss_color = torch.tensor(0.0, device=pred.device)

    # ---- 模糊颜色一致性损失（惩罚全局亮度/色彩偏移） ----
    w_blur = float(fm_cfg.get("lambda_blur", 0.0))
    if w_blur > 0:
        pred_blur = _gaussian_blur(pred, kernel_size=21)
        target_blur = _gaussian_blur(target, kernel_size=21)
        loss_blur = charbonnier_loss(pred_blur, target_blur)
    else:
        loss_blur = torch.tensor(0.0, device=pred.device)

    # ---- 感知损失 ----
    if perceptual_fn is not None:
        loss_perc = perceptual_fn(pred, target)
    else:
        loss_perc = torch.tensor(0.0, device=pred.device)

    # ---- 对抗损失 ----
    # 判别器始终前向计算（保证 d_loss 有梯度图），G 的对抗项仅在 warmup 后生效。
    # GAN 损失裁剪防止数值爆炸。
    if discriminator is not None:
        logits_fake = discriminator(pred)
        loss_gan_g = torch.clamp(hinge_g_loss(logits_fake), min=-10.0, max=10.0)

        with torch.no_grad():
            logits_fake_d = discriminator(pred.detach())
        logits_real = discriminator(target)
        loss_gan_d = hinge_d_loss(logits_real, logits_fake_d)
    else:
        loss_gan_g = torch.tensor(0.0, device=pred.device)
        loss_gan_d = torch.tensor(0.0, device=pred.device)

    # ---- 像素空间总损失（聚合字段，供 CSV 记录） ----
    loss_pixel_total = lambda_pix * (
        w_charb * loss_charb + w_ssim * loss_ssim + w_perc * loss_perc
        + w_color * loss_color + w_blur * loss_blur
    )

    # ---- K-step ODE 训练分支（使用共享 run_fmir_ode） ----
    ode_metrics = {}
    loss_ode_pixel_charb = torch.tensor(0.0, device=X_lq.device)
    loss_ode_pixel_ssim_val = torch.tensor(0.0, device=X_lq.device)
    loss_flow_improve_val = torch.tensor(0.0, device=X_lq.device)
    loss_latent_improve_val = torch.tensor(0.0, device=X_lq.device)

    ode_active = (lambda_ode_pixel > 0 or lambda_ode_ssim > 0 or
                  lambda_flow_improve > 0 or lambda_latent_improve > 0)

    # ODE loss warmup：初期 MMSE/Decoder 未稳，不强加 ODE 约束
    ode_loss_start_step = int(fm_cfg.get("ode_loss_start_step", 5000))
    ode_loss_warmup_steps = int(fm_cfg.get("ode_loss_warmup_steps", 10000))
    gs = int(global_step)
    if gs < ode_loss_start_step:
        ode_ramp = 0.0
    else:
        ode_ramp = min(1.0, (gs - ode_loss_start_step) / max(ode_loss_warmup_steps, 1))

    if ode_active and ode_ramp > 0:
        # 使用与正式推理一致的 ODE condition（detach_wavelet 防止 ODE loss 更新 WaveletStem）
        ode_cond = model.prepare_fmir_ode_condition(x_lq, X_lq.detach(), X_mmse.detach(),
                                                     detach_wavelet=True)
        z_ode = model.run_fmir_ode(X_mmse.detach(), ode_cond, track_grad=True)

        # Decoder condition detach
        dec_cond_ode = _detach_dict(dec_cond) if dec_cond is not None else None

        need_pixel = (lambda_ode_pixel > 0 or lambda_ode_ssim > 0 or lambda_flow_improve > 0)
        pred_ode_c = None  # 确保变量在 if 外可见
        if need_pixel:
            with temporarily_freeze_module(model.dec):
                pred_ode = _decode_with_module(model.dec, z_ode, cond_dict=dec_cond_ode, x_lq=x_lq)
            assert z_ode.requires_grad, "z_ode must require grad for ODE auxiliary loss"
            assert pred_ode.requires_grad, "pred_ode must require grad (Decoder not in no_grad)"
            pred_ode_c = torch.clamp(pred_ode, 0.0, 1.0)
            target_c = torch.clamp(x_hq, 0.0, 1.0)
        if lambda_ode_pixel > 0 and pred_ode_c is not None:
            loss_ode_pixel_charb = charbonnier_loss(pred_ode_c, target_c)
        if lambda_ode_ssim > 0 and pred_ode_c is not None:
            loss_ode_pixel_ssim_val = 1.0 - ssim(pred_ode_c, target_c, data_range=1.0)

        # ---- 逐样本 improvement 约束 ----
        # pixel improvement：需要 pred_ode_c（来自 pixel decode 分支）
        if lambda_flow_improve > 0 and pred_ode_c is not None:
            with torch.no_grad():
                pred_mmse_ref = _decode_with_module(
                    model.dec, X_mmse.detach(), cond_dict=dec_cond_ode, x_lq=x_lq).clamp(0, 1)
            err_mmse_pix = charbonnier_map(pred_mmse_ref, x_hq).flatten(1).mean(1).detach()
            err_ode_pix = charbonnier_map(pred_ode_c, x_hq).flatten(1).mean(1)
            loss_flow_improve_val = torch.relu(
                err_ode_pix - err_mmse_pix + flow_improve_margin).mean()

        # latent improvement：不依赖 pixel decode，独立计算
        if lambda_latent_improve > 0:
            err_mmse_latent = charbonnier_map(
                X_mmse.detach(), X_hq.detach()).flatten(1).mean(1).detach()
            err_ode_latent = charbonnier_map(z_ode, X_hq.detach()).flatten(1).mean(1)
            loss_latent_improve_val = torch.relu(
                err_ode_latent - err_mmse_latent + latent_improve_margin).mean()

        with torch.no_grad():
            ode_metrics["loss_ode_pixel_charb_raw"] = (
                loss_ode_pixel_charb.detach() if isinstance(loss_ode_pixel_charb, torch.Tensor) else 0.0)
            ode_metrics["loss_ode_pixel_ssim_raw"] = (
                loss_ode_pixel_ssim_val.detach() if isinstance(loss_ode_pixel_ssim_val, torch.Tensor) else 0.0)
            ode_metrics["loss_flow_improve_raw"] = (
                loss_flow_improve_val.detach() if isinstance(loss_flow_improve_val, torch.Tensor) else 0.0)
            ode_metrics["loss_latent_improve_raw"] = (
                loss_latent_improve_val.detach() if isinstance(loss_latent_improve_val, torch.Tensor) else 0.0)
    if ode_active:
        ode_metrics["ode_loss_ramp"] = torch.tensor(ode_ramp, device=X_lq.device)
    else:
        ode_metrics["ode_loss_ramp"] = torch.tensor(0.0, device=X_lq.device)

    # ---- 一次性 ODE 辅助 loss 梯度审计（代表参数，不遍历全量） ----
    if ode_active and ode_ramp > 0 and not getattr(model, '_ode_audit_done', False):
        # 审计用 loss 不乘 ramp，避免首次生效时梯度太小被误判为 0
        ode_aux_audit_loss = (
            lambda_ode_pixel * loss_ode_pixel_charb
            + lambda_ode_ssim * loss_ode_pixel_ssim_val
            + lambda_flow_improve * loss_flow_improve_val
            + lambda_latent_improve * loss_latent_improve_val
        )

        def _first_trainable_param(module, name_filter=None):
            if module is None:
                return None, None
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if name_filter is None or name_filter(n):
                    return n, p
            return None, None

        audit_names = []
        audit_params = []

        def _add(label, param):
            if param is not None and param.requires_grad:
                audit_names.append(label)
                audit_params.append(param)

        # FMIR velocity head
        _add("FMIR.final_proj.weight", model.fmir.final_proj.weight)

        # FMIR condition 代表参数
        cn, cp = _first_trainable_param(model.fmir, lambda n: "cond" in n.lower())
        _add(f"FMIR.condition:{cn}", cp)

        # WaveletStem 代表参数（预期 None）
        wn, wp = _first_trainable_param(model.wavelet_stem) if model.wavelet_stem else (None, None)
        _add(f"WaveletStem:{wn}", wp)

        # Decoder 代表参数（预期 None，因 temporary freeze）
        dn, dp = _first_trainable_param(model.dec) if hasattr(model, 'dec') and model.dec else (None, None)
        _add(f"Decoder:{dn}", dp)

        # MMSE 代表参数（预期 None）
        mn, mp = _first_trainable_param(model.mmse)
        _add(f"MMSE:{mn}", mp)

        audit_grads = torch.autograd.grad(
            ode_aux_audit_loss, audit_params,
            retain_graph=True, allow_unused=True)

        model._ode_audit_done = True

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"[ODE auxiliary gradient audit]")
            for name, grad in zip(audit_names, audit_grads):
                if grad is None:
                    print(f"  {name}: None")
                else:
                    print(f"  {name}: {grad.detach().float().norm().item():.8e}")
            print(f"  Encoder: frozen by configuration")

    # ---- 总损失 ----
    g_loss = (
        lambda_fm * loss_fm
        + lambda_mmse_char * loss_mmse_char
        + lambda_low_t_endpoint * loss_low_t_endpoint
        + lambda_mmse_guard * loss_mmse_guard
        + lambda_bridge * loss_bridge
        + lambda_final_latent * loss_final_latent
        + w_dino * loss_dino
        + lambda_calib_latent * loss_calib_latent
        + lambda_flow_img * loss_flow_img
        + loss_pixel_total
        + w_gan * loss_gan_g
        + ode_ramp * lambda_ode_pixel * loss_ode_pixel_charb
        + ode_ramp * lambda_ode_ssim * loss_ode_pixel_ssim_val
        + ode_ramp * lambda_flow_improve * loss_flow_improve_val
        + ode_ramp * lambda_latent_improve * loss_latent_improve_val
    )

    # ---- 诊断：总 loss 中 final_latent 占比 ----
    ret = dict(bridge_metrics)
    ret.update(low_t_metrics)
    ret.update(calib_metrics)
    ret.update(ode_metrics)
    ret["total_g_loss_value"] = g_loss.detach()
    ret["cfm_cosine"] = cfm_cos.detach()
    ret["cfm_cosine_raw"] = cfm_cos_raw.detach()
    # ---- residual amplification 诊断 ----
    ret["residual_target_scale_value"] = torch.tensor(residual_target_scale, device=g_loss.device)
    ret["residual_infer_scale_value"] = torch.tensor(residual_infer_scale, device=g_loss.device)
    ret["cfm_target_v_norm_raw"] = cfm_target_v_norm_raw.detach()
    ret["cfm_target_v_norm_amp"] = cfm_target_v_norm_amp.detach()
    ret["cfm_v_pred_norm_raw"] = cfm_v_pred_norm_raw.detach()
    ret["cfm_v_pred_norm_infer"] = cfm_v_pred_norm_infer.detach()
    if lambda_final_latent > 0:
        weighted_fl = (lambda_final_latent * loss_final_latent).detach()
        ret["total_contains_final_latent_debug"] = weighted_fl / (g_loss.detach().abs() + 1e-8)
    else:
        ret["total_contains_final_latent_debug"] = torch.tensor(0.0, device=g_loss.device)
    ret.update({
        "loss_fm": loss_fm.detach(),                           # CFM velocity + consistency（纯 FMIR）
        "loss_fm_weighted": (lambda_fm * loss_fm).detach(),
        "loss_mmse_char": loss_mmse_char.detach(),              # MMSE → HQ Char（纯 MMSE）
        "loss_mmse_char_weighted": (lambda_mmse_char * loss_mmse_char).detach(),
        "loss_charb": loss_charb.detach(),
        "loss_ssim": loss_ssim.detach(),
        "loss_color": loss_color.detach() if isinstance(loss_color, torch.Tensor) else loss_color,
        "loss_blur": loss_blur.detach() if isinstance(loss_blur, torch.Tensor) else loss_blur,
        "loss_dino": loss_dino.detach() if isinstance(loss_dino, torch.Tensor) else loss_dino,
        "loss_perc": loss_perc.detach() if isinstance(loss_perc, torch.Tensor) else loss_perc,
        "loss_pixel_total": loss_pixel_total.detach() if isinstance(loss_pixel_total, torch.Tensor) else loss_pixel_total,
        "loss_gan_g": loss_gan_g.detach() if isinstance(loss_gan_g, torch.Tensor) else loss_gan_g,
        "loss_gan_d": loss_gan_d.detach() if isinstance(loss_gan_d, torch.Tensor) else loss_gan_d,
        "warmup_ratio": warmup_ratio,
    })
    return g_loss, loss_gan_d, ret
