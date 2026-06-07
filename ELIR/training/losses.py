import torch
import torch.nn.functional as F
import math

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
    return x.float()


def _prepare_fmir_condition(model, x_lq):
    if x_lq is None:
        return None
    if x_lq.ndim != 4 or x_lq.shape[1] != 3:
        return None
    # 通过 model._build_fmir_condition 走完整路径（含 wavelet 频带融合）
    if hasattr(model, "_build_fmir_condition"):
        wavelet_cond = None
        if hasattr(model, "wavelet_stem") and model.wavelet_stem is not None:
            if getattr(model, "use_wavelet_fmir_cond", False):
                wavelet_cond = model.wavelet_stem(x_lq)
        return model._build_fmir_condition(x_lq, wavelet_cond=wavelet_cond)
    if hasattr(model, "fmir") and hasattr(model.fmir, "make_condition"):
        return model.fmir.make_condition(x_lq)
    return None


def _prepare_decoder_condition(model, x_lq, spatial_cond):
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

def e2e_gan_loss(model, x_hq, x_lq, fm_cfg, discriminator, perceptual_fn, step, dino_encoder=None, tmodel=None):
    """端到端训练损失：FMIR + Decoder 联合优化，含对抗和感知损失。

    Returns:
        (g_loss, d_loss): generator loss and discriminator loss.
        调用方负责分别 backward。
    """
    t_dim = fm_cfg.get("t_emb_dim", 160)
    sigma_min = fm_cfg.get("sigma_min", 1e-5)
    sigma_s = fm_cfg.get("sigma_s", 0.1)
    alpha_cfm = fm_cfg.get("alpha", 0.001)
    K = fm_cfg.get("k_steps")
    dt = fm_cfg.get("dt", 0.05)
    beta_cfm = fm_cfg.get("beta", 0.001)

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
    loss_fm = charbonnier_loss(X_hq, X_mmse)

    # ---- DINOv2 表示对齐：MMSE latent → DINOv2(HQ) patch tokens ----
    w_dino = float(fm_cfg.get("lambda_dino", 0.0))
    if w_dino > 0 and dino_encoder is not None and model.dino_projector is not None:
        dino_hq = dino_encoder(x_hq)                              # [B, 256, 768]
        dino_mmse = model.dino_projector(X_mmse)                   # [B, 256, 768]
        loss_dino = F.mse_loss(dino_mmse, dino_hq.detach())
    else:
        loss_dino = torch.tensor(0.0, device=X_lq.device)

    bs = x_hq.shape[0]
    eps = torch.randn_like(X_mmse)
    X_mmse_noisy = X_mmse.detach() + sigma_s * eps
    t = (1 - dt) * torch.rand([bs, 1, 1, 1], device=X_hq.device, dtype=X_hq.dtype)

    segments = torch.linspace(0, 1, K + 1, device=X_hq.device, dtype=X_hq.dtype)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
    seg_ends = segments[seg_indices]
    X_ends = (1 - (1 - sigma_min) * seg_ends) * X_mmse_noisy + seg_ends * X_hq

    Xt = (1 - (1 - sigma_min) * t) * X_mmse_noisy + t * X_hq
    v0 = model.fmir(Xt, pos_emb(t, t_dim), cond=cond, t=t)

    with torch.no_grad():
        r = t + dt
        Xr = (1 - (1 - sigma_min) * r) * X_mmse_noisy + r * X_hq
        v0_ = model.fmir(Xr, pos_emb(r, t_dim), cond=cond, t=r)

    f0 = Xt + (seg_ends - t) * v0
    r_less = r < seg_ends
    f0_ = r_less * (Xr + (seg_ends - r) * v0_) + (~r_less) * X_ends
    loss_fm += (1 - beta_cfm) * (charbonnier_loss(f0, f0_) + alpha_cfm * charbonnier_loss(v0, v0_))

    f1 = f0.detach()
    v1 = model.fmir(f1, pos_emb(seg_ends, t_dim), cond=cond, t=seg_ends)
    X1 = f1 + (1 - seg_ends) * v1
    loss_fm += beta_cfm * charbonnier_loss(X_hq, X1)

    # ---- 像素空间解码 ----
    x_pred_pixel = _decode_with_module(model.dec, X1, cond_dict=dec_cond, x_lq=x_lq)
    pred = torch.clamp(x_pred_pixel, 0.0, 1.0)
    target = torch.clamp(x_hq, 0.0, 1.0)

    # ---- 像素损失 ----
    loss_charb = charbonnier_loss(pred, target)
    loss_ssim = 1.0 - ssim(pred, target, data_range=1.0)

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
        loss_color = torch.tensor(0.0, device=pred.device)

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

    # ---- 总损失 ----
    g_loss = (
        loss_fm
        + w_dino * loss_dino
        + lambda_pix * (w_charb * loss_charb + w_ssim * loss_ssim + w_perc * loss_perc + w_color * loss_color + w_blur * loss_blur)
        + w_gan * loss_gan_g
    )

    return g_loss, loss_gan_d, {
        "loss_fm": loss_fm.detach(),
        "loss_charb": loss_charb.detach(),
        "loss_ssim": loss_ssim.detach(),
        "loss_color": loss_color.detach() if isinstance(loss_color, torch.Tensor) else loss_color,
        "loss_blur": loss_blur.detach() if isinstance(loss_blur, torch.Tensor) else loss_blur,
        "loss_dino": loss_dino.detach() if isinstance(loss_dino, torch.Tensor) else loss_dino,
        "loss_perc": loss_perc.detach() if isinstance(loss_perc, torch.Tensor) else loss_perc,
        "loss_gan_g": loss_gan_g.detach() if isinstance(loss_gan_g, torch.Tensor) else loss_gan_g,
        "loss_gan_d": loss_gan_d.detach() if isinstance(loss_gan_d, torch.Tensor) else loss_gan_d,
        "warmup_ratio": warmup_ratio,
    }
