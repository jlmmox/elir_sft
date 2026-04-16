import torch
import torch.nn.functional as F
import math
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


def _decode_with_module(module, z):
    if module is None:
        return z
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
    if hasattr(model, "fmir") and hasattr(model.fmir, "make_condition"):
        return model.fmir.make_condition(x_lq)
    return None


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
    x_pred_pixel = _decode_with_module(model.dec, X1)
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
    x_pred_pixel = _decode_with_module(model.dec, X1)

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
    elif method == "charbonnier_ssim_cfm_loss":
        loss = charbonnier_ssim_cfm_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    elif method == "l2_cfm_mse_loss_pixel_ablation":
        # Keep formula unchanged; this is a name alias for config compatibility.
        loss = l2_cfm_mse_loss(model, x_hq, x_lq, fm_cfg, tmodel)
    else:
        assert False, "Error: Unknown training method!"
    return loss
