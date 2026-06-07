import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AttnBlock(nn.Module):
    """瓶颈自注意力：低分辨率下全局交互，开销极小。"""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))                 # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)                 # each [B, C, H, W]

        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # [B, h, N, d]
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                      # [B, h, N, d]

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out) + x                          # 残差连接





def haar_iwt(ll, lh, hl, hh):
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, C, H2, W2 = ll.shape
    out = torch.zeros(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


class DwtSkipEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hf_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.SiLU(),
        )
        nn.init.zeros_(self.hf_conv[0].weight)
        nn.init.zeros_(self.hf_conv[0].bias)
        from ELIR.models.wavelet_stem import HaarDWT2D
        self.dwt = HaarDWT2D(in_channels=channels)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        # 残差: zero-init 时 hf_conv 输出 0, 子带原样通过, IWT 还原原始特征
        lh = lh + self.hf_conv(lh)
        hl = hl + self.hf_conv(hl)
        hh = hh + self.hf_conv(hh)
        return haar_iwt(ll, lh, hl, hh)

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_emb_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, time_emb_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_emb_dim, time_emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        out = self.linear2(x)
        return out


class Upsample(nn.Module):
    def __init__(self,  in_channels, out_channels, use_convtr):
        super().__init__()
        self.use_convtr = use_convtr
        if self.use_convtr:
            self.convtr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.use_convtr:
            x = self.convtr(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.avgpool(self.conv(x))


class TimeGatedDilatedConv(nn.Module):
    """时间驱动的通道级门控空洞卷积。

    t_emb → MLP → C 维 Sigmoid 门控 → 控制大视野分支的每个通道参与量。
    零初始化 (bias=-5): 训练第 0 步 gate≈0.007≈0, 等价于标准 ResBlock。
    """

    def __init__(self, in_channels, out_channels, time_dim, dilation=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        # 零初始化：gate ≈ 0, 退化到标准 ResBlock
        nn.init.zeros_(self.gate_mlp[1].weight)
        nn.init.constant_(self.gate_mlp[1].bias, -5.0)

    def forward(self, x, t_emb):
        out = self.conv(x)
        gate = torch.sigmoid(self.gate_mlp(t_emb))  # [B, C, 1, 1]
        self._last_gate = gate.detach().mean().item()
        return out * gate.view(-1, out.shape[1], 1, 1)


class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=32, overparametrization=False):
        super().__init__()
        if overparametrization:
            self.block = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, 4*out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.Conv2d(4*out_channels, out_channels, kernel_size=1, stride=1, padding="same")
            )
        else:
            self.block = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
            )

    def forward(self, x):
        return self.block(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, overparametrization=False,
                 use_time_dilate=False):
        super().__init__()
        self.use_time_dilate = bool(use_time_dilate)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = Block2D(in_channels, out_channels, overparametrization=overparametrization)
        self.block2 = Block2D(out_channels, out_channels, overparametrization=overparametrization)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if self.use_time_dilate:
            self.dilated_conv = TimeGatedDilatedConv(out_channels, out_channels, time_dim, dilation=3)
            self._last_gate = None

    def forward(self, x, emb):
        h = self.block1(x)
        h += self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        if self.use_time_dilate:
            h = h + self.dilated_conv(h, emb)
            self._last_gate = self.dilated_conv._last_gate
        out = h + self.conv2d(x)
        return out


class ConditionStem(nn.Module):
    def __init__(self, in_channels=3, cond_base_channels=16):
        super().__init__()
        # 原生 256x256 条件提取，不再进行 PixelUnshuffle。
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, cond_base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_base_channels, cond_base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(x)


class TanhSFT(nn.Module):
    def __init__(self, feat_channels, cond_channels, gamma_scale=0.2, beta_scale=0.2):
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, feat_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )
        self.gamma_head = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)
        self.beta_head = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, feat, cond, w=1.0):
        if cond is None:
            return feat
        if cond.shape[-2:] != feat.shape[-2:]:
            # Align condition map to feature resolution to support variable-size inputs.
            cond = F.interpolate(cond, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        h = self.shared(cond)
        gamma = torch.tanh(self.gamma_head(h)) * self.gamma_scale
        beta = torch.tanh(self.beta_head(h)) * self.beta_scale
        gamma = gamma.to(device=feat.device, dtype=feat.dtype)
        beta = beta.to(device=feat.device, dtype=feat.dtype)

        if not torch.is_tensor(w):
            w = torch.tensor(float(w), device=feat.device, dtype=feat.dtype).view(1, 1, 1, 1)
        w = w.to(device=feat.device, dtype=feat.dtype)
        if w.ndim == 0:
            w = w.view(1, 1, 1, 1)
        if w.ndim == 1:
            w = w.view(-1, 1, 1, 1)

        gamma = gamma * w
        beta = beta * w
        return feat * (1.0 + gamma) + beta


class LUnet(nn.Module):
    def __init__(self, ch_mult=[1,2,1,2], n_mid_blocks=3, in_channels=16, hid_channels=128,
                 out_channels=16, t_emb_dim=160, use_rescale_conv=True, overparametrization=False,
                 use_cgfm=True, cond_in_channels=3, cond_downscale_factor=4,
                 cond_base_channels=16, cond_pyramid_channels=(16, 32, 64),
                 gate_w_min=0.1, gate_p=1.5, sft_gamma_scale=0.2, sft_beta_scale=0.2,
                 use_checkpoint=False, use_attn=False, attn_heads=8,
                 use_time_dilate=False, use_dwt_skip=False):
        super(LUnet, self).__init__()
        self.overparametrization = overparametrization
        self.t_emb_dim = t_emb_dim
        self.use_dwt_skip = bool(use_dwt_skip)
        self.use_cgfm = bool(use_cgfm)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_attn = bool(use_attn)
        self.use_time_dilate = bool(use_time_dilate)
        self.gate_w_min = float(gate_w_min)
        self.gate_p = float(gate_p)
        _ = cond_downscale_factor  # 保留参数兼容旧配置，当前原生尺寸方案不使用该参数。
        time_dim_out = 4*t_emb_dim
        self.time_mlp = TimestepEmbedding(in_channels=t_emb_dim, time_emb_dim=time_dim_out)
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self._down_feat_channels = []
        self._up_feat_channels = []

        self.first_proj = nn.Conv2d(in_channels, hid_channels, kernel_size=1)

        # Down blocks
        chs = hid_channels
        for mult in ch_mult:
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization, use_time_dilate=self.use_time_dilate)
            self._down_feat_channels.append(chs)
            if mult!=1:
                downsample = Downsample(chs, mult * chs, use_conv=use_rescale_conv)
                chs = mult * chs
            else:
                downsample = nn.Identity()
            self.down_blocks.append(nn.ModuleList([resnet, downsample]))

        self._mid_feat_channels = chs

        # Mid blocks
        for i in range(n_mid_blocks):
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization, use_time_dilate=self.use_time_dilate)
            self.mid_blocks.append(resnet)

        # Bottleneck attention (插在 mid blocks 之后, 低分辨率全局交互)
        self.mid_attn = None
        if self.use_attn:
            self.mid_attn = AttnBlock(chs, num_heads=attn_heads)

        # Up blocks
        for mult in ch_mult[::-1]:
            if mult!=1:
                upsample = Upsample(chs, chs//mult, use_convtr=use_rescale_conv)
                chs = chs // mult
            else:
                upsample = nn.Identity()
            resnet = ResnetBlock2D(2*chs, chs, time_dim_out, overparametrization=overparametrization, use_time_dilate=self.use_time_dilate)
            self._up_feat_channels.append(chs)
            self.up_blocks.append(nn.ModuleList([upsample, resnet]))

        # DWT Skip: 浅层 skip 频带增强 (zero-init, 等价标准 skip)
        self.dwt_skip_layers = None
        if self.use_dwt_skip:
            self.dwt_skip_layers = nn.ModuleList([
                DwtSkipEnhance(self._down_feat_channels[0]),  # Level 0 (32x32)
                DwtSkipEnhance(self._down_feat_channels[1]),  # Level 1 (16x16)
            ])

        self.final_block = Block2D(chs, chs, overparametrization=overparametrization)
        self.final_proj = nn.Conv2d(chs, out_channels, kernel_size=1)

        if self.use_cgfm:
            if len(cond_pyramid_channels) == 3:
                c256, c128, c64 = cond_pyramid_channels
                c32 = c64
            elif len(cond_pyramid_channels) == 4:
                c256, c128, c64, c32 = cond_pyramid_channels
            else:
                raise ValueError(
                    "cond_pyramid_channels must be length 3 (256/128/64) or 4 (256/128/64/32)."
                )

            cond_ch_map = {"256": c256, "128": c128, "64": c64, "32": c32}

            def _cond_key_from_down_pow(down_pow):
                if down_pow <= 0:
                    return "256"
                if down_pow == 1:
                    return "128"
                if down_pow == 2:
                    return "64"
                return "32"

            self.condition_stem = ConditionStem(
                in_channels=cond_in_channels,
                cond_base_channels=cond_base_channels,
            )
            # 256 -> 128 -> 64 的轻量条件金字塔。
            self.cond_to_256 = nn.Conv2d(cond_base_channels, c256, kernel_size=1, stride=1, padding=0)
            self.cond_down_128 = nn.Sequential(
                nn.Conv2d(c256, c128, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            )
            self.cond_down_64 = nn.Sequential(
                nn.Conv2d(c128, c64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            )
            self.cond_down_32 = nn.Sequential(
                nn.Conv2d(c64, c32, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            )

            # 按真实下采样层级自动分配条件分辨率，兼容不同 ch_mult。
            self._down_cond_keys = []
            down_pow = 0
            for idx, mult in enumerate(ch_mult):
                self._down_cond_keys.append(None if idx == 0 else _cond_key_from_down_pow(down_pow))
                if mult != 1:
                    down_pow += 1

            if down_pow > 3:
                raise ValueError(
                    "Native-SFT condition pyramid only provides up to 256/128/64/32 scales, "
                    f"but current ch_mult={ch_mult} creates {down_pow} downsample stages. "
                    "Please use a <=3-stage downsample setup (e.g. ch_mult=[1,2,2,4]) "
                    "or extend condition pyramid with extra native scales."
                )

            mid_cond_key = _cond_key_from_down_pow(down_pow)
            self._mid_cond_key = mid_cond_key

            self._up_cond_keys = []
            up_down_pow = down_pow
            for up_idx, mult in enumerate(ch_mult[::-1]):
                if mult != 1:
                    up_down_pow = max(0, up_down_pow - 1)
                self._up_cond_keys.append(None if up_idx == 0 else _cond_key_from_down_pow(up_down_pow))

            self.sft_down = nn.ModuleList([
                nn.Identity() if key is None else TanhSFT(
                    self._down_feat_channels[idx],
                    cond_ch_map[key],
                    sft_gamma_scale,
                    sft_beta_scale,
                )
                for idx, key in enumerate(self._down_cond_keys)
            ])

            # Mid 全注入：使用与 bottleneck 分辨率一致的条件尺度。
            self.sft_mid = nn.ModuleList([
                TanhSFT(self._mid_feat_channels, cond_ch_map[mid_cond_key], sft_gamma_scale, sft_beta_scale)
                for _ in range(len(self.mid_blocks))
            ])

            self.sft_up = nn.ModuleList([
                nn.Identity() if key is None else TanhSFT(
                    self._up_feat_channels[idx],
                    cond_ch_map[key],
                    sft_gamma_scale,
                    sft_beta_scale,
                )
                for idx, key in enumerate(self._up_cond_keys)
            ])

    def time_gate(self, t):
        if t is None:
            return 1.0
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), dtype=torch.float32)
        t = t.float()
        return self.gate_w_min + (1.0 - self.gate_w_min) * torch.pow(1.0 - t, self.gate_p)

    def make_condition(self, x_lq):
        if (not self.use_cgfm) or x_lq is None:
            return None
        # Build native-resolution condition pyramid from current input size.
        cond256 = self.cond_to_256(self.condition_stem(x_lq))
        cond128 = self.cond_down_128(cond256)
        cond64 = self.cond_down_64(cond128)
        cond32 = self.cond_down_32(cond64)
        return {"256": cond256, "128": cond128, "64": cond64, "32": cond32}

    def reset(self):
        for n, m in self.named_modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def collapse_conv(self, conv1, conv2):
        collapsed_conv = nn.Conv2d(conv1.in_channels,
                                   conv2.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding)
        kx, ky = conv1.weight.shape[2] + conv2.weight.shape[2] - 1, conv1.weight.shape[3] + conv2.weight.shape[3] - 1
        x_pad, y_pad = 2 * kx - 1, 2 * ky - 1
        in_tensor = torch.eye(conv1.weight.shape[1], device=conv1.weight.device)
        in_tensor = torch.unsqueeze(torch.unsqueeze(in_tensor, 2), 3)
        in_tensor = F.pad(in_tensor, (int(math.ceil((x_pad - 1) / 2)),
                                      int(math.floor((x_pad - 1) / 2)),
                                      int(math.ceil((y_pad - 1) / 2)),
                                      int(math.floor((y_pad - 1) / 2))))
        # Run first Conv2D
        conv1_out = F.conv2d(input=in_tensor, weight=conv1.weight, stride=conv1.stride, padding=(0, 0))
        # Run second Conv2D
        conv2_out = F.conv2d(input=conv1_out, weight=conv2.weight, stride=conv2.stride)
        # Extract collapsed kernel from output: the collapsed kernel is the output of the convolution after fixing the dimension
        collapsed_kernel = torch.permute(torch.flip(conv2_out, [3, 2]), dims=[1, 0, 2, 3])
        collapsed_bias = torch.matmul(torch.sum(conv2.weight, dim=(2, 3)), conv1.bias) + conv2.bias
        sd = {"weight": collapsed_kernel, "bias": collapsed_bias}
        collapsed_conv.load_state_dict(sd)
        return collapsed_conv

    def collapse(self):
        if self.overparametrization:
            for _, m in self.named_modules():
                if isinstance(m, Block2D):
                    conv1, conv2 = m.block[2], m.block[3]
                    collapsed_conv = self.collapse_conv(conv1, conv2)
                    m.block[2] = collapsed_conv
                    m.block[3] = nn.Identity()
            self.overparametrization = False

    def load_weights(self, model_path):
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            if model_path.endswith(".ckpt"):
                state_dict = state_dict["state_dict_fmir"]
            self.load_state_dict(state_dict)

    def _run_with_checkpoint(self, module, x, emb=None):
        if (not self.use_checkpoint) or (not self.training):
            return module(x, emb) if emb is not None else module(x)

        if emb is None:
            return checkpoint(lambda _x: module(_x), x, use_reentrant=False)
        return checkpoint(lambda _x, _emb: module(_x, _emb), x, emb, use_reentrant=False)

    def forward(self, xt, t_emb, cond=None, t=None):
        emb = self.time_mlp(t_emb)
        x = self.first_proj(xt)
        w_t = self.time_gate(t)

        cond256 = None
        cond128 = None
        cond64 = None
        cond32 = None
        if isinstance(cond, dict):
            cond256 = cond.get("256")
            cond128 = cond.get("128")
            cond64 = cond.get("64")
            cond32 = cond.get("32")
        cond_map = {"256": cond256, "128": cond128, "64": cond64, "32": cond32}

        # Down blocks
        skip_connect = []
        for idx, (resnet, downsample) in enumerate(self.down_blocks):
            x = self._run_with_checkpoint(resnet, x, emb)
            if self.use_cgfm and idx > 0:
                down_key = self._down_cond_keys[idx]
                x = self.sft_down[idx](x, cond_map.get(down_key), w_t)
            skip_connect.append(x)
            x = downsample(x)
        # Mid blocks
        for mid_idx, resnet in enumerate(self.mid_blocks):
            x = self._run_with_checkpoint(resnet, x, emb)
            if self.use_cgfm:
                x = self.sft_mid[mid_idx](x, cond_map.get(self._mid_cond_key), w_t)
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        # Up blocks
        num_up = len(self.up_blocks)
        for up_idx, (upsample, resnet) in enumerate(self.up_blocks):
            x = upsample(x)
            skip = skip_connect.pop()
            # DWT Skip: 浅层 skip 频带增强, 深层不变
            # (num_up-1-up_idx)=0 最浅层, =1 次浅层, 直接映射到 dwt_skip_layers
            if self.use_dwt_skip:
                shallow_idx = num_up - 1 - up_idx
                if shallow_idx < len(self.dwt_skip_layers):
                    skip = self.dwt_skip_layers[shallow_idx](skip)
            x = torch.concat([x, skip], dim=1)
            x = self._run_with_checkpoint(resnet, x, emb)
            if self.use_cgfm and up_idx > 0:
                up_key = self._up_cond_keys[up_idx]
                x = self.sft_up[up_idx](x, cond_map.get(up_key), w_t)
        x = self._run_with_checkpoint(self.final_block, x)
        x = self.final_proj(x)
        return x

    def collect_gate_curve(self, x, cond=None, num_points=10):
        """采集不同 t 下的平均门控激活值, 用于画频率解耦曲线。

        Returns: ts (list[float]), gates (list[float])
        """
        self.eval()

        # Local pos_emb to avoid circular import
        def _pos_emb(t, dim, scale=1000):
            half_dim = dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t.float() * emb.unsqueeze(0)
            emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
            return emb

        ts = torch.linspace(0, 1, num_points)
        gates = []

        for t_val in ts:
            t_batch = t_val.expand(x.shape[0])
            t_emb = _pos_emb(t_batch, self.t_emb_dim).to(x.device)
            # 重置所有 _last_gate
            self._reset_gates()

            with torch.no_grad():
                _ = self.forward(x, t_emb, cond=cond, t=t_batch)

            g = self._collect_gates()
            gates.append(g)

        return ts.tolist(), gates

    def _reset_gates(self):
        for module in self.modules():
            if hasattr(module, '_last_gate'):
                module._last_gate = None

    def _collect_gates(self):
        vals = []
        for module in self.modules():
            if hasattr(module, '_last_gate') and module._last_gate is not None:
                vals.append(module._last_gate)
        return sum(vals) / len(vals) if vals else 0.0
