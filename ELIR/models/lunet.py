import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint





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
    def __init__(self, in_channels, out_channels, time_dim, overparametrization=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = Block2D(in_channels, out_channels, overparametrization=overparametrization)
        self.block2 = Block2D(out_channels, out_channels, overparametrization=overparametrization)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        h = self.block1(x)
        h += self.mlp(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        out = h + self.conv2d(x)
        return out


class ConditionStem(nn.Module):
    def __init__(self, in_channels=3, downscale_factor=4, cond_base_channels=32):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        stem_in_channels = in_channels * (downscale_factor ** 2)
        self.proj = nn.Sequential(
            nn.Conv2d(stem_in_channels, cond_base_channels, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(cond_base_channels, cond_base_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.unshuffle(x)
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
            cond = F.interpolate(cond, size=feat.shape[-2:], mode="nearest")
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
                 cond_base_channels=32, cond_pyramid_channels=(32, 48, 64),
                 gate_w_min=0.1, gate_p=1.5, sft_gamma_scale=0.2, sft_beta_scale=0.2,
                 use_checkpoint=False):
        super(LUnet, self).__init__()
        self.overparametrization = overparametrization
        self.t_emb_dim = t_emb_dim
        self.use_cgfm = bool(use_cgfm)
        self.use_checkpoint = bool(use_checkpoint)
        self.gate_w_min = float(gate_w_min)
        self.gate_p = float(gate_p)
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
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization)
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
            resnet = ResnetBlock2D(chs, chs, time_dim_out, overparametrization=overparametrization)
            self.mid_blocks.append(resnet)

        # Up blocks
        for mult in ch_mult[::-1]:
            if mult!=1:
                upsample = Upsample(chs, chs//mult, use_convtr=use_rescale_conv)
                chs = chs // mult
            else:
                upsample = nn.Identity()
            resnet = ResnetBlock2D(2*chs, chs, time_dim_out, overparametrization=overparametrization)
            self._up_feat_channels.append(chs)
            self.up_blocks.append(nn.ModuleList([upsample, resnet]))

        self.final_block = Block2D(chs, chs, overparametrization=overparametrization)
        self.final_proj = nn.Conv2d(chs, out_channels, kernel_size=1)

        if self.use_cgfm:
            c64, c32, c16 = cond_pyramid_channels
            self.condition_stem = ConditionStem(
                in_channels=cond_in_channels,
                downscale_factor=cond_downscale_factor,
                cond_base_channels=cond_base_channels,
            )
            self.cond_to_64 = nn.Conv2d(cond_base_channels, c64, kernel_size=1, stride=1, padding=0)
            self.cond_down_32 = nn.Conv2d(c64, c32, kernel_size=3, stride=2, padding=1)
            self.cond_down_16 = nn.Conv2d(c32, c16, kernel_size=3, stride=2, padding=1)

            # Five-point injection: E2, E4, M2, D1, D3.
            e2_ch = self._down_feat_channels[1] if len(self._down_feat_channels) > 1 else self._down_feat_channels[-1]
            e4_ch = self._down_feat_channels[3] if len(self._down_feat_channels) > 3 else self._down_feat_channels[-1]
            m2_ch = self._mid_feat_channels
            d1_ch = self._up_feat_channels[0] if len(self._up_feat_channels) > 0 else self._up_feat_channels[-1]
            d3_ch = self._up_feat_channels[2] if len(self._up_feat_channels) > 2 else self._up_feat_channels[-1]

            self.sft_e2 = TanhSFT(feat_channels=e2_ch, cond_channels=c64,
                                  gamma_scale=sft_gamma_scale, beta_scale=sft_beta_scale)
            self.sft_e4 = TanhSFT(feat_channels=e4_ch, cond_channels=c32,
                                  gamma_scale=sft_gamma_scale, beta_scale=sft_beta_scale)
            self.sft_m2 = TanhSFT(feat_channels=m2_ch, cond_channels=c16,
                                  gamma_scale=sft_gamma_scale, beta_scale=sft_beta_scale)
            self.sft_d1 = TanhSFT(feat_channels=d1_ch, cond_channels=c32,
                                  gamma_scale=sft_gamma_scale, beta_scale=sft_beta_scale)
            self.sft_d3 = TanhSFT(feat_channels=d3_ch, cond_channels=c64,
                                  gamma_scale=sft_gamma_scale, beta_scale=sft_beta_scale)

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
        cond64 = self.cond_to_64(self.condition_stem(x_lq))
        cond32 = self.cond_down_32(cond64)
        cond16 = self.cond_down_16(cond32)
        return {"64": cond64, "32": cond32, "16": cond16}

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

        cond64 = None
        cond32 = None
        cond16 = None
        if isinstance(cond, dict):
            cond64 = cond.get("64")
            cond32 = cond.get("32")
            cond16 = cond.get("16")

        # Down blocks
        skip_connect = []
        for idx, (resnet, downsample) in enumerate(self.down_blocks):
            x = self._run_with_checkpoint(resnet, x, emb)
            if idx == 1:
                x = self.sft_e2(x, cond64, w_t) if self.use_cgfm else x
            if idx == 3:
                x = self.sft_e4(x, cond32, w_t) if self.use_cgfm else x
            skip_connect.append(x)
            x = downsample(x)
        # Mid blocks
        for mid_idx, resnet in enumerate(self.mid_blocks):
            x = self._run_with_checkpoint(resnet, x, emb)
            if mid_idx == 1:
                x = self.sft_m2(x, cond16, w_t) if self.use_cgfm else x
        # Up blocks
        for up_idx, (upsample, resnet) in enumerate(self.up_blocks):
            x = upsample(x)
            x = torch.concat([x,skip_connect.pop()], dim=1)
            x = self._run_with_checkpoint(resnet, x, emb)
            if up_idx == 0:
                x = self.sft_d1(x, cond32, w_t) if self.use_cgfm else x
            if up_idx == 2:
                x = self.sft_d3(x, cond64, w_t) if self.use_cgfm else x
        x = self._run_with_checkpoint(self.final_block, x)
        x = self.final_proj(x)
        return x
