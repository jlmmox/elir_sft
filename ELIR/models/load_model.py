from ELIR.utils import get_model_size
from safetensors import safe_open
import torch
from utils import get_device

device = get_device()


def get_model(cfg):
    model_name = cfg.get("name")
    model_path = cfg.get("path")
    model_params = cfg.get("params", {})
    trainable = cfg.get("trainable",False)

    if model_name == "elir":
        from ELIR.models.elir import Elir
        model = Elir(**model_params)
        model.load_weights(model_path)
        return model
    elif model_name == "lunet":
        from ELIR.models.lunet import LUnet
        model = LUnet(**model_params)
        model.load_weights(model_path)
    elif model_name == "rrdbnet":
        from ELIR.models.rrdbnet import RRDBNet
        model = RRDBNet(**model_params)
        model.load_weights(model_path)
    elif model_name == "tiny_enc":
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        from ELIR.models.taesd import TAESD
        model = TAESD()
        model.load_state_dict(pretrained.state_dict())
        model = model.encoder
    elif model_name == "tiny_dec":
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        from ELIR.models.taesd import TAESD
        model = TAESD()
        model.load_state_dict(pretrained.state_dict())
        model = model.decoder
    elif model_name == "taesd":
        from ELIR.models.taesd import TAESD
        model = TAESD()
        from diffusers import AutoencoderTiny
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        model.load_state_dict(pretrained.state_dict())
    elif model_name == "sft_taesd_finetuner":
        from diffusers import AutoencoderTiny
        from ELIR.models.sft_taesd_finetuner import SFT_TAESDFineTuner

        # 使用 diffusers 官方 tiny VAE 权重构造 decoder 主干。
        pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
        decoder = pretrained.decoder
        model = SFT_TAESDFineTuner(decoder, **model_params)

        # 可选加载 SFT 权重（支持完整 state_dict 或仅 SFT 子字典）。
        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            if model_path.endswith(".ckpt"):
                state_dict = state_dict.get("state_dict_sft", state_dict)
            if "state_dict_sft" in state_dict:
                state_dict = state_dict["state_dict_sft"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print("[sft_taesd_finetuner] missing keys:", missing)
            if unexpected:
                print("[sft_taesd_finetuner] unexpected keys:", unexpected)
    elif model_name == "encoder_skip_fusion":
        from ELIR.models.encoder_skip_fusion import EncoderSkipFusion

        taesd_encoder = model_params.pop("taesd_encoder", None)
        if taesd_encoder is None:
            # fallback: 从 diffusers 加载独立 TAESD decoder
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            taesd_encoder = pretrained.encoder
            taesd_decoder = pretrained.decoder
        else:
            # taesd_encoder 已由 elir.py 注入；从 diffusers 加载 decoder 主干
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            taesd_decoder = pretrained.decoder

        model = EncoderSkipFusion(
            taesd_encoder=taesd_encoder,
            taesd_decoder=taesd_decoder,
            **model_params,
        )

        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            if model_path.endswith(".ckpt"):
                state_dict = state_dict.get("state_dict_sft", state_dict)
            if "state_dict_sft" in state_dict:
                state_dict = state_dict["state_dict_sft"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print("[encoder_skip_fusion] missing keys:", missing)
            if unexpected:
                print("[encoder_skip_fusion] unexpected keys:", unexpected)
    elif model_name == "encoder_skip_wavelet_fusion":
        from ELIR.models.encoder_skip_wavelet_fusion import EncoderSkipWaveletFusion

        taesd_encoder = model_params.pop("taesd_encoder", None)
        if taesd_encoder is None:
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            taesd_encoder = pretrained.encoder
            taesd_decoder = pretrained.decoder
        else:
            from diffusers import AutoencoderTiny
            pretrained = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
            taesd_decoder = pretrained.decoder

        model = EncoderSkipWaveletFusion(
            taesd_encoder=taesd_encoder,
            taesd_decoder=taesd_decoder,
            **model_params,
        )

        if model_path:
            state_dict = torch.load(model_path, weights_only=True)
            if model_path.endswith(".ckpt"):
                state_dict = state_dict.get("state_dict_sft", state_dict)
            if "state_dict_sft" in state_dict:
                state_dict = state_dict["state_dict_sft"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print("[encoder_skip_wavelet_fusion] missing keys:", missing)
            if unexpected:
                print("[encoder_skip_wavelet_fusion] unexpected keys:", unexpected)
    elif model_name == "fm_vae":
        from ELIR.models.fm_vae import FMVAEWrapper
        model = FMVAEWrapper(pretrained_path=model_path)
    elif model_name == "sdvae":
        from ELIR.models.sdvae import SDVAEWrapper
        model = SDVAEWrapper()
    else:
        raise Exception("Model {} is unknown!".format(model_name))

    if model_name != "fm_vae":
        if trainable:
            # 默认可训练分支：模型全参数可训练。
            for param in model.parameters():
                param.requires_grad = True

            # 包装器强制恢复为"仅融合模块可训练"。
            if hasattr(model, "set_trainable_only"):
                model.set_trainable_only()
            elif hasattr(model, "set_sft_trainable_only"):
                model.set_sft_trainable_only()
            model.train()
        else:
            # Freeze all
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    model.to(device)
    print("{} was created! Number of parameters: {:0.2f}M".format(model_name, get_model_size(model)/1e6))
    return model



