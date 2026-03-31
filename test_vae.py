from args_handler import argument_handler, set_overides
from hyperpyyaml import load_hyperpyyaml
from utils import set_seed
from torchvision.utils import save_image
import torch.nn.functional as F
from torchvision.io import read_image
import torch
import os
from ELIR.models.load_model import get_model
from ELIR.metrics import MetricEval
from tqdm import tqdm
from utils import get_device
import glob
import warnings
warnings.filterwarnings("ignore")
device = get_device()

IMAGE_EXTENSION = ('jpg','png','jpeg')


def _pad_to_multiple(x, multiple=8):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, h, w
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, h, w


def preprocess(img_tensor, img_size=None):
    img_tensor = img_tensor[:3, ...]
    img_tensor = img_tensor.unsqueeze(0) / 255.0
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    if img_size is not None:
        img_tensor = F.interpolate(img_tensor, size=(img_size, img_size), mode='bicubic')
    img_tensor = img_tensor.to(device)
    return img_tensor


def encode_decode(model, x):
    if hasattr(model, "vae") and hasattr(model.vae, "encoder") and hasattr(model.vae, "decoder"):
        return model.vae.decoder(model.vae.encoder(x))
    if hasattr(model, "enc") and hasattr(model, "dec") and hasattr(model.enc, "encode") and hasattr(model.dec, "decode"):
        return model.dec.decode(model.enc.encode(x))
    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        return model.decoder(model.encoder(x))
    if hasattr(model, "enc") and hasattr(model, "dec"):
        return model.dec(model.enc(x))
    raise AttributeError("Model does not provide encoder/decoder interfaces for VAE test")


def run_vae(conf):
    # ----------------------------
    # Set environmnet
    # ----------------------------
    env_cfg = conf.get("env_cfg")
    seed = env_cfg.get("seed",0)
    set_seed(seed)

    # ----------------------------
    # Create model
    # ----------------------------
    model_cfg = conf.get("model_cfg")
    arch_cfg = model_cfg.get("arch_cfg")
    model = get_model(arch_cfg)

    # ----------------------------
    # Infer all images in folder
    # ----------------------------
    eval_cfg = conf.get("eval_cfg")
    image_folder = eval_cfg.get("image_folder", None)
    if image_folder is None:
        image_folder = eval_cfg.get("in_folder", None)
    if image_folder is None:
        raise ValueError("Missing eval_cfg.image_folder or eval_cfg.in_folder in yaml config")
    out_image_folder = eval_cfg.get("out_folder", os.path.join(image_folder, "..", "out"))
    img_size = eval_cfg.get("image_size", 512)
    if isinstance(img_size, str) and img_size.lower() in {"none", "null"}:
        img_size = None
    os.makedirs(out_image_folder, exist_ok=True) # run folder
    mse_values = []
    metrics = eval_cfg.get("metrics", [])
    metric_evals = [MetricEval(metric, device, None) for metric in metrics]
    pbar = tqdm(sorted(glob.glob(os.path.join(image_folder, "*.*"))))
    for img_path in pbar:
        if not img_path.lower().endswith(IMAGE_EXTENSION):
            continue
        img_tensor = read_image(img_path)
        x = preprocess(img_tensor, img_size)
        x_in, h0, w0 = _pad_to_multiple(x, multiple=8)
        y = encode_decode(model, x_in)
        # Align decoded output back to original spatial size for fair MSE.
        h_cmp = min(h0, y.shape[-2])
        w_cmp = min(w0, y.shape[-1])
        y = y[..., :h_cmp, :w_cmp]
        x_cmp = x[..., :h_cmp, :w_cmp]
        out_path = os.path.join(out_image_folder,os.path.basename(img_path))
        save_image(y, out_path)
        mse = torch.mean((y - x_cmp) ** 2)
        mse_values.append(mse.item())
        for metric_eval in metric_evals:
            metric_eval.compute(y, x_cmp)
        pbar.set_postfix(mse=f"{mse.item():0.6f}")
        del img_tensor, y

    if len(mse_values) > 0:
        mean_mse = sum(mse_values) / len(mse_values)
        print(f"Mean MSE={mean_mse:0.6f}")
    if len(metric_evals) > 0:
        metric_parts = []
        for metric_eval in metric_evals:
            result = metric_eval.get_final().item()
            metric_parts.append(f"{metric_eval.metric}={result:0.4f}")
        print(", ".join(metric_parts))
    print("Done! images are at {}".format(out_image_folder))

if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    yaml_path, overides = argument_handler()
    with open(yaml_path) as yaml_stream:
        conf = load_hyperpyyaml(yaml_stream)
    set_overides(conf, overides)

    # ----------------------------
    # Eval
    # ----------------------------
    run_vae(conf)