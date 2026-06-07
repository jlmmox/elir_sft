import torch
import torch.nn as nn

from ELIR.models.load_model import get_model


def build_model():
    # 通过统一模型工厂构建：主干 decoder 冻结，SFT 可训练。
    cfg = {
        "name": "sft_taesd_finetuner",
        "path": None,
        "trainable": True,
        "params": {
            "gamma_scale": 0.2,
            "beta_scale": 0.2,
        },
    }
    model = get_model(cfg)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)

    # 关键：优化器仅接收 SFT 参数，避免任何误更新。
    optimizer = torch.optim.AdamW(model.sft_parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    # ---------- 最小 batch 示例 ----------
    bsz = 2
    z = torch.randn(bsz, 16, 32, 32, device=device)
    cond_dict = {
        "cond32": torch.randn(bsz, 64, 32, 32, device=device),
        "cond64": torch.randn(bsz, 64, 64, 64, device=device),
        "cond128": torch.randn(bsz, 32, 128, 128, device=device),
        "cond256": torch.randn(bsz, 16, 256, 256, device=device),
    }
    target = torch.randn(bsz, 3, 256, 256, device=device)

    # 防呆检查：decoder 参数必须全部冻结。
    decoder_params = list(model.taesd_decoder.parameters())
    if any(p.requires_grad for p in decoder_params):
        raise RuntimeError("TAESD decoder should be frozen, but some params are trainable.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(z, cond_dict)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    # 统计梯度状态，验证只有 SFT 被更新。
    sft_grad_count = 0
    for p in model.sft_parameters():
        if p.grad is not None:
            sft_grad_count += 1

    print(f"loss={loss.item():.6f}, sft_grad_tensors={sft_grad_count}")
    print("Done: only SFT modules are optimized.")


if __name__ == "__main__":
    main()
