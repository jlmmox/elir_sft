import torch


def get_optimizer(train_cfg, model):
    lr = train_cfg.get("lr", 0.0001)
    enc_lr_mult = float(train_cfg.get("enc_lr_mult", 0.1))
    optimizer_params = train_cfg.get("optimizer_params", {})

    # 拆分 Encoder 参数和其他参数，Encoder 用更低的学习率
    enc_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("enc."):
            enc_params.append(p)
        else:
            other_params.append(p)

    if len(other_params) + len(enc_params) == 0:
        raise ValueError("No trainable parameters found.")

    param_groups = [
        {"params": other_params, "lr": lr, **optimizer_params},
    ]
    if enc_params:
        param_groups.append({
            "params": enc_params,
            "lr": lr * enc_lr_mult,
            **{k: v for k, v in optimizer_params.items() if k != "lr"},
        })

    optimizer = train_cfg.get("optimizer", None)
    if optimizer:
        return optimizer(param_groups)
    else:
        return torch.optim.Adam(param_groups)

def get_scheduler(train_cfg, optimizer):
    scheduler_params = train_cfg.get("scheduler_params", {})
    scheduler = train_cfg.get("scheduler", None)
    if scheduler:
        scheduler = scheduler(optimizer, **scheduler_params)
    return scheduler

def get_opt_sched(train_cfg, model):
    # Optimizer
    optimizer = get_optimizer(train_cfg, model)
    # Scheduler
    scheduler = get_scheduler(train_cfg, optimizer)
    return optimizer, scheduler

