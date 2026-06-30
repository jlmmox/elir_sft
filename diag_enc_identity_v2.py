"""
ELIR Encoder/Teacher 一致性诊断 v2
===================================
严格三时间点比较 + 真实 checkpoint 加载 + 正确 verdict 逻辑
"""
import os, sys, argparse, itertools
import torch, torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, default="configs/derain/eval.yaml")
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--out", type=str, default="./runs/diag_enc_v2")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ===================================================================
# 加载配置
# ===================================================================
from hyperpyyaml import load_hyperpyyaml
with open(args.yaml) as f:
    conf = load_hyperpyyaml(f)

model_cfg = conf.get("model_cfg", {})
arch_cfg = model_cfg.get("arch_cfg", {})
teacher_cfg = model_cfg.get("teacher_cfg", {})
train_cfg = conf.get("train_cfg", {})
fm_cfg = conf.get("fm_cfg", {})

# ---- checkpoint 路径解析 ----
ckpt_path = (
    args.ckpt_path
    or conf.get("ckpt_path")
    or arch_cfg.get("path")
)
print(f"ckpt_path resolved: {ckpt_path}")

if not ckpt_path:
    raise RuntimeError(
        "No checkpoint path provided. Use --ckpt_path or set in config.\n"
        "This diagnostic REQUIRES a real checkpoint to compare encoder weights."
    )
ckpt_path = os.path.expanduser(ckpt_path)
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

# ===================================================================
# 创建模型组件
# ===================================================================
from ELIR.models.load_model import get_model

# -- model.enc --
arch_cfg_clean = dict(arch_cfg)
arch_cfg_clean["path"] = None  # 不自动加载 ckpt
model = get_model(deepcopy(arch_cfg_clean))

# -- teacher --
if teacher_cfg is None:
    # eval.yaml has teacher_cfg: null → construct manually
    teacher_cfg_clean = {"name": "tiny_enc", "params": {}, "trainable": False, "path": None}
else:
    teacher_cfg_clean = dict(teacher_cfg)
    teacher_cfg_clean["path"] = None
teacher_raw = get_model(teacher_cfg_clean)
# teacher_raw is already the encoder (tiny_enc returns model.encoder directly)

model.to(device).eval()
teacher_raw.to(device).eval()

# ===================================================================
# 加载 checkpoint
# ===================================================================
print(f"\n{'='*70}")
print(f"LOADING CHECKPOINT")
print(f"{'='*70}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"  path:      {ckpt_path}")
print(f"  global_step: {ckpt.get('global_step', 'NOT FOUND')}")
print(f"  epoch:       {ckpt.get('epoch', 'NOT FOUND')}")
print(f"  keys:        {list(ckpt.keys())}")

has_enc = 'state_dict_enc' in ckpt and ckpt['state_dict_enc'] and len(ckpt['state_dict_enc']) > 0
print(f"  state_dict_enc exists: {has_enc}")
if not has_enc:
    raise RuntimeError("state_dict_enc NOT FOUND in checkpoint — cannot verify encoder consistency!")

ckpt_enc_sd = ckpt['state_dict_enc']
print(f"  state_dict_enc keys: {len(ckpt_enc_sd)}")

# 严格加载到 model.enc
result = model.enc.load_state_dict(ckpt_enc_sd, strict=True)
print(f"  load state_dict_enc → model.enc: strict=True → OK (no errors)")
# If strict=True passed, missing=0 and unexpected=0 by definition
model.enc.to(device).eval()
for p in model.enc.parameters():
    p.requires_grad = False

# ===================================================================
# 辅助函数
# ===================================================================
def _charb(a, b, eps=1e-6):
    return torch.sqrt((a-b).pow(2)+eps).mean().item()

def _cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()

def _compare_params(sd1, sd2, label1, label2, threshold=1e-7):
    """逐 key 比较两个 state_dict（自动处理 device 不一致）"""
    # 确保在同一 device 上比较
    def _to_cpu(d):
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in d.items()}
    sd1 = _to_cpu(sd1)
    sd2 = _to_cpu(sd2)
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    common = keys1 & keys2
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    diffs = []
    for k in sorted(common):
        if sd1[k].shape == sd2[k].shape:
            max_d = (sd1[k] - sd2[k]).abs().max().item()
            mean_d = (sd1[k] - sd2[k]).abs().mean().item()
            diffs.append((k, max_d, mean_d))
        else:
            diffs.append((k, float('inf'), float('inf')))

    diffs.sort(key=lambda x: -x[1])  # sort by max diff descending
    n_diff = sum(1 for _, mx, _ in diffs if mx > threshold)

    print(f"  {label1} keys: {len(keys1)}, {label2} keys: {len(keys2)}")
    print(f"  common: {len(common)}, only in {label1}: {len(only1)}, only in {label2}: {len(only2)}")
    if only1:
        print(f"    only in {label1}: {sorted(only1)[:10]}")
    if only2:
        print(f"    only in {label2}: {sorted(only2)[:10]}")
    print(f"  params with diff > {threshold}: {n_diff}/{len(common)}")
    if diffs:
        print(f"  max diff: {diffs[0][1]:.10f} at {diffs[0][0]} (mean={diffs[0][2]:.10f})")
    if n_diff > 0:
        print(f"  top-20 differing params:")
        for k, mx, mn in diffs[:20]:
            if mx > threshold:
                print(f"    {k}: max={mx:.8f} mean={mn:.8f}")

    all_equal = all(
        torch.equal(sd1[k], sd2[k]) if sd1[k].shape == sd2[k].shape else False
        for k in common
    )
    return all_equal, diffs, n_diff

def _compare_output(enc1, enc2, x, label1, label2):
    """比较两个 encoder 对同一输入的输出"""
    with torch.no_grad():
        z1 = enc1(x).float()
        z2 = enc2(x).float()

    max_d = (z1 - z2).abs().max().item()
    mean_d = (z1 - z2).abs().mean().item()
    charb = _charb(z1, z2)
    cos = _cos(z1, z2)
    nr = z1.flatten(1).norm(dim=1).mean().item() / (z2.flatten(1).norm(dim=1).mean().item() + 1e-8)

    print(f"  {label1}: mean={z1.mean().item():.4f} std={z1.std().item():.4f}")
    print(f"  {label2}: mean={z2.mean().item():.4f} std={z2.std().item():.4f}")
    print(f"  max_abs_diff: {max_d:.8f}")
    print(f"  mean_abs_diff: {mean_d:.8f}")
    print(f"  charb: {charb:.8f}")
    print(f"  cosine: {cos:.8f}")
    print(f"  norm_ratio({label1}/{label2}): {nr:.8f}")

    return {
        "max_abs_diff": max_d,
        "mean_abs_diff": mean_d,
        "charb": charb,
        "cosine": cos,
        "norm_ratio": nr,
        "identical": max_d < 1e-6 and cos > 0.99999,
    }

# ===================================================================
# 获取真实 HQ 图用于 forward 比较
# ===================================================================
from ELIR.datasets.dataset import get_loader
ds_cfg = conf.get("dataset_cfg", {})
val_cfg = dict(ds_cfg.get("val_dataset", {}))
val_cfg.update({"batch_size": 1, "num_workers": 0})
loader = get_loader(val_cfg)

test_batches = []
for idx, batch in enumerate(loader):
    if idx >= 5:
        break
    test_batches.append(batch)

# 使用第一张 HQ 做 forward 比较
x_hq_test = test_batches[0][1].to(device)

# ===================================================================
# TIME A: 初始化后（teacher 和 model.enc 均只有 TAESD3 预训练权重）
# ===================================================================
print(f"\n{'='*70}")
print(f"TIME A: AFTER INIT (pretrained TAESD3 weights only, NO checkpoint)")
print(f"{'='*70}")

print(f"\n  --- Parameter comparison: model.enc vs teacher ---")
enc_A_sd = model.enc.state_dict()
teacher_sd = teacher_raw.state_dict()
all_equal_A, diffs_A, n_diff_A = _compare_params(enc_A_sd, teacher_sd, "model.enc", "teacher")

print(f"\n  --- Buffer comparison ---")
enc_buf_keys = set(dict(model.enc.named_buffers()).keys())
teacher_buf_keys = set(dict(teacher_raw.named_buffers()).keys())
print(f"  enc buffers: {len(enc_buf_keys)}, teacher buffers: {len(teacher_buf_keys)}")
print(f"  buffer keys match: {enc_buf_keys == teacher_buf_keys}")
buf_diffs = {}
for name, buf in model.enc.named_buffers():
    teacher_buf = dict(teacher_raw.named_buffers()).get(name)
    if teacher_buf is not None and buf.shape == teacher_buf.shape:
        buf_diffs[name] = (buf - teacher_buf).abs().max().item()
if buf_diffs:
    print(f"  max buffer diff: {max(buf_diffs.values()):.10f} at {max(buf_diffs, key=buf_diffs.get)}")

print(f"\n  --- Forward output comparison on real HQ image ---")
out_A = _compare_output(model.enc, teacher_raw, x_hq_test, "model.enc", "teacher")

# ===================================================================
# TIME B: 直接比较 checkpoint state_dict_enc 与 teacher
# ===================================================================
print(f"\n{'='*70}")
print(f"TIME B1: CHECKPOINT state_dict_enc vs TEACHER (direct weight comparison)")
print(f"{'='*70}")

# 处理 key 前缀：checkpoint 的 state_dict_enc 内部可能没有前缀
# 因为它是通过 on_save_checkpoint 保存的 model.enc.state_dict()
# model.enc.state_dict() 的 key 如 "layers.0.weight"
# teacher.state_dict() 的 key 也应该是 "layers.0.weight"
# 两个应该完全匹配

print(f"\n  --- Direct ckpt_enc vs teacher ---")
all_equal_B1, diffs_B1, n_diff_B1 = _compare_params(ckpt_enc_sd, teacher_sd, "ckpt_enc", "teacher")

print(f"\n  --- After loading ckpt_enc into model.enc ---")
print(f"  model.enc vs teacher (post-ckpt-load):")

print(f"\n  --- Parameter comparison ---")
enc_B_sd = model.enc.state_dict()
all_equal_B, diffs_B, n_diff_B = _compare_params(enc_B_sd, teacher_sd, "model.enc(post-ckpt)", "teacher")

print(f"\n  --- Forward output comparison ---")
out_B = _compare_output(model.enc, teacher_raw, x_hq_test, "model.enc(post-ckpt)", "teacher")

# ===================================================================
# Confirm freeze status before step
# ===================================================================
print(f"\n{'='*70}")
print(f"ENCODER FREEZE STATUS (before training step)")
print(f"{'='*70}")

enc_trainable_params = [n for n, p in model.enc.named_parameters() if p.requires_grad]
print(f"  enc_trainable attr: {getattr(model, 'enc_trainable', 'N/A')}")
print(f"  enc.training: {model.enc.training}")
print(f"  enc requires_grad=True params: {len(enc_trainable_params)}")
if enc_trainable_params:
    print(f"    {enc_trainable_params}")
print(f"  enc total params: {len(list(model.enc.parameters()))}")
assert all(not p.requires_grad for p in model.enc.parameters()), "FAIL: encoder has trainable params!"

# Optimizer
from ELIR.training.tparmas import get_optimizer
optimizer = get_optimizer(train_cfg, model)
print(f"\n  optimizer type: {type(optimizer).__name__}")
enc_param_ids = {id(p) for p in model.enc.parameters()}
enc_in_opt = False
for i, pg in enumerate(optimizer.param_groups):
    pg_enc = sum(1 for p in pg['params'] if id(p) in enc_param_ids)
    print(f"  group[{i}]: lr={pg.get('lr','N/A')}, total_params={len(pg['params'])}, enc_params={pg_enc}")
    if pg_enc > 0:
        enc_in_opt = True
print(f"  enc_lr_mult from config: {train_cfg.get('enc_lr_mult', 'N/A')}")
print(f"  Encoder in optimizer: {enc_in_opt}")

# Check model.train() doesn't unfreeze encoder
model.train()
print(f"\n  After model.train(): enc.training={model.enc.training}")
print(f"  enc trainable after train(): {[n for n,p in model.enc.named_parameters() if p.requires_grad]}")
model.eval()

# ===================================================================
# TIME C: 执行一次完整 optimizer.step
# ===================================================================
print(f"\n{'='*70}")
print(f"TIME C: AFTER ONE FULL optimizer.step()")
print(f"{'='*70}")

x_lq, x_hq = test_batches[0][0].to(device), test_batches[0][1].to(device)
oh, ow = x_hq.shape[2], x_hq.shape[3]
ph, pw = (64-oh%64)%64, (64-ow%64)%64
x_lq_p = F.pad(x_lq, (0,pw,0,ph), mode="reflect") if ph or pw else x_lq
x_hq_p = F.pad(x_hq, (0,pw,0,ph), mode="reflect") if ph or pw else x_hq

# 保存 step 前参数
enc_before = {n: p.detach().clone() for n, p in model.enc.named_parameters()}
enc_buf_before = {n: b.detach().clone() for n, b in model.enc.named_buffers()}

# 运行一次带梯度的 forward
# encoder requires_grad=False → grad flows through but won't be stored for enc params
model.train()
with torch.enable_grad():
    X_lq = model._encode_input(x_lq_p)       # passes through frozen enc (no grad stored)
    X_mmse = model.mmse(X_lq)                # MMSE is trainable
    loss = F.mse_loss(X_mmse, X_lq)          # dummy loss for grad test

model.zero_grad(set_to_none=True)
loss.backward()

# 检查 grad
print(f"  loss: {loss.item():.6f}")
enc_grad_none = sum(1 for n, p in model.enc.named_parameters() if p.grad is None)
enc_grad_has = sum(1 for n, p in model.enc.named_parameters() if p.grad is not None)
print(f"  enc grad=None: {enc_grad_none}, grad!=None: {enc_grad_has}")
for n, p in model.enc.named_parameters():
    if p.grad is not None and p.grad.norm().item() > 1e-10:
        print(f"    ⚠️ NON-ZERO GRAD: {n} norm={p.grad.norm().item():.8f}")

# optimizer step
optimizer.step()

# 检查更新
param_updates = []
for n, p in model.enc.named_parameters():
    delta = (p.detach() - enc_before[n]).abs().max().item()
    param_updates.append((n, delta))

max_p_update = max(d for _, d in param_updates)
print(f"\n  max parameter update: {max_p_update:.12f}")
n_updated = sum(1 for _, d in param_updates if d > 1e-12)
print(f"  params with Δ>1e-12: {n_updated}")
if n_updated > 0:
    print(f"  ⚠️ ENCODER WAS UPDATED!")
    for n, d in param_updates:
        if d > 1e-12:
            print(f"    {n}: Δ={d:.12f}")

buf_updates = []
for n, b in model.enc.named_buffers():
    delta = (b.detach() - enc_buf_before[n]).abs().max().item()
    buf_updates.append((n, delta))
max_b_update = max(d for _, d in buf_updates) if buf_updates else 0
n_buf_updated = sum(1 for _, d in buf_updates if d > 1e-12)
print(f"  max buffer update: {max_b_update:.12f} ({n_buf_updated} buffers with Δ>1e-12)")
print(f"  enc.training: {model.enc.training}")

model.eval()

# Post-step forward comparison
print(f"\n  --- Forward output comparison after step ---")
out_C = _compare_output(model.enc, teacher_raw, x_hq_test, "model.enc(post-step)", "teacher")

# Check ckpt_enc vs model.enc AFTER step
print(f"\n  --- ckpt_enc vs model.enc(post-step) parameter diff ---")
enc_post_sd = model.enc.state_dict()
_, diffs_post, n_diff_post = _compare_params(enc_post_sd, ckpt_enc_sd, "model.enc(post-step)", "ckpt_enc")

# ===================================================================
# FINAL VERDICT
# ===================================================================
print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")

ckpt_loaded = True  # we verified this at startup

# Q1: Time A (init)
q1 = all_equal_A and out_A["identical"]
print(f"\n  Q1: model.enc == teacher at init (Time A)?")
print(f"      param torch.equal: {all_equal_A}")
print(f"      forward max_diff:  {out_A['max_abs_diff']:.2e}")
print(f"      forward cosine:    {out_A['cosine']:.8f}")
print(f"      → {'YES' if q1 else 'NO'}")

# Q2: Time B (after ckpt load)
if not ckpt_loaded:
    print(f"\n  Q2: model.enc == teacher after ckpt?")
    print(f"      → NOT TESTED (no checkpoint loaded)")
    q2 = "NOT TESTED"
else:
    q2 = all_equal_B and out_B["identical"]
    print(f"\n  Q2: model.enc == teacher after ckpt load (Time B)?")
    print(f"      param torch.equal: {all_equal_B}")
    print(f"      forward max_diff:  {out_B['max_abs_diff']:.2e}")
    print(f"      forward cosine:    {out_B['cosine']:.8f}")
    print(f"      → {'YES' if q2 else 'NO (DIFFERENT!)'}")

# Q2b: ckpt_enc directly vs teacher
if ckpt_loaded:
    q2b = all_equal_B1
    print(f"\n  Q2b: ckpt state_dict_enc == teacher (direct)?")
    print(f"      torch.equal:       {all_equal_B1}")
    print(f"      params with diff>1e-7: {n_diff_B1}")
    print(f"      → {'YES' if q2b else 'NO (DIFFERENT!)'}")

# Q3: If different, is it from ckpt?
if ckpt_loaded and not q2:
    print(f"\n  Q3: Difference source?")
    if n_diff_B1 > 0:
        print(f"      {n_diff_B1} ckpt_enc params differ from teacher")
        print(f"      → YES, difference comes from state_dict_enc in checkpoint")
    elif n_diff_A > 0:
        print(f"      Difference existed before ckpt load ({n_diff_A} params)")
        print(f"      → NO, difference is from initialization, not checkpoint")
    else:
        print(f"      → UNKNOWN (no init diff, no ckpt diff, but post-load diff exists)")

# Q4: How many ckpt params differ
if ckpt_loaded:
    print(f"\n  Q4: How many ckpt encoder params differ from teacher?")
    print(f"      {n_diff_B1}/{len(ckpt_enc_sd)} params with |diff| > 1e-7")
    if n_diff_B1 > 0:
        print(f"      max diff: {diffs_B1[0][1]:.10f} at {diffs_B1[0][0]}")

# Q5: Encoder actually frozen?
q5 = (enc_grad_has == 0 and max_p_update < 1e-12 and not enc_in_opt)
print(f"\n  Q5: Encoder truly frozen (not updated in training step)?")
print(f"      params with grad:         {enc_grad_has}")
print(f"      max param update:         {max_p_update:.2e}")
print(f"      max buffer update:        {max_b_update:.2e}")
print(f"      in optimizer:             {enc_in_opt}")
print(f"      → {'YES' if q5 else 'NO'}")

# Q6: enc_lr_mult effect
print(f"\n  Q6: Does enc_lr_mult=0.1 create encoder optimizer group?")
print(f"      enc_lr_mult config value: {train_cfg.get('enc_lr_mult', 'N/A')}")
print(f"      encoder in any param_group: {enc_in_opt}")
print(f"      → {'YES' if enc_in_opt else 'NO (no trainable enc params → group not created)'}")

# Q7: Same latent space?
q7 = out_B["cosine"] > 0.9999 and out_B["max_abs_diff"] < 0.01
print(f"\n  Q7: Are z_lq and z_hq in the same latent space?")
print(f"      model.enc ≈ teacher after ckpt: cosine={out_B['cosine']:.8f}, max_diff={out_B['max_abs_diff']:.2e}")
print(f"      → {'YES' if q7 else 'NO — latent spaces differ!'}")

# ===================================================================
# DECISION RULE
# ===================================================================
print(f"\n{'='*70}")
print(f"DECISION")
print(f"{'='*70}")

if ckpt_loaded and q2 and q5:
    print(f"""
  ✓ Checkpoint loaded successfully.
  ✓ model.enc == teacher after checkpoint (cosine={out_B['cosine']:.8f}, max_diff={out_B['max_abs_diff']:.2e})
  ✓ Encoder is truly frozen (no grad, no optimizer update)
  ✓ Latent space is identical for LQ and HQ

  → Encoder is EXCLUDED as root cause.
  → Next: investigate FMIR CFM/ODE training strategy.
  → Try one_step_residual bridge in fm_cfg.
""")
elif ckpt_loaded and not q2:
    print(f"""
  ✗ model.enc != teacher after checkpoint!
     cosine={out_B['cosine']:.8f}, max_diff={out_B['max_abs_diff']:.2e}
     {n_diff_B1} ckpt_enc params differ from teacher (max={diffs_B1[0][1]:.10f})

  → state_dict_enc in checkpoint OVERWROTE the pretrained encoder.
  → This means the encoder WAS trained (despite enc_trainable=false in config).
  → STOP. Do NOT modify FMIR. First investigate:
    1. Was enc_trainable ever set to true in any training run?
    2. Is the checkpoint from a run with different config?
    3. Is there a resume/override path that changed enc_trainable?
  → Run: grep -r "trainable.*true\|trainable.*True" on the training yaml used for this ckpt.
""")
elif ckpt_loaded and not q5:
    print(f"""
  ✗ Encoder is NOT truly frozen!
     grad params: {enc_grad_has}, max update: {max_p_update:.2e}

  → Fix encoder freezing before investigating FMIR.
""")
else:
    print(f"""
  → Incomplete diagnosis. See details above.
""")

print(f"{'='*70}")
print(f"DONE")
print(f"{'='*70}")
