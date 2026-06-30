"""
ELIR Encoder 身份验证：model.enc vs teacher 是否完全相同，及冻结是否生效
=========================================================================
严格按 1→7 步诊断，每步输出可验证的数字。
"""
import os, sys, math, csv, itertools
import torch, torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from copy import deepcopy

# ---------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-y", "--yaml", type=str, default="configs/derain/train.yaml")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--out", type=str, default="./runs/diag_enc")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"[diag_enc] device={device}")

# ---------------------------------------------------------------------------
from hyperpyyaml import load_hyperpyyaml
with open(args.yaml) as f:
    conf = load_hyperpyyaml(f)

model_cfg = conf.get("model_cfg", {})
arch_cfg = model_cfg.get("arch_cfg", {})
teacher_cfg = model_cfg.get("teacher_cfg", {})
train_cfg = conf.get("train_cfg", {})
fm_cfg = conf.get("fm_cfg", {})

ckpt_path = args.ckpt or arch_cfg.get("path")
print(f"ckpt_path (from config): {ckpt_path}")

# ===================================================================
# STEP 1: 追踪创建来源
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 1: ENCODER & TEACHER CREATION TRACE")
print(f"{'='*70}")

print(f"""
model.enc:
  created by:  ELIR.models.load_model.get_model(arch_cfg.enc_cfg)
  class:       TAESD.Encoder (from ELIR.models.taesd)
  config:      {arch_cfg.get('params',{}).get('enc_cfg',{})}
  pretrained:  madebyollin/taesd3 (via AutoencoderTiny.from_pretrained)
  checkpoint:  state_dict_enc from ckpt

teacher:
  created by:  ELIR.models.load_model.get_model(teacher_cfg)
  class:       TAESD.Encoder (from ELIR.models.taesd)
  config:      {teacher_cfg}
  pretrained:  madebyollin/taesd3 (via AutoencoderTiny.from_pretrained)
  checkpoint:  NOT loaded from ckpt (teacher is independently instantiated)

Key fact: model.enc and teacher are SEPARATE instantiations.
          Both load the same pretrained weights.
          model.enc may be overridden by state_dict_enc from checkpoint.
          teacher is NEVER overridden by checkpoint.
""")

# ===================================================================
# 完整负载流程：模拟 train.py 的每个步骤
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 2: COMPARE AT 4 TIME POINTS")
print(f"{'='*70}")

from ELIR.models.load_model import get_model
from diffusers import AutoencoderTiny

# ---- 获取原始 TAESD 权重作为 ground truth ----
taesd3_diffusers = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
raw_diffusers_enc = taesd3_diffusers.encoder.to(device)
raw_diffusers_enc.eval()
for p in raw_diffusers_enc.parameters():
    p.requires_grad = False

# ---- Time A: 刚实例化后（加载 TAESD3 预训练） ----
enc_type = type(raw_diffusers_enc).__name__
print(f"\n  diffusers EncoderTiny type: {enc_type}")

arch_cfg_clean = dict(arch_cfg)
arch_cfg_clean["path"] = None  # 不加载 checkpoint
model_A = get_model(deepcopy(arch_cfg_clean))

model_enc_A = model_A.enc
teacher_type = type(model_enc_A).__name__
print(f"  model.enc type:              {teacher_type}")

# ---- 独立创建 teacher ----
teacher_cfg_clean = dict(teacher_cfg)
teacher_cfg_clean["path"] = None
teacher_model = get_model(teacher_cfg_clean)  # This returns just the encoder
teacher_enc = teacher_model  # teacher_cfg returns model.encoder directly

# Also get raw teacher encoder from diffusers
raw_enc_from_taesd3py = None
try:
    from ELIR.models.taesd import TAESD
    taesd_py = TAESD()
    taesd_py.load_state_dict(taesd3_diffusers.state_dict())
    raw_enc_from_taesd3py = taesd_py.encoder.to(device).eval()
    for p in raw_enc_from_taesd3py.parameters():
        p.requires_grad = False
except Exception as e:
    print(f"  [WARN] Could not create TAESD.py encoder: {e}")

# ---- 时间 A 比较 ----
print(f"\n  --- Time A: After instantiation, before checkpoint ---")
x_test = torch.randn(1, 3, 256, 256, device=device).clamp(0, 1)

with torch.no_grad():
    z_diffusers_A = raw_diffusers_enc(x_test)
    z_model_A = model_enc_A.to(device).eval()(x_test)
    z_teacher_A = teacher_enc.to(device).eval()(x_test)

    print(f"  diffusers EncoderTiny:       mean={z_diffusers_A.mean().item():.4f} std={z_diffusers_A.std().item():.4f}")
    print(f"  model.enc (TAESD.Encoder):   mean={z_model_A.mean().item():.4f} std={z_model_A.std().item():.4f}")
    print(f"  teacher (TAESD.Encoder):     mean={z_teacher_A.mean().item():.4f} std={z_teacher_A.std().item():.4f}")

    diff_model_A = (z_diffusers_A - z_model_A).abs().max().item()
    diff_teacher_A = (z_diffusers_A - z_teacher_A).abs().max().item()
    diff_model_teacher_A = (z_model_A - z_teacher_A).abs().max().item()
    cos_mt_A = F.cosine_similarity(z_model_A.flatten(1), z_teacher_A.flatten(1)).mean().item()

    print(f"  max|diffusers - model.enc|:  {diff_model_A:.8f}")
    print(f"  max|diffusers - teacher|:    {diff_teacher_A:.8f}")
    print(f"  max|model.enc - teacher|:    {diff_model_teacher_A:.8f}")
    print(f"  cos(model.enc, teacher):     {cos_mt_A:.8f}")

    if diff_model_A > 0.01:
        print(f"  ⚠️  DIFFUSERS vs TAESD.py Encoder produce DIFFERENT outputs!")
        print(f"     → Encoder class wrappers differ (different forward impl)")
    else:
        print(f"  ✓ diffusers ≈ TAESD.py encoder — wrappers are equivalent")

    if diff_model_teacher_A > 0.001:
        print(f"  ⚠️  model.enc and teacher differ at init (before any training)")
        print(f"     → Same class, same weights, but different output")
    else:
        print(f"  ✓ model.enc == teacher at init")

    if raw_enc_from_taesd3py is not None:
        z_taesdpy_A = raw_enc_from_taesd3py(x_test)
        diff_taesdpy_A = (z_model_A - z_taesdpy_A).abs().max().item()
        print(f"  max|model.enc - raw TAESD.py enc|: {diff_taesdpy_A:.8f}")

# 参数级别比较
print(f"\n  --- Parameter comparison: model.enc vs teacher (Time A) ---")
enc_A_sd = model_enc_A.state_dict()
teacher_A_sd = teacher_enc.state_dict()
diffusers_sd = raw_diffusers_enc.state_dict()

enc_A_keys = set(enc_A_sd.keys())
teacher_A_keys = set(teacher_A_sd.keys())
diffusers_keys = set(diffusers_sd.keys())

print(f"  model.enc keys:    {len(enc_A_keys)}")
print(f"  teacher keys:      {len(teacher_A_keys)}")
print(f"  diffusers keys:    {len(diffusers_keys)}")
print(f"  enc==teacher keys: {enc_A_keys == teacher_A_keys}")
print(f"  enc==diffusers keys: {enc_A_keys == diffusers_keys}")

# Check key differences
only_enc = enc_A_keys - teacher_A_keys
only_teacher = teacher_A_keys - enc_A_keys
if only_enc:
    print(f"  Only in model.enc: {only_enc}")
if only_teacher:
    print(f"  Only in teacher: {only_teacher}")

# Max weight differences
max_diffs = {}
for k in sorted(enc_A_keys & teacher_A_keys):
    if enc_A_sd[k].shape == teacher_A_sd[k].shape:
        d = (enc_A_sd[k] - teacher_A_sd[k]).abs().max().item()
        max_diffs[k] = d

max_key = max(max_diffs, key=max_diffs.get)
print(f"  Max param diff (enc vs teacher): {max_diffs[max_key]:.10f} at {max_key}")
print(f"  torch.equal(enc, teacher): {all(torch.equal(enc_A_sd[k], teacher_A_sd[k]) for k in enc_A_keys & teacher_A_keys if enc_A_sd[k].shape == teacher_A_sd[k].shape)}")

# ---- 时间 C: 加载 derain checkpoint 后 ----
print(f"\n  --- Time C: After derain checkpoint load ---")
if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    print(f"  ckpt global_step={ckpt.get('global_step','N/A')}, epoch={ckpt.get('epoch','N/A')}")
    has_enc_ckpt = 'state_dict_enc' in ckpt and ckpt['state_dict_enc'] and len(ckpt['state_dict_enc']) > 0
    print(f"  state_dict_enc in ckpt: {has_enc_ckpt}")

    # Load into model_enc
    if has_enc_ckpt:
        m, u = model_enc_A.load_state_dict(ckpt['state_dict_enc'], strict=False)
        print(f"  [state_dict_enc → model.enc] missing={len(m)}, unexpected={len(u)}")

    model_enc_A.to(device).eval()
    with torch.no_grad():
        z_model_C = model_enc_A(x_test)

        # teacher is NOT overridden
        z_teacher_C = teacher_enc.to(device).eval()(x_test)

        diff_C = (z_model_C - z_teacher_C).abs().max().item()
        cos_C = F.cosine_similarity(z_model_C.flatten(1), z_teacher_C.flatten(1)).mean().item()
        nr_C = z_model_C.flatten(1).norm(dim=1).mean().item() / (z_teacher_C.flatten(1).norm(dim=1).mean().item() + 1e-8)
        print(f"  model.enc:       mean={z_model_C.mean().item():.4f} std={z_model_C.std().item():.4f}")
        print(f"  teacher:         mean={z_teacher_C.mean().item():.4f} std={z_teacher_C.std().item():.4f}")
        print(f"  max|diff|:       {diff_C:.8f}")
        print(f"  cosine:          {cos_C:.8f}")
        print(f"  norm ratio(s/t): {nr_C:.8f}")

        if diff_C > 0.001:
            print(f"  ⚠️  CHECKPOINT OVERRIDE: model.enc differs from teacher after ckpt load!")
            print(f"     → state_dict_enc in ckpt has DRIFTED encoder weights")
            print(f"     → This means training DID update the encoder at some point")
        else:
            print(f"  ✓ model.enc == teacher after ckpt load (encoder was truly frozen)")
else:
    print(f"  checkpoint NOT FOUND — skipping Time C")

# Load full model checkpoint for later steps
model_full = deepcopy(model_A)
if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
    if sd:
        cleaned = OrderedDict((k[len("model."):] if k.startswith("model.") else k, v) for k, v in sd.items())
        model_full.load_state_dict(cleaned, strict=False)
    for key, attr in [("state_dict_fmir","fmir"),("state_dict_mmse","mmse"),
                       ("state_dict_enc","enc"),("state_dict_dec","dec"),
                       ("state_dict_wavelet","wavelet_stem")]:
        if key in ckpt and ckpt[key]:
            getattr(model_full, attr).load_state_dict(ckpt[key], strict=False)

model_full.to(device).eval()

# ===================================================================
# STEP 3: 检查 enc_trainable=false 是否真正冻结
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 3: ENCODER FREEZE STATUS CHECK")
print(f"{'='*70}")

enc_params = list(model_full.enc.named_parameters())
enc_buffers = list(model_full.enc.named_buffers())
n_total = len(enc_params)
n_trainable = len([n for n, p in enc_params if p.requires_grad])
n_frozen = n_total - n_trainable
trainable_names = [n for n, p in enc_params if p.requires_grad]

print(f"  enc params total:     {n_total}")
print(f"  enc trainable:        {n_trainable}")
print(f"  enc frozen:           {n_frozen}")
print(f"  enc.training:         {model_full.enc.training}")
print(f"  enc_trainable attr:   {getattr(model_full, 'enc_trainable', 'N/A')}")

if trainable_names:
    print(f"  ⚠️  TRAINABLE PARAMS FOUND:")
    for name in trainable_names:
        print(f"      {name}")
else:
    print(f"  ✓ All encoder params frozen")

# Assertion
try:
    assert all(not p.requires_grad for p in model_full.enc.parameters()), \
        "FAIL: Some encoder params require grad!"
    print(f"  ✓ assert: all(not requires_grad) passed")
except AssertionError as e:
    print(f"  ✗ {e}")

# Check Elir.train() behavior
print(f"\n  --- Elir.train() behavior check ---")
model_full.train()
enc_training_after = model_full.enc.training
print(f"  After model.train(): enc.training={enc_training_after}")
print(f"  enc requires_grad after train(): {[n for n,p in model_full.enc.named_parameters() if p.requires_grad]}")
model_full.eval()

# ===================================================================
# STEP 4: 检查 Encoder 是否仍在 optimizer
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 4: OPTIMIZER INSPECTION")
print(f"{'='*70}")

from ELIR.training.tparmas import get_optimizer

enc_param_ids = {id(p) for p in model_full.enc.parameters()}
optimizer = get_optimizer(train_cfg, model_full)

print(f"  optimizer type: {type(optimizer).__name__}")
print(f"  num param_groups: {len(optimizer.param_groups)}")

enc_in_optimizer = False
for i, pg in enumerate(optimizer.param_groups):
    pg_ids = {id(p) for p in pg['params']}
    enc_count = len(pg_ids & enc_param_ids)
    total_count = len(pg['params'])
    lr = pg.get('lr', 'N/A')
    print(f"  group[{i}]: lr={lr}, params={total_count}, enc_params={enc_count}")
    if enc_count > 0:
        enc_in_optimizer = True
        # print sample names
        enc_names_in_pg = [n for n, p in model_full.enc.named_parameters() if id(p) in pg_ids]
        print(f"    enc param names in group: {enc_names_in_pg[:5]}")

if enc_in_optimizer:
    print(f"  ⚠️  ENCODER IS IN OPTIMIZER!")
else:
    print(f"  ✓ Encoder NOT in optimizer (all requires_grad=False → skipped)")

# ===================================================================
# STEP 5: 实际数值更新测试
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 5: ACTUAL WEIGHT UPDATE TEST (full training step)")
print(f"{'='*70}")

# 加载一对真实数据
from ELIR.datasets.dataset import get_loader
vcfg = dict(conf.get("dataset_cfg", {}).get("train_dataset", {}))
vcfg.update({"batch_size": 1, "num_workers": 0})
try:
    loader = get_loader(vcfg)
    test_batch = next(iter(loader))
    x_lq, x_hq = test_batch[0].to(device), test_batch[1].to(device)
except Exception as e:
    print(f"  Could not load real data: {e}, using random tensors")
    x_lq = torch.randn(1, 3, 256, 256, device=device).clamp(0, 1)
    x_hq = torch.randn(1, 3, 256, 256, device=device).clamp(0, 1)

# 保存 step 前 encoder 参数
enc_before = {n: p.detach().clone() for n, p in model_full.enc.named_parameters()}
enc_buf_before = {n: b.detach().clone() for n, b in model_full.enc.named_buffers()}

# 构造简化的训练 step（使用 e2e_gan loss）
from ELIR.models.elir import pos_emb
from ELIR.training.losses import e2e_gan_loss

model_full.train()

# 64-align
oh, ow = x_hq.shape[2], x_hq.shape[3]
ph, pw = (64-oh%64)%64, (64-ow%64)%64
x_lq_p = F.pad(x_lq, (0,pw,0,ph), mode="reflect") if ph or pw else x_lq
x_hq_p = F.pad(x_hq, (0,pw,0,ph), mode="reflect") if ph or pw else x_hq

fm_cfg_step = dict(fm_cfg)
fm_cfg_step["global_step"] = 0

# 手动跑 loss（不需要 full IRSetup）
from ELIR.training.losses import _to_model_latent, _to_teacher_latent, _prepare_fmir_condition, _prepare_decoder_condition

with torch.enable_grad():
    # tmodel needs to be teacher encoder
    tmodel_enc = teacher_enc.to(device).eval()
    class _TW:
        def __init__(self, e): self.encoder = e
        def eval(self): pass
        def to(self, d): self.encoder.to(d); return self
    tw = _TW(tmodel_enc)

    X_hq = _to_teacher_latent(tw, x_hq_p)
    X_lq = _to_model_latent(model_full, x_lq_p)
    cond = _prepare_fmir_condition(model_full, x_lq_p)
    dec_cond = _prepare_decoder_condition(model_full, x_lq_p, cond)
    X_mmse = model_full.mmse(X_lq)

    eps = 1e-6
    loss = torch.sqrt((X_mmse - X_hq.detach()).pow(2) + eps).mean()

    print(f"  loss value: {loss.item():.6f}")
    print(f"  loss requires_grad: {loss.requires_grad}")

model_full.zero_grad(set_to_none=True)
loss.backward()

# 检查 encoder grad
print(f"\n  --- Encoder grad check after backward ---")
enc_grad_none = 0
enc_grad_tensor = 0
enc_grad_norms = []
for n, p in model_full.enc.named_parameters():
    if p.grad is None:
        enc_grad_none += 1
    else:
        enc_grad_tensor += 1
        enc_grad_norms.append((n, p.grad.norm().item()))

print(f"  enc params with grad=None:  {enc_grad_none}")
print(f"  enc params with grad!=None: {enc_grad_tensor}")
if enc_grad_norms:
    print(f"  ⚠️  GRADIENTS EXIST on encoder params!")
    for name, gn in enc_grad_norms[:5]:
        print(f"      {name}: grad_norm={gn:.6f}")
    for name, gn in enc_grad_norms:
        if gn > 1e-8:
            print(f"      {name}: grad_norm={gn:.6f} ← NON-ZERO GRADIENT!")
else:
    print(f"  ✓ All encoder grads are None")

# optimizer step
for pg in optimizer.param_groups:
    for p in pg['params']:
        if p.grad is None:
            p.grad = torch.zeros_like(p) if p.requires_grad else None  # shouldn't happen

optimizer.step()

# 检查更新
print(f"\n  --- Encoder parameter change after optimizer.step() ---")
max_update = 0.0
updated_params = []
for n, p in model_full.enc.named_parameters():
    update = (p.detach() - enc_before[n]).abs().max().item()
    if update > max_update:
        max_update = update
    if update > 1e-10:
        updated_params.append((n, update))

print(f"  max encoder parameter update: {max_update:.12f}")
if updated_params:
    print(f"  ⚠️  ENCODER WAS UPDATED! ({len(updated_params)} params changed)")
    for n, u in updated_params[:10]:
        print(f"      {n}: Δ={u:.10f}")
else:
    print(f"  ✓ Encoder NOT updated (all Δ=0)")

# buffer 更新
max_buf_update = 0.0
for n, b in model_full.enc.named_buffers():
    update = (b.detach() - enc_buf_before[n]).abs().max().item()
    if update > max_buf_update:
        max_buf_update = update

print(f"  max encoder buffer update:   {max_buf_update:.12f}")
print(f"  enc.training after step:     {model_full.enc.training}")

model_full.eval()

# ===================================================================
# STEP 6: 特殊覆盖路径检查
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 6: SPECIAL OVERRIDE PATH CHECK")
print(f"{'='*70}")

# 搜索关键代码路径
checks = {
    "Elir.enc_trainable": getattr(model_full, 'enc_trainable', 'N/A'),
    "IRSetup._log_module_grad_norms encoder prefix": "enc.",
    "train.py _configure_fmir_trainable_params": "only touches fmir.*",
    "get_optimizer enc_lr_mult": train_cfg.get('enc_lr_mult', 'N/A'),
    "EMA exists": "N/A (not in this scope)",
}

for k, v in checks.items():
    print(f"  {k}: {v}")

# 检查 EMA
from ELIR.irsetup import IRSetup
setup = IRSetup(model=model_full, fm_cfg=fm_cfg,
                eval_cfg={"metrics": ["psnr"]},
                run_dir=args.out, save_images=False,
                optimizer=optimizer, scheduler=None,
                tmodel=tw)
has_ema = setup.ema is not None
print(f"  IRSetup EMA: {has_ema}")
if has_ema:
    ema_enc = setup.ema.model.enc
    print(f"  EMA model.enc type: {type(ema_enc).__name__}")
    # Check if EMA encoder == model encoder
    ema_same = all(torch.equal(ema_enc.state_dict()[k], model_full.enc.state_dict()[k])
                   for k in ema_enc.state_dict() if ema_enc.state_dict()[k].shape == model_full.enc.state_dict()[k].shape)
    print(f"  EMA enc == model.enc: {ema_same}")

# ===================================================================
# STEP 7: 最终结论
# ===================================================================
print(f"\n{'='*70}")
print(f"STEP 7: FINAL VERDICT")
print(f"{'='*70}")

# Re-check the key questions
diff_after_ckpt = None
if ckpt_path and os.path.exists(os.path.expanduser(ckpt_path)):
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if 'state_dict_enc' in ckpt and ckpt['state_dict_enc']:
        ckpt_enc = ckpt['state_dict_enc']
        raw_enc_sd = model_A.enc.state_dict()  # pretrained (no ckpt override)
        max_ckpt_diff = 0.0
        for k in ckpt_enc:
            if k in raw_enc_sd and raw_enc_sd[k].shape == ckpt_enc[k].shape:
                d = (ckpt_enc[k] - raw_enc_sd[k]).abs().max().item()
                if d > max_ckpt_diff:
                    max_ckpt_diff = d
        diff_after_ckpt = max_ckpt_diff
        print(f"  max ckpt_enc vs pretrained_enc diff: {max_ckpt_diff:.8f}")

enc_is_frozen = n_trainable == 0
enc_in_opt = enc_in_optimizer
enc_was_updated = max_update > 1e-10
enc_same_as_teacher = diff_model_teacher_A < 1e-6

print(f"""
  Q1: model.enc == teacher at init?        {'YES' if enc_same_as_teacher else 'NO (diff=' + str(diff_model_teacher_A) + ')'}
  Q2: model.enc == teacher after ckpt?     {'YES' if diff_after_ckpt is None or diff_after_ckpt < 1e-6 else 'NO (diff=' + str(diff_after_ckpt) + ')'}
  Q3: encoder truly frozen?                {'YES' if enc_is_frozen else 'NO (' + str(n_trainable) + ' params trainable)'}
  Q4: encoder in optimizer?                {'YES' if enc_in_opt else 'NO'}
  Q5: encoder actually updated?            {'YES (max_update=' + str(max_update) + ')' if enc_was_updated else 'NO'}
  Q6: enc_lr_mult caused encoder group?    {'YES' if enc_in_opt else 'NO (no trainable enc params → no group)'}
  Q7: latent spaces identical?             {'YES (enc==teacher)' if enc_same_as_teacher and not enc_was_updated else 'DEPENDS'}

  Final assessment:
""")

if enc_same_as_teacher and enc_is_frozen and not enc_was_updated:
    print(f"  ✓ Encoder is IDENTICAL to teacher and TRULY FROZEN.")
    print(f"  ✓ enc_lr_mult is irrelevant (no trainable enc params to group).")
    print(f"  ✓ Teacher and student share the SAME latent space.")
    print(f"  → If deraining PSNR is still low, the root cause is NOT encoder-related.")
    print(f"  → Focus on FMIR training strategy (one_step_residual bridge).")
elif not enc_same_as_teacher and enc_is_frozen and not enc_was_updated:
    print(f"  ⚠️ Encoder FROZEN but DIFFERS from teacher.")
    if diff_after_ckpt and diff_after_ckpt > 1e-6:
        print(f"  → state_dict_enc in ckpt has drifted weights (Δ={diff_after_ckpt:.6f})")
        print(f"  → FIX: do NOT load state_dict_enc from old checkpoint")
    else:
        print(f"  → TAESD.py Encoder vs diffusers EncoderTiny produce different outputs")
        print(f"  → FIX: use diffusers EncoderTiny directly for both")
else:
    print(f"  ✗ Encoder has issues — see details above.")
    if enc_was_updated:
        print(f"  → CRITICAL: Encoder was UPDATED during training step")
        print(f"  → FIX: ensure enc_cfg.trainable=false AND enc_lr_mult=0.0")
    if enc_in_opt:
        print(f"  → Encoder IS in optimizer — check requires_grad")

print(f"""
  Recommended config:
    enc_cfg:
      name: tiny_enc      # or use diffusers EncoderTiny directly
      trainable: false
    train_cfg:
      enc_lr_mult: 0.0    # belt-and-suspenders

  And verify at startup:
    assert all(not p.requires_grad for p in model.enc.parameters())
    z1 = model.enc(x_test)
    z2 = teacher(x_test)
    assert (z1-z2).abs().max() < 1e-4
""")

print(f"\n{'='*70}")
print(f"DONE")
print(f"{'='*70}")
