# ELIR 项目上下文 — GPT Projects 导入文件

## 项目简介

ELIR (Encoder-based Low-light Image Restoration) 是一个基于流匹配 (Flow Matching) 的通用图像复原框架。当前阶段聚焦低光增强 (LOL) 和去雾 (RESIDE) 两个任务，最终目标为通用多退化联合复原。

## 当前架构 (最终版本)

```
LQ → TAESD_enc (冻结) → z_lq [B,16,32,32]
                           ├→ MMSE (RRDBNet, 53M) → z_mmse
                           │      └→ L_align = Charb(z_mmse, V_fused)  ← MMSE 唯一监督
                           │           V_fused = SKFusion(TAESD_enc(HQ), DinoSpatialProj(DINOv2(HQ)))
                           │           HQ 侧冻结, SK 模块仅 ~12K 参数, bias=[3,-3]
                           │
                           ├→ z_mmse+noise → FMIR ODE(K=5) → z_rest → Decoder+SFT → ŷ
                           │      ↑                                      ↑
                           │  频率金字塔 (静态全景)                   空间金字塔 (静态)
                           │  LQ→HaarDWT→CrossBandAttention          LQ→ConditionStem(CNN)
                           │     →cat→WaveletStem→FMIR SFT             →Decoder SFT
                           │
                           └→ FM LOSS: L_cfm + L_end (X_hq 定义 ODE 轨迹)
```

**双路分离**：FMIR 只收频率条件，Decoder 只收空间条件。CondFusion1x1 融合已废弃。
**TanhSFT**：时间系数 w_t 已移除。时间调制保留在 ResBlock.time_mlp(t_emb)。

## 核心文件

| 文件 | 作用 |
|---|---|
| `ELIR/models/elir.py` | 主模型 Elir 类, 包含双路条件构建, ODE 循环 |
| `ELIR/models/lunet.py` | FMIR (LUnet) + AttnBlock + DwtSkipEnhance + TimeGatedDilatedConv |
| `ELIR/models/wavelet_stem.py` | HaarDWT + CrossBandAttention + TimeBandWeight + WaveletStem |
| `ELIR/training/losses.py` | e2e_gan_loss 函数, 含 FM/CFM/SK/pixel 全部损失 |
| `ELIR/training/dino_align.py` | DINOv2Encoder + DINOProjector + DinoSpatialProjector + SKFusion |
| `ELIR/training/perceptual.py` | LPIPS + VGG 回退 |
| `ELIR/models/gan.py` | NLayerDiscriminator + Hinge loss |
| `ELIR/irsetup.py` | Lightning 训练设置, 手动优化 G+D, Cosine LR |
| `ELIR/training/tparmas.py` | 优化器(含 enc_lr_mult 参数分组) |
| `ELIR/metrics.py` | PSNR/SSIM (DiffUIR MATLAB 兼容, Y通道) |
| `train.py` | 训练入口, train_mode 切换 (e2e/stage1/sft) |
| `eval.py` | 评估入口 |

## 实验结论

### 模块有效性

| 模块 | LOL | RESIDE | 判决 |
|---|---|---|---|
| CrossBandAttention | +? (全训练) | +2.44 dB | **有效** |
| AttnBlock (瓶颈注意力) | +0.38 (过拟合) | -1.03 | 仅小数据有效 |
| DWT Skip (浅层频带) | +0.08 (1000步) | 未测 | 中性辅助 |
| SK Fusion (TASED+DINO) | 待重新验证 | 未测 | 当前 bias 锁死 |
| TimeGatedDilatedConv | -0.35~-1.28 | -0.40 | **无效** |
| Encoder 可训练 | ~0 | ~0 | **小数据禁用** |
| GroupNorm (Encoder) | ~0 | ~0 | **禁用** |
| TimeBandWeight | 逻辑矛盾 | — | **废弃** |

### 最佳结果

| 任务 | PSNR | SSIM | 关键配置 |
|---|---|---|---|
| RESIDE 去雾 | 31.68 | 0.981 | CrossBandAttention, K=10推理 |
| LOL 低光 | 25.13 | 0.910 | Attn+DINO+CBand+K=15+TTA, 从头训30万步 |

### 最佳配置 (LOL)

```yaml
fmir: ch_mult=[1,2,2,2], hid=192, n_mid=4
mmse: c_hid=128, n_rrdb=4, overparam=true (53M)
enc_trainable: false
use_attn: true, attn_heads: 8
band_attn: true, fmir_use_wavelet_cond: true
use_time_dilate: false
K_train: 5, K_infer: 15, TTA: true
损失: Charb+SSIM+Color+Blur, lambda_gan=0, lambda_perc=0
```

## 当前待解决问题

1. **TanhSFT 去掉 w_t 导致性能下降**：验证为 -1.5 dB (200步过拟合)。是否回退待定。
2. **SK Fusion bias=[3,-3] 锁死不训练**：loss_dino 全程不变，无法自适应融合。
3. **去掉 L_mmse 后 MMSE 单一监督源强度不够**：PSNR 先涨到23再跌到18。
4. **架构到 25.13 的差距主要在哪**：待确认是 TanhSFT w_t、还是拆路、还是两者的叠加。
5. **30万步过拟合 (LOL)**：最佳在20k步出现，需要早停。

## 过拟合快速验证方法

```bash
# 200步, 12分钟, patch_size=128
python train.py -y configs/overfit_tests/baseline.yaml
# baseline 1000步 ~18.34, 200步 ~15.8 (新代码)
# 显著差异阈值为 >0.5 dB
```

## 开发规范

1. 本地代码和服务器代码必须同步——修改后 `git push` + 服务器 `git pull`
2. 模块改动前先跑过拟合验证 (200/1000步)
3. 一次只改一个变量, 做 A/B 对比
4. `.pyc` 缓存问题是常见错误源: `find . -name "*.pyc" -delete`
5. 服务器网络不可用时 DINOv2 会自动回退本地缓存加载
6. Config 现在集中在 `configs/` 根目录 + `configs/overfit_tests/` + `configs/multitask/`
7. 评估指标使用 DiffUIR MATLAB 兼容 PSNR/SSIM (Y通道, 8-bit量化)

## 继续对话所需文件

上传以下文件到 GPT Projects 即可无缝继续：

### 必需
- `ELIR/models/elir.py`
- `ELIR/models/lunet.py`
- `ELIR/models/wavelet_stem.py`
- `ELIR/training/losses.py`
- `ELIR/training/dino_align.py`
- `ELIR/irsetup.py`
- `train.py`
- `TRAINING_PIPELINE.md`

### 推荐
- `configs/elir_train_e2e_large_lol.yaml`
- `configs/overfit_tests/baseline.yaml`
- `ELIR/metrics.py`
- `ELIR/models/load_model.py`
- `ELIR/training/tparmas.py`
- `ELIR/training/perceptual.py`
- `ELIR/models/gan.py`
