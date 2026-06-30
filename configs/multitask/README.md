# 通用图像复原多任务配置

每个任务共享同一套模型架构，仅数据集和任务相关超参不同。

## 架构（所有任务通用）

```
FMIR (LUnet): ch_mult=[1,2,2,2], hid=192, n_mid=4
MMSE (RRDBNet): c_hid=128, n_rrdb=4, overparam=true
Condition: HaarDWT -> CrossBandAttention -> WaveletStem (频率 -> FMIR)
          ConditionStem CNN (空间 -> Decoder)
双路分离注入，不做融合

TanhSFT 无时间门控 w_t
Encoder 冻结 (独立 teacher 编码 HQ)
损失: FM(Charb) + SSIM + 余弦颜色 + 模糊一致性
```

## 任务差异

| 参数 | 大数据 (RESIDE) | 小数据 (LOL) |
|---|---|---|
| use_attn | false | true |
| dino_align | false | true |
| max_steps | 300000 | 300000 |

## 训练

```bash
# 去雾
python train.py -y configs/multitask/train_reside_dehaze.yaml

# 低光
python train.py -y configs/multitask/train_lol_lowlight.yaml
```

## 评估

```bash
python eval.py -y configs/multitask/eval_reside_dehaze.yaml
python eval.py -y configs/multitask/eval_lol_lowlight.yaml
```

## 新增退化任务

复制任一 train config，修改 `dataset_cfg` 部分即可：

```yaml
dataset_cfg:
  train_dataset:
    name: YOUR_DATASET    # 需在 dataset.py 注册
    path: /path/to/train
    lq_subdir: ...
    hq_subdir: ...
  val_dataset:
    name: YOUR_DATASET
    path: /path/to/val
    lq_subdir: ...
    hq_subdir: ...
```
