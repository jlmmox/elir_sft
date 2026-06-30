#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV 指标可视化脚本（只在 CSV 同级目录生成图片）

功能：
- 扫描指定 run 目录或 logs 根目录，找到其中的 metrics.csv 并绘制曲线。
- 仅将 PNG 保存到 metrics.csv 所在目录（与 CSV 同级）。

识别规则：
- step 列：支持 "step"、"global_step"、"steps"（大小写不敏感）。
- 指标列（若存在就绘制，大小写不敏感）：
  - train/loss、val/loss、val/fid、val/ssim、val/psnr、val/mse、train/grad_norm
- 学习率列：任意列名包含 "lr"（如 "lr-Adam"、"lr_0"）。
- 梯度相关列：任意列名包含 "grad"（如 "grad"、"grad_norm"）。

使用（Windows 示例）：
- 仅处理某个 run：
    python plot_metrics_csv.py --run "C:\\Users\\moxt\\OneDrive\\Desktop\\cfm实验\\rainy\\2"
    python plot_metrics_csv.py --run "C:\\Users\\moxt\\OneDrive\\Desktop\elir\\elir_sftunet_ED\\lol_baseline\\trainm"
- 扫描 logs 根目录（默认 logs）：
   python plot_metrics_csv.py --logs-dir "C:\\Users\\moxt\\OneDrive\\Desktop\\cfm实验\\elirruns\\test"   

依赖：pandas、matplotlib
    pip install pandas matplotlib
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# 常见候选指标（使用小写以便大小写无关匹配）
CANDIDATE_COLUMNS = [
    "train/loss",
    "train/loss_step",
    "train/ssim_epoch",
    "train/psnr_epoch",
    "val/loss",
    "val/fid",
    "val/ssim",
    "val/psnr",
    "val/mse",
    "train/grad_norm",
    "loss/fm",
    "loss/endpoint_raw",
    "loss/endpoint_weighted",
    "loss/total",
    "test/ssim",
    "test/psnr",
    "test/mse",
    "test/fid",
    "test/loss",
    "audit/cos_sim",
    "audit/mag_ratio",
    "train_loss",
    "psnr",
    "ssim",
    "fid",
]

def find_runs(logs_dir: str) -> List[Path]:
    """查找包含 metrics.csv 的 run 目录"""
    p = Path(logs_dir)
    if not p.exists():
        return []
    runs: List[Path] = []
    for sub in sorted(p.iterdir()):
        if sub.is_dir():
            if (sub / "metrics.csv").exists():
                runs.append(sub)
    return runs

def _detect_step_col(df: pd.DataFrame) -> str:
    """在 DataFrame 中检测 step 列，返回实际列名（保持原大小写）。"""
    candidates = ["global_step", "step"]
    for want in candidates:
        for col in df.columns:
            if col.lower() == want:
                return col
    return None

def _plot_and_save(df: pd.DataFrame, step_col: str, metric_col: str, out_dir: Path):
    """将指定列绘制为曲线并保存到 out_dir。"""
    try:
        s_step = pd.to_numeric(df[step_col], errors="coerce")
        s_val = pd.to_numeric(df[metric_col], errors="coerce")
        mask = s_step.notna() & s_val.notna()
        if mask.sum() == 0:
            print(f"[跳过空列] {metric_col}")
            return
        x = s_step[mask]
        y = s_val[mask]
        order = x.argsort()
        x = x.iloc[order]
        y = y.iloc[order]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, label=metric_col, linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel(metric_col)
        ax.set_title(metric_col)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fname = f"{metric_col.replace('/', '_')}_curve.png"
        fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
    except Exception as e:
        print(f"[错误] 绘制失败 {metric_col}: {e}")
    finally:
        plt.close("all")

def plot_run(csv_path: Path, run_dir: Path):
    """绘制单个 run（metrics.csv 所在目录）的所有可用指标。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[错误] 读取失败: {csv_path}: {e}")
        return

    step_col = _detect_step_col(df)
    if not step_col:
        print(f"[跳过] 未找到 step 列: {csv_path}")
        return

    # 输出目录：CSV 同级目录
    out_dir = run_dir

    # 识别学习率和梯度相关列
    lr_cols = [c for c in df.columns if "lr" in c.lower()]
    grad_cols = [c for c in df.columns if "grad" in c.lower()]

    plotted_any = False

    # 候选指标（大小写无关匹配）
    for want in CANDIDATE_COLUMNS:
        matched = None
        for col in df.columns:
            if col.lower() == want:
                matched = col
                break
        if matched is None:
            continue
        plotted_any = True
        _plot_and_save(df, step_col, matched, out_dir)

    # 学习率列
    for col in lr_cols:
        plotted_any = True
        _plot_and_save(df, step_col, col, out_dir)

    # 梯度相关列
    for col in grad_cols:
        plotted_any = True
        _plot_and_save(df, step_col, col, out_dir)

    # 若仍未绘制，尝试所有 val/* 列
    if not plotted_any:
        val_cols = [c for c in df.columns if c.lower().startswith("val/")]
        for col in val_cols:
            plotted_any = True
            _plot_and_save(df, step_col, col, out_dir)

    if plotted_any:
        print(f"[完成] 生成图像于: {out_dir}")
    else:
        print(f"[提示] 无可绘制列: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="基于 metrics.csv 的指标可视化")
    parser.add_argument("--logs-dir", default="logs", help="logs 根目录（默认：logs）")
    parser.add_argument("--run", default=None, help="指定单个 run 目录（包含 metrics.csv）")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run).resolve()
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            print(f"[错误] 未找到 metrics.csv: {csv_path}")
            return
        plot_run(csv_path, run_dir)
        return

    runs = find_runs(args.logs_dir)
    if not runs:
        print(f"[提示] 未在 {args.logs_dir} 下找到任何 metrics.csv")
        return
    for run in runs:
        plot_run(run / "metrics.csv", run)

if __name__ == "__main__":
    main()