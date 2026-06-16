#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Google TimesFM 2.5 对兰州 2022-2024 负荷数据进行零样本预测。
- 输入：2024-12-01 之前的历史数据
- 预测：2024-12-01 起未来 10 天（240 小时）的 NEXTLOAD
- 对比：与数据中对应日期的实测 NEXTLOAD 进行精度评估
- 主变量：LOAD
- 动态数值协变量：WBGT、ICHB、ET、THI、CHI

运行前请确保已安装依赖（示例）：
    pip install jax[cpu] scikit-learn pandas openpyxl matplotlib
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install timesfm -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# 国内访问 HuggingFace 较慢，优先使用镜像端点
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 在 Windows 上优先使用支持中文的黑体，避免图表标签乱码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
COVARIATES = ["WBGT", "ICHB", "ET", "THI", "CHI"]
TARGET_VAR = "LOAD"          # 用于 TimesFM 输入的历史负荷列
HORIZON_HOURS = 10 * 24      # 未来 10 天 = 240 小时
MAX_CONTEXT = 8192           # TimesFM 2.5 最大支持 16384，这里用 8192 平衡速度与效果
MAX_HORIZON = 256            # 必须 >= HORIZON_HOURS
SPLIT_DATE = pd.Timestamp("2024-06-25")


def main() -> int:
    parser = argparse.ArgumentParser(description="TimesFM 负荷预测")
    parser.add_argument(
        "--input",
        default="data/lanzhouNew2022-2024.xlsx",
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="输出目录",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[错误] 找不到输入文件: {input_path.resolve()}")
        return 1

    # ---------------------------------------------------------------------
    # 1) 读取数据并按日期切分
    # ---------------------------------------------------------------------
    print(f"[1/7] 读取数据: {input_path}")
    df = pd.read_excel(input_path)
    required_cols = ["DATATIME", TARGET_VAR, "LOAD"] + COVARIATES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[错误] 缺少列: {missing}")
        return 1

    df = df.sort_values("DATATIME").reset_index(drop=True)
    df["DATATIME"] = pd.to_datetime(df["DATATIME"])
    print(f"       时间范围: {df['DATATIME'].min()} ~ {df['DATATIME'].max()}")
    print(f"       总行数: {len(df)}")

    context_df = df[df["DATATIME"] < SPLIT_DATE].copy()
    horizon_end = SPLIT_DATE + pd.Timedelta(hours=HORIZON_HOURS)
    horizon_df = df[
        (df["DATATIME"] >= SPLIT_DATE) & (df["DATATIME"] < horizon_end)
    ].copy()

    if len(context_df) == 0:
        print("[错误] 2024-12-01 之前无数据，无法构建上下文。")
        return 1
    if len(horizon_df) < HORIZON_HOURS:
        print(
            f"[错误] 2024-12-01 起未来 10 天数据不足，"
            f"实际只有 {len(horizon_df)} 条，需要 {HORIZON_HOURS} 条。"
        )
        return 1

    print(
        f"       上下文: {context_df['DATATIME'].min()} ~ {context_df['DATATIME'].max()} "
        f"({len(context_df)} 条)"
    )
    print(
        f"       预测期: {horizon_df['DATATIME'].min()} ~ {horizon_df['DATATIME'].max()} "
        f"({len(horizon_df)} 条)"
    )

    # ---------------------------------------------------------------------
    # 2) 构造输入序列与协变量
    # ---------------------------------------------------------------------
    print("[2/7] 构造 LOAD 上下文序列与协变量...")
    load_context = context_df[TARGET_VAR].iloc[-MAX_CONTEXT:].to_numpy(
        dtype=np.float32
    )
    context_len = len(load_context)
    print(f"       实际使用上下文长度: {context_len}")

    dynamic_numerical_covariates: dict[str, Sequence[Sequence[float]]] = {}
    for cov in COVARIATES:
        cov_context = context_df[cov].iloc[-context_len:].to_numpy(dtype=np.float32)
        cov_horizon = horizon_df[cov].to_numpy(dtype=np.float32)
        arr = np.concatenate([cov_context, cov_horizon])
        dynamic_numerical_covariates[cov] = [arr.tolist()]
        print(
            f"       {cov}: context={len(cov_context)}, horizon={len(cov_horizon)}, "
            f"total={len(arr)}"
        )

    # ---------------------------------------------------------------------
    # 3) 加载 TimesFM 2.5 模型
    # ---------------------------------------------------------------------
    print("[3/7] 加载 TimesFM 2.5 模型（首次会从 HuggingFace 下载约 800MB 权重）...")
    import timesfm

    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"       使用设备: {device}")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True,  # XReg / 协变量预测必须开启
        )
    )

    # ---------------------------------------------------------------------
    # 4) 执行带协变量的预测
    # ---------------------------------------------------------------------
    print("[4/7] 执行 forecast_with_covariates ...")
    point_forecast, quantile_forecast = model.forecast_with_covariates(
        inputs=[load_context.tolist()],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        dynamic_categorical_covariates={},
        static_numerical_covariates={},
        static_categorical_covariates={},
        xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True,
        ridge=0.0,
        force_on_cpu=(device == "cpu"),
    )
    # forecast_with_covariates 返回的是 list[np.ndarray]；本例只有单条序列
    pred = point_forecast[0]
    lower_80 = quantile_forecast[0][:, 1]   # q10
    upper_80 = quantile_forecast[0][:, 9]   # q90
    mean_fc = quantile_forecast[0][:, 0]    # mean

    print(f"       预测长度: {len(pred)} 小时")

    # ---------------------------------------------------------------------
    # 5) 提取实测值并计算误差指标
    # ---------------------------------------------------------------------
    print("[5/7] 与实测 LOAD 对比...")
    actual = horizon_df["LOAD"].to_numpy(dtype=np.float32)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    coverage = float(
        np.mean((actual >= lower_80) & (actual <= upper_80)) * 100
    )

    print(f"       MAE : {mae:.3f}")
    print(f"       RMSE: {rmse:.3f}")
    print(f"       MAPE: {mape:.2f}%")
    print(f"       80% PI Coverage: {coverage:.1f}%")

    # ---------------------------------------------------------------------
    # 6) 保存结果
    # ---------------------------------------------------------------------
    print("[6/7] 保存预测结果...")
    future_dt = horizon_df["DATATIME"].values

    result_df = pd.DataFrame(
        {
            "DATATIME": future_dt,
            "LOAD_ACTUAL": np.round(actual, 3),
            "LOAD_PRED": np.round(pred, 3),
            "LOAD_MEAN": np.round(mean_fc, 3),
            "LOAD_Q10": np.round(lower_80, 3),
            "LOAD_Q90": np.round(upper_80, 3),
        }
    )

    csv_path = output_dir / "load_forecast_10days.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"       CSV 已保存: {csv_path.resolve()}")

    # ---------------------------------------------------------------------
    # 7) 可视化
    # ---------------------------------------------------------------------
    print("[7/7] 绘制可视化图表...")
    # 取上下文最后 240 小时作为图的左侧历史
    plot_context_hours = 240
    plot_context = (
        context_df[TARGET_VAR].iloc[-plot_context_hours:].to_numpy(dtype=np.float32)
    )
    plot_context_dt = context_df["DATATIME"].iloc[-plot_context_hours:]
    split_line = horizon_df["DATATIME"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        plot_context_dt,
        plot_context,
        label="历史 LOAD",
        color="steelblue",
        linewidth=1.5,
    )
    ax.plot(
        future_dt,
        pred,
        label="LOAD 预测 (median)",
        color="tab:orange",
        linewidth=2,
    )
    ax.plot(
        future_dt,
        actual,
        label="LOAD 实测",
        color="tab:green",
        linewidth=1.5,
        linestyle="--",
        marker="o",
        markersize=2,
    )
    ax.fill_between(
        future_dt,
        lower_80,
        upper_80,
        alpha=0.25,
        color="tab:orange",
        label="80% 预测区间 (q10-q90)",
    )
    ax.axvline(split_line, color="gray", linestyle="--", linewidth=1)
    ax.set_title(
        "兰州负荷 2024-06-25 起未来 10 天预测（TimesFM 2.5 + 协变量）\n"
        f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  Coverage={coverage:.1f}%"
    )
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷 (MW)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / "load_forecast_10days.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"       图片已保存: {png_path.resolve()}")

    print("\n[完成] 预测与对比结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
