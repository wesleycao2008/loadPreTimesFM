#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Amazon Chronos-2 对兰州 2022-2024 负荷数据进行零样本预测。
- 输入：2024-07-25 之前的历史数据
- 预测：2024-07-25 起未来 10 天（240 小时）的 NEXTLOAD/LOAD
- 对比：与数据中对应日期的实测 LOAD 进行精度评估
- 主变量：LOAD
- 动态数值协变量：WBGT、ICHB、ET、THI、CHI

运行前请确保已安装依赖（示例）：
    pip install chronos-forecasting pandas openpyxl matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
TARGET_VAR = "LOAD"
HORIZON_HOURS = 10 * 24      # 未来 10 天 = 240 小时
SPLIT_DATE = pd.Timestamp("2024-06-25")
ID_COLUMN = "item_id"
TIMESTAMP_COLUMN = "DATATIME"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronos-2 负荷预测")
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
    parser.add_argument(
        "--model",
        default="amazon/chronos-2",
        help="Chronos-2 模型名称或 HuggingFace 路径",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备：cuda 或 cpu",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="模型使用的最大上下文长度，默认使用模型默认值",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="推理 batch size",
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
    required_cols = [TIMESTAMP_COLUMN, TARGET_VAR] + COVARIATES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[错误] 缺少列: {missing}")
        return 1

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    print(f"       时间范围: {df[TIMESTAMP_COLUMN].min()} ~ {df[TIMESTAMP_COLUMN].max()}")
    print(f"       总行数: {len(df)}")

    context_df = df[df[TIMESTAMP_COLUMN] < SPLIT_DATE].copy()
    horizon_end = SPLIT_DATE + pd.Timedelta(hours=HORIZON_HOURS)
    horizon_df = df[
        (df[TIMESTAMP_COLUMN] >= SPLIT_DATE) & (df[TIMESTAMP_COLUMN] < horizon_end)
    ].copy()

    if len(context_df) == 0:
        print("[错误] 2024-06-25 之前无数据，无法构建上下文。")
        return 1
    if len(horizon_df) < HORIZON_HOURS:
        print(
            f"[错误] 2024-06-25 起未来 10 天数据不足，"
            f"实际只有 {len(horizon_df)} 条，需要 {HORIZON_HOURS} 条。"
        )
        return 1

    print(
        f"       上下文: {context_df[TIMESTAMP_COLUMN].min()} ~ {context_df[TIMESTAMP_COLUMN].max()} "
        f"({len(context_df)} 条)"
    )
    print(
        f"       预测期: {horizon_df[TIMESTAMP_COLUMN].min()} ~ {horizon_df[TIMESTAMP_COLUMN].max()} "
        f"({len(horizon_df)} 条)"
    )

    # ---------------------------------------------------------------------
    # 2) 构造 Chronos-2 需要的 long-format 输入
    # ---------------------------------------------------------------------
    print("[2/7] 构造 Chronos-2 输入格式（long-format）...")

    # Chronos-2 要求时间序列具有严格的规则频率，先构造完整小时索引并插值
    def _resample_to_hourly(df_in: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        df_in = df_in.set_index(TIMESTAMP_COLUMN).sort_index()
        full_index = pd.date_range(start=start, end=end, freq="h")
        df_in = df_in.reindex(full_index)
        # 对 LOAD、NEXTLOAD 以及协变量进行线性插值
        interp_cols = [TARGET_VAR, "NEXTLOAD"] + COVARIATES
        for col in interp_cols:
            if col in df_in.columns:
                df_in[col] = df_in[col].interpolate(method="linear")
        df_in[TIMESTAMP_COLUMN] = df_in.index
        return df_in.reset_index(drop=True)

    # context：从数据起始到 SPLIT_DATE 前一个小时
    context_start = context_df[TIMESTAMP_COLUMN].min().floor("h")
    context_end = SPLIT_DATE - pd.Timedelta(hours=1)
    context_df = _resample_to_hourly(context_df, context_start, context_end)

    # horizon：从 SPLIT_DATE 起共 HORIZON_HOURS 小时
    horizon_start = SPLIT_DATE
    horizon_end_exact = SPLIT_DATE + pd.Timedelta(hours=HORIZON_HOURS) - pd.Timedelta(hours=1)
    horizon_df = _resample_to_hourly(horizon_df, horizon_start, horizon_end_exact)

    context_df[ID_COLUMN] = "series_1"
    horizon_df[ID_COLUMN] = "series_1"

    # context：包含目标列 + 协变量
    context_cols = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_VAR] + COVARIATES
    chronos_context = context_df[context_cols].copy()

    # future_df：仅包含未来已知的协变量（不需要目标列）
    future_cols = [ID_COLUMN, TIMESTAMP_COLUMN] + COVARIATES
    chronos_future = horizon_df[future_cols].copy()

    # ---------------------------------------------------------------------
    # 3) 加载 Chronos-2 模型
    # ---------------------------------------------------------------------
    print(f"[3/7] 加载 Chronos-2 模型: {args.model}（首次会从 HuggingFace 下载权重）...")
    from chronos import Chronos2Pipeline

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"       使用设备: {device}")

    pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=device)

    # ---------------------------------------------------------------------
    # 4) 执行预测
    # ---------------------------------------------------------------------
    print("[4/7] 执行 Chronos-2 predict_df ...")
    pred_df = pipeline.predict_df(
        df=chronos_context,
        future_df=chronos_future,
        id_column=ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        target=TARGET_VAR,
        prediction_length=HORIZON_HOURS,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=args.batch_size,
        context_length=args.context_length,
    )

    # 按时间排序，确保顺序与 horizon_df 一致
    pred_df = pred_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    pred = pred_df["0.5"].to_numpy(dtype=np.float32)      # median
    mean_fc = pred_df["predictions"].to_numpy(dtype=np.float32)  # mean
    lower_80 = pred_df["0.1"].to_numpy(dtype=np.float32)
    upper_80 = pred_df["0.9"].to_numpy(dtype=np.float32)

    print(f"       预测长度: {len(pred)} 小时")

    # ---------------------------------------------------------------------
    # 5) 提取实测值并计算误差指标
    # ---------------------------------------------------------------------
    print("[5/7] 与实测 LOAD 对比...")
    actual = horizon_df[TARGET_VAR].to_numpy(dtype=np.float32)

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
    future_dt = horizon_df[TIMESTAMP_COLUMN].values

    result_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: future_dt,
            "LOAD_ACTUAL": np.round(actual, 3),
            "LOAD_PRED": np.round(pred, 3),
            "LOAD_MEAN": np.round(mean_fc, 3),
            "LOAD_Q10": np.round(lower_80, 3),
            "LOAD_Q90": np.round(upper_80, 3),
        }
    )

    csv_path = output_dir / "load_forecast_chronos_10days.csv"
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
    plot_context_dt = context_df[TIMESTAMP_COLUMN].iloc[-plot_context_hours:]
    split_line = horizon_df[TIMESTAMP_COLUMN].iloc[0]

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
        "兰州负荷 2024-06-25 起未来 10 天预测（Chronos-2 + 协变量）\n"
        f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  Coverage={coverage:.1f}%"
    )
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷 (MW)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / "load_forecast_chronos_10days.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"       图片已保存: {png_path.resolve()}")

    print("\n[完成] 预测与对比结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
