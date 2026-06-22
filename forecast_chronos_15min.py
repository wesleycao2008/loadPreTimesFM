#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Amazon Chronos-2 对 15 分钟级负荷数据进行零样本预测，并将结果写入达梦数据库。
- 输入：直接从达梦数据库读取 MEA 发电数据、NWP 温度/辐照数据（15 分钟一条记录）
- 协变量：TEMPERATURE、RADI
- 目标变量：y
- 预测：未来 10 天，每 15 分钟一条记录（共 960 条）
- 对比：与数据中对应日期的实测 y 进行精度评估
- 入库：每天一条记录，V0000~V2345 存 00:00~23:45 的预测值，V2400 可配置

新增滚动预测：
- 通过 --start-date 和 --end-date 指定起止日期，系统按天为单位执行滚动预测。
- 每个日期预测未来 10 天；输出文件按日期后缀命名。
- 每次预测只从数据库读取该次所需的输入时间窗口，不一次性加载全量数据。
- MEA 发电数据与 NWP 气象数据可分别配置 ID。

输出控制：
- 默认写入达梦数据库，不生成 CSV/PNG 文件。
- 加 --skip-db 时跳过数据库写入，改为仅生成 CSV 和 PNG 文件。

运行前请确保已安装依赖（示例）：
    pip install chronos-forecasting pandas openpyxl matplotlib
    # dmPython 通常由达梦数据库安装包提供，需确保 Python 环境可导入
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
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
COVARIATES = ["TEMPERATURE", "RADI"]
TARGET_VAR = "y"
TIMESTAMP_COLUMN = "ds"
ID_COLUMN = "item_id"

HORIZON_DAYS = 10
STEPS_PER_DAY = 24 * 4              # 15 分钟一条记录，一天 96 条
HORIZON_STEPS = HORIZON_DAYS * STEPS_PER_DAY
FREQ = "15min"                      # 数据频率

# 达梦数据库表列定义
V_COLUMNS = [f"V{h:02d}{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
V2400_COLUMN = "V2400"
ALL_V_COLUMNS = V_COLUMNS + [V2400_COLUMN]
PK_COLUMNS = ["ID", "MEAS_TYPE", "DATASOURCE_ID", "FORECAST_TYPE", "FB_TIME", "YB_TIME", "FB_SEQ"]
META_COLUMNS = ["FORE_CYCLE", "UPDATE_TIME"]
INSERT_COLUMNS = PK_COLUMNS + META_COLUMNS + ALL_V_COLUMNS


def _v_column_name(ts: pd.Timestamp) -> str:
    """把 00:00~23:45 的时间戳映射到 V0000~V2345 列名。"""
    return f"V{ts.hour:02d}{ts.minute:02d}"


def _resample_to_15min(
    df_in: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """将数据重采样为严格的 15 分钟频率，缺失值线性插值。"""
    df_in = df_in.set_index(TIMESTAMP_COLUMN).sort_index()
    full_index = pd.date_range(start=start, end=end, freq=FREQ)
    df_in = df_in.reindex(full_index)
    interp_cols = [TARGET_VAR] + COVARIATES
    for col in interp_cols:
        if col in df_in.columns:
            df_in[col] = df_in[col].interpolate(method="linear", limit_direction="both")
    df_in[TIMESTAMP_COLUMN] = df_in.index
    return df_in.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 数据库按需读取
# ---------------------------------------------------------------------------
def _yearly_table(schema: str, base_table: str, year: int) -> str:
    """
    根据 schema、表名前缀和年份生成完整表名。
    格式："SCHEMA"."TABLE_YYYY"
    """
    return f'"{schema}"."{base_table}_{year}"'


def _read_mea_range(
    conn, start: pd.Timestamp, end: pd.Timestamp, args: argparse.Namespace
) -> pd.DataFrame:
    """
    从 MEA 表按时间窗口读取发电数据；若窗口跨年，则分别查询对应年份的表。
    表中每行 1 小时（V00/V15/V30/V45），展开为 15 分钟一行。
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    years = list(range(start_ts.year, end_ts.year + 1))

    all_records = []
    for year in years:
        table = _yearly_table(args.db_user, args.db_mea_table, year)
        year_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end_ts, pd.Timestamp(f"{year}-12-31 23:59:59"))

        cursor = conn.cursor()
        start_hour = year_start.floor("h")
        end_hour = year_end.floor("h")
        sql = (
            "SELECT DATA_TIME, V00, V15, V30, V45 "
            f"FROM {table} "
            "WHERE ID=? AND MEAS_TYPE=? AND DATA_TIME>=? AND DATA_TIME<=? "
            "ORDER BY DATA_TIME"
        )
        cursor.execute(sql, (args.db_id, args.db_mea_meas_type, start_hour, end_hour))
        rows = cursor.fetchall()
        cursor.close()

        for row in rows:
            base_time = row[0]
            for minute_offset, val in zip([0, 15, 30, 45], row[1:]):
                if val is not None:
                    ds = base_time + timedelta(minutes=minute_offset)
                    if start <= ds <= end:
                        all_records.append({TIMESTAMP_COLUMN: ds, TARGET_VAR: float(val)})

    df = pd.DataFrame(all_records)
    if not df.empty:
        df.sort_values(TIMESTAMP_COLUMN, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _read_nwp_range(
    conn,
    start: pd.Timestamp,
    end: pd.Timestamp,
    meas_type: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """
    从 NWP 表按时间窗口读取温度或辐照数据；若窗口跨年，则分别查询对应年份的表。
    表中每行 1 天（V0000~V2300，每小时 1 个点），线性插值为 15 分钟一行。
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    years = list(range(start_ts.year, end_ts.year + 1))

    all_hourly_records = []
    for year in years:
        table = _yearly_table(args.db_user, args.db_nwp_table, year)
        year_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end_ts, pd.Timestamp(f"{year}-12-31 23:59:59"))

        cursor = conn.cursor()
        hourly_cols = ", ".join([f"V{i:02d}00" for i in range(24)])
        start_day = year_start.floor("D")
        end_day = year_end.floor("D")
        sql = (
            f"SELECT YB_TIME, {hourly_cols} "
            f"FROM {table} "
            "WHERE ID=? AND MEAS_TYPE=? AND YB_TIME=FB_TIME+1 "
            "AND YB_TIME>=? AND YB_TIME<=? "
            "ORDER BY YB_TIME"
        )
        cursor.execute(sql, (args.db_id_nwp, meas_type, start_day, end_day))
        rows = cursor.fetchall()
        cursor.close()

        for row in rows:
            base_date = row[0]
            for hour in range(24):
                val = row[1 + hour]
                if val is not None:
                    ts = base_date + timedelta(hours=hour)
                    all_hourly_records.append({TIMESTAMP_COLUMN: ts, "value": float(val)})

    hourly_df = pd.DataFrame(all_hourly_records)
    if hourly_df.empty:
        return pd.DataFrame(columns=[TIMESTAMP_COLUMN, "value"])

    hourly_df.set_index(TIMESTAMP_COLUMN, inplace=True)
    hourly_df.sort_index(inplace=True)

    start_time = pd.Timestamp(start).floor("15min")
    end_time = pd.Timestamp(end).floor("15min")
    target_index = pd.date_range(start=start_time, end=end_time, freq=FREQ)

    target_df = (
        hourly_df.reindex(hourly_df.index.union(target_index))
        .sort_index()
        .interpolate(method="linear")
    )
    target_df = target_df.loc[target_index].reset_index().rename(
        columns={"index": TIMESTAMP_COLUMN, "value": "value"}
    )
    return target_df


def _load_forecast_data(
    conn, split_date: pd.Timestamp, args: argparse.Namespace
) -> pd.DataFrame | None:
    """
    为单次预测从数据库读取所需时间窗口的数据：
    - 上下文：split_date 前 max_context_days 天
    - 预测期：split_date 起未来 HORIZON_DAYS 天
    """
    horizon_end = split_date + pd.Timedelta(days=HORIZON_DAYS)
    context_start = split_date - pd.Timedelta(days=args.max_context_days)

    print(f"       数据库读取窗口: {context_start} ~ {horizon_end}")

    df_mea = _read_mea_range(conn, context_start, horizon_end, args)
    if df_mea.empty:
        print(f"[错误] {split_date.strftime('%Y-%m-%d')} 在数据库中未读取到 MEA 发电数据。")
        return None

    df_temp = _read_nwp_range(conn, context_start, horizon_end, args.db_nwp_temp_meas_type, args)
    df_temp.rename(columns={"value": "TEMPERATURE"}, inplace=True)

    df_radi = _read_nwp_range(conn, context_start, horizon_end, args.db_nwp_radi_meas_type, args)
    df_radi.rename(columns={"value": "RADI"}, inplace=True)

    # 使用 outer merge，保留 NWP 未来协变量行，即使 MEA 未来实测负荷尚未入库
    df = df_mea.merge(df_temp, on=TIMESTAMP_COLUMN, how="outer")
    df = df.merge(df_radi, on=TIMESTAMP_COLUMN, how="outer")

    # 删除历史上下文中 RADI 全天为 0 的异常日期；保留未来预测期
    df["date"] = df[TIMESTAMP_COLUMN].dt.date
    days_all_zero = df.groupby("date")["RADI"].transform(lambda x: (x == 0).all())
    context_mask = df[TIMESTAMP_COLUMN] < split_date
    removed_days = df.loc[context_mask & days_all_zero, "date"].unique()
    df = df[~(context_mask & days_all_zero)].copy()
    df.drop(columns=["date"], inplace=True)
    if len(removed_days):
        print(
            f"       移除 {len(removed_days)} 天历史 RADI 全为 0 的日期: "
            f"{sorted(str(d) for d in removed_days)}"
        )

    # 补齐完整时间轴：未来 y 缺失时设为 NaN，协变量由 NWP 插值填充
    full_start = context_start.floor(FREQ)
    full_end = (horizon_end - pd.Timedelta(minutes=15)).floor(FREQ)
    full_index = pd.date_range(start=full_start, end=full_end, freq=FREQ)

    df = df.set_index(TIMESTAMP_COLUMN).reindex(full_index)
    for col in COVARIATES:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
    df[TIMESTAMP_COLUMN] = df.index
    df = df.reset_index(drop=True)

    df.sort_values(TIMESTAMP_COLUMN, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df


def _build_db_rows(result_df: pd.DataFrame, fb_time: datetime, args: argparse.Namespace) -> list[dict]:
    """把 15 分钟预测结果按天拆成数据库行。"""
    df = result_df.copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df["date"] = df[TIMESTAMP_COLUMN].dt.date

    rows: list[dict] = []
    dates = sorted(df["date"].unique())

    for idx, date in enumerate(dates):
        day_df = df[df["date"] == date].sort_values(TIMESTAMP_COLUMN)
        if len(day_df) != STEPS_PER_DAY:
            print(
                f"[警告] {date} 的数据量不是 {STEPS_PER_DAY} 条，实际 {len(day_df)} 条，跳过入库。"
            )
            continue

        values: dict[str, float | None] = {}
        for _, row in day_df.iterrows():
            col = _v_column_name(pd.Timestamp(row[TIMESTAMP_COLUMN]))
            values[col] = round(float(row["y_PRED"]), 4)

        # V2400 的处理策略
        if args.fill_v2400 == "next_day_first":
            if idx + 1 < len(dates):
                next_date = dates[idx + 1]
                next_first = df[df["date"] == next_date].sort_values(TIMESTAMP_COLUMN).iloc[0]
                values[V2400_COLUMN] = round(float(next_first["y_PRED"]), 4)
            else:
                values[V2400_COLUMN] = round(float(day_df.iloc[-1]["y_PRED"]), 4)
        elif args.fill_v2400 == "last":
            values[V2400_COLUMN] = round(float(day_df.iloc[-1]["y_PRED"]), 4)
        else:  # null
            values[V2400_COLUMN] = None

        rows.append(
            {
                "ID": args.db_id,
                "MEAS_TYPE": args.db_meas_type,
                "DATASOURCE_ID": args.db_datasource_id,
                "FORECAST_TYPE": int(args.db_forecast_type),
                "FB_TIME": fb_time,
                "YB_TIME": datetime.combine(date, datetime.min.time()),
                "FB_SEQ": int(args.db_fb_seq),
                "FORE_CYCLE": int(args.db_fore_cycle),
                "UPDATE_TIME": datetime.now(),
                **values,
            }
        )

    return rows


def _write_to_db(rows: list[dict], args: argparse.Namespace) -> tuple[int, int]:
    """连接达梦数据库，按 YB_TIME 年份分表执行更新或插入。返回 (inserted, updated)。"""
    import dmPython

    conn = dmPython.connect(
        user=args.db_user,
        password=args.db_password,
        server=args.db_host,
        port=args.db_port,
    )
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        set_clause = ",".join([f'"{c}"=?' for c in ALL_V_COLUMNS + ["UPDATE_TIME"]])
        where_clause = " AND ".join([f'"{c}"=?' for c in PK_COLUMNS])
        insert_col_str = ",".join([f'"{c}"' for c in INSERT_COLUMNS])
        insert_placeholders = ",".join(["?"] * len(INSERT_COLUMNS))

        # 按 YB_TIME 年份分表
        rows_by_year: dict[int, list[dict]] = {}
        for row in rows:
            year = row["YB_TIME"].year
            rows_by_year.setdefault(year, []).append(row)

        inserted = 0
        updated = 0
        for year, year_rows in rows_by_year.items():
            table = _yearly_table(args.db_user, args.db_table, year)
            update_sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
            insert_sql = f"INSERT INTO {table} ({insert_col_str}) VALUES ({insert_placeholders})"

            for row in year_rows:
                update_params = [row[c] for c in ALL_V_COLUMNS + ["UPDATE_TIME"]] + [
                    row[c] for c in PK_COLUMNS
                ]
                cursor.execute(update_sql, update_params)
                if cursor.rowcount and cursor.rowcount > 0:
                    updated += 1
                else:
                    insert_params = [row[c] for c in INSERT_COLUMNS]
                    cursor.execute(insert_sql, insert_params)
                    inserted += 1

        conn.commit()
        return inserted, updated
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        cursor.close()
        conn.close()


def _run_single_forecast(
    split_date: pd.Timestamp,
    df: pd.DataFrame,
    pipeline,
    args: argparse.Namespace,
    output_dir: Path,
    use_date_suffix: bool = True,
) -> bool:
    """执行单日 split_date 起未来 10 天的预测并保存/入库，返回是否成功。"""
    date_str = split_date.strftime("%Y-%m-%d")
    date_suffix = split_date.strftime("%Y%m%d")
    file_suffix = f"_{date_suffix}" if use_date_suffix else ""
    horizon_end = split_date + pd.Timedelta(minutes=HORIZON_STEPS * 15)

    if args.start_date and args.end_date:
        # 滚动回测：FB_TIME 为预测起始日前一天
        fb_time = (split_date - pd.Timedelta(days=1)).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        # 单次运行：FB_TIME 由 --fb-date 指定
        fb_time = pd.Timestamp(args.fb_date).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    print(f"\n---------- 滚动预测日期: {date_str} ----------")

    # 1) 按日期切分
    context_df = df[df[TIMESTAMP_COLUMN] < split_date].copy()
    horizon_df = df[
        (df[TIMESTAMP_COLUMN] >= split_date) & (df[TIMESTAMP_COLUMN] < horizon_end)
    ].copy()

    if len(context_df) == 0:
        print(f"[错误] {date_str} 之前无数据，无法构建上下文，跳过。")
        return False
    if len(horizon_df) == 0:
        print(f"[错误] {date_str} 起未来 10 天无数据（含协变量），无法预测，跳过。")
        return False

    horizon_end_exact = split_date + pd.Timedelta(minutes=15 * (HORIZON_STEPS - 1))
    if (
        horizon_df[TIMESTAMP_COLUMN].min() > split_date
        or horizon_df[TIMESTAMP_COLUMN].max() < horizon_end_exact
    ):
        print(
            f"[错误] {date_str} 起未来 10 天数据不完整，"
            f"需要覆盖 {split_date} ~ {horizon_end_exact}，跳过。"
        )
        return False

    missing_context = max(
        0,
        len(
            pd.date_range(
                context_df[TIMESTAMP_COLUMN].min().floor(FREQ),
                split_date - pd.Timedelta(minutes=15),
                freq=FREQ,
            )
        )
        - len(context_df),
    )
    missing_horizon = max(0, HORIZON_STEPS - len(horizon_df))
    if missing_context or missing_horizon:
        print(
            f"       注意：原始数据存在缺失，将使用线性插值补齐"
            f"（上下文缺 {missing_context} 条，预测期缺 {missing_horizon} 条）"
        )

    print(
        f"       上下文: {context_df[TIMESTAMP_COLUMN].min()} ~ {context_df[TIMESTAMP_COLUMN].max()} "
        f"({len(context_df)} 条)"
    )
    print(
        f"       预测期: {horizon_df[TIMESTAMP_COLUMN].min()} ~ {horizon_df[TIMESTAMP_COLUMN].max()} "
        f"({len(horizon_df)} 条)"
    )

    # 2) 构造 Chronos-2 需要的 long-format 输入
    print("       构造 Chronos-2 输入格式（long-format）...")

    context_start = context_df[TIMESTAMP_COLUMN].min().floor(FREQ)
    context_end = split_date - pd.Timedelta(minutes=15)
    context_df = _resample_to_15min(context_df, context_start, context_end)

    horizon_start = split_date
    horizon_df = _resample_to_15min(horizon_df, horizon_start, horizon_end_exact)

    context_df[ID_COLUMN] = "series_1"
    horizon_df[ID_COLUMN] = "series_1"

    context_cols = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_VAR] + COVARIATES
    chronos_context = context_df[context_cols].copy()

    future_cols = [ID_COLUMN, TIMESTAMP_COLUMN] + COVARIATES
    chronos_future = horizon_df[future_cols].copy()

    for df_check, name in [(chronos_context, "上下文"), (chronos_future, "预测期协变量")]:
        nan_cols = [c for c in df_check.columns if df_check[c].isna().any()]
        if nan_cols:
            print(f"[错误] {name} 在列 {nan_cols} 中仍存在缺失值，请检查原始数据。")
            return False

    # 3) 执行预测
    print("       执行 Chronos-2 predict_df ...")
    pred_df = pipeline.predict_df(
        df=chronos_context,
        future_df=chronos_future,
        id_column=ID_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        target=TARGET_VAR,
        prediction_length=HORIZON_STEPS,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=args.batch_size,
        context_length=args.context_length,
    )

    pred_df = pred_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    pred = pred_df["0.5"].to_numpy(dtype=np.float32)              # median
    mean_fc = pred_df["predictions"].to_numpy(dtype=np.float32)   # mean
    lower_80 = pred_df["0.1"].to_numpy(dtype=np.float32)
    upper_80 = pred_df["0.9"].to_numpy(dtype=np.float32)

    print(f"       预测长度: {len(pred)} 条（15 分钟/条）")

    # 4) 判断是否存在预测期实测值，并决定是否计算误差指标
    has_actual = horizon_df[TARGET_VAR].notna().any()

    if has_actual:
        print("       与实测 y 对比...")
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
    else:
        print("       未找到预测期实测 y，执行纯未来预测（不计算误差指标）...")
        actual = np.full_like(pred, np.nan)
        mae = rmse = mape = coverage = None

    # 5) 构造预测结果（用于入库或导出）
    print("       构造预测结果...")
    future_dt = horizon_df[TIMESTAMP_COLUMN].values

    result_data = {
        TIMESTAMP_COLUMN: future_dt,
        "y_PRED": np.round(pred, 3),
        "y_MEAN": np.round(mean_fc, 3),
        "y_Q10": np.round(lower_80, 3),
        "y_Q90": np.round(upper_80, 3),
    }
    if has_actual:
        result_data["y_ACTUAL"] = np.round(actual, 3)
        # 保持与原有输出顺序一致
        result_data = {
            TIMESTAMP_COLUMN: result_data[TIMESTAMP_COLUMN],
            "y_ACTUAL": result_data["y_ACTUAL"],
            "y_PRED": result_data["y_PRED"],
            "y_MEAN": result_data["y_MEAN"],
            "y_Q10": result_data["y_Q10"],
            "y_Q90": result_data["y_Q90"],
        }
    result_df = pd.DataFrame(result_data)

    if args.skip_db:
        # 仅生成 CSV 和图表
        print("       保存预测结果到 CSV...")
        csv_path = output_dir / f"load_forecast_chronos_15min{file_suffix}_10days.csv"
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"       CSV 已保存: {csv_path.resolve()}")

        print("       绘制可视化图表...")
        plot_context_steps = HORIZON_STEPS
        plot_context = (
            context_df[TARGET_VAR].iloc[-plot_context_steps:].to_numpy(dtype=np.float32)
        )
        plot_context_dt = context_df[TIMESTAMP_COLUMN].iloc[-plot_context_steps:]
        split_line = horizon_df[TIMESTAMP_COLUMN].iloc[0]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            plot_context_dt,
            plot_context,
            label="历史 y",
            color="steelblue",
            linewidth=1.5,
        )
        ax.plot(
            future_dt,
            pred,
            label="y 预测 (median)",
            color="tab:orange",
            linewidth=2,
        )
        if has_actual:
            ax.plot(
                future_dt,
                actual,
                label="y 实测",
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
        if has_actual:
            title_metrics = f"MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  Coverage={coverage:.1f}%"
        else:
            title_metrics = "无实测对比（纯未来预测）"
        ax.set_title(
            f"15 分钟级负荷 {date_str} 起未来 10 天预测（Chronos-2 + 协变量）\n"
            f"{title_metrics}"
        )
        ax.set_xlabel("时间")
        ax.set_ylabel("负荷")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = output_dir / f"load_forecast_chronos_15min{file_suffix}_10days.png"
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"       图片已保存: {png_path.resolve()}")
    else:
        # 写入达梦数据库
        print("       写入达梦数据库...")
        try:
            db_rows = _build_db_rows(result_df, fb_time, args)
            print(f"       待写入 {len(db_rows)} 天记录，FB_TIME={fb_time}")
            inserted, updated = _write_to_db(db_rows, args)
            print(f"       插入 {inserted} 条，更新 {updated} 条")
        except Exception as exc:
            print(f"[错误] 数据库写入失败: {exc}")
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronos-2 15 分钟级负荷预测 + 达梦入库")
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
        "--start-date",
        default=None,
        help="滚动预测开始日期（含），格式 YYYY-MM-DD；需与 --end-date 同时指定",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="滚动预测结束日期（含），格式 YYYY-MM-DD；需与 --start-date 同时指定",
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

    # 数据库相关参数
    parser.add_argument(
        "--db-host",
        default="127.0.0.1",
        help="达梦数据库地址",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5236,
        help="达梦数据库端口",
    )
    parser.add_argument(
        "--db-user",
        default="RES2000_FJ",
        help="达梦数据库用户名",
    )
    parser.add_argument(
        "--db-password",
        default="damengres2000",
        help="达梦数据库密码",
    )
    # MEA 发电数据源配置
    parser.add_argument(
        "--db-mea-table",
        default="RES_CON_PWRGRID_H1_MEA",
        help="MEA 发电数据表名前缀，实际表名为 \"db_user\".\"prefix_YYYY\"",
    )
    parser.add_argument(
        "--db-mea-meas-type",
        default="10132001",
        help="MEA 发电数据 MEAS_TYPE",
    )
    # NWP 气象数据源配置
    parser.add_argument(
        "--db-nwp-table",
        default="RES_CON_NWP_F_FORECAST",
        help="NWP 气象数据表名前缀，实际表名为 \"db_user\".\"prefix_YYYY\"",
    )
    parser.add_argument(
        "--db-id-nwp",
        default="0101350600",
        help="NWP 气象数据 ID",
    )
    parser.add_argument(
        "--db-nwp-temp-meas-type",
        default="00001002",
        help="NWP 温度数据 MEAS_TYPE",
    )
    parser.add_argument(
        "--db-nwp-radi-meas-type",
        default="40071013",
        help="NWP 辐照数据 MEAS_TYPE",
    )
    parser.add_argument(
        "--max-context-days",
        type=int,
        default=60,
        help="每次预测从数据库读取的上下文历史天数",
    )
    parser.add_argument(
        "--db-table",
        default="RES_CON_PWRGRID_F_FORECAST",
        help="目标表名前缀，实际表名为 \"db_user\".\"prefix_YYYY\"",
    )
    parser.add_argument(
        "--db-id",
        default="0101350600",
        help="数据 ID",
    )
    parser.add_argument(
        "--db-meas-type",
        default="10132024",
        help="MEAS_TYPE",
    )
    parser.add_argument(
        "--db-datasource-id",
        default="0021350000",
        help="DATASOURCE_ID",
    )
    parser.add_argument(
        "--db-forecast-type",
        type=int,
        default=1001,
        help="FORECAST_TYPE",
    )
    parser.add_argument(
        "--db-fb-seq",
        type=int,
        default=1,
        help="FB_SEQ",
    )
    parser.add_argument(
        "--db-fore-cycle",
        type=int,
        default=1,
        help="FORE_CYCLE",
    )
    parser.add_argument(
        "--fb-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="预报时间 FB_TIME，格式 YYYY-MM-DD；默认当日日期",
    )
    parser.add_argument(
        "--fill-v2400",
        choices=["next_day_first", "last", "null"],
        default="next_day_first",
        help="V2400（24:00）填充策略：次日凌晨首值/当日末值/NULL",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="跳过数据库写入，改为仅生成 CSV 和图表；默认写入数据库时不生成文件",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定预测日期列表
    if args.start_date and args.end_date:
        # 滚动回测：--start-date/--end-date 为预测起始日期
        start_date = pd.Timestamp(args.start_date)
        end_date = pd.Timestamp(args.end_date)
        if end_date < start_date:
            print("[错误] --end-date 不能早于 --start-date")
            return 1
        split_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    elif args.start_date or args.end_date:
        print("[错误] --start-date 和 --end-date 必须同时指定")
        return 1
    else:
        # 单次运行：由 FB_TIME 推导预测起始日期（split_date = fb_date + 1 天）
        fb_date = pd.Timestamp(args.fb_date)
        split_date = fb_date + pd.Timedelta(days=1)
        split_dates = [split_date]

    print(f"       预测日期: {', '.join([d.strftime('%Y-%m-%d') for d in split_dates])}")

    # 1) 连接数据库
    print("[1/4] 连接达梦数据库...")
    import dmPython

    conn = dmPython.connect(
        user=args.db_user,
        password=args.db_password,
        server=args.db_host,
        port=args.db_port,
    )
    print(f"       数据库: {args.db_host}:{args.db_port} / {args.db_user}")

    try:
        # 2) 加载模型
        print(f"[2/4] 加载 Chronos-2 模型: {args.model}（首次会从 HuggingFace 下载权重）...")
        from chronos import Chronos2Pipeline

        device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
        print(f"       使用设备: {device}")

        pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=device)

        # 3) 滚动预测
        print(f"[3/4] 开始滚动预测，共 {len(split_dates)} 个日期...")
        success_count = 0
        fail_count = 0

        for split_date in split_dates:
            # 每次只读取本次预测所需时间窗口的数据
            df = _load_forecast_data(conn, split_date, args)
            if df is None or df.empty:
                print(f"[错误] {split_date.strftime('%Y-%m-%d')} 数据加载失败，跳过。")
                fail_count += 1
                continue

            print(
                f"       时间范围: {df[TIMESTAMP_COLUMN].min()} ~ {df[TIMESTAMP_COLUMN].max()}"
            )
            print(f"       行数: {len(df)}")
            print(f"       数据频率: 15 分钟")

            ok = _run_single_forecast(
                split_date,
                df,
                pipeline,
                args,
                output_dir,
                use_date_suffix=(len(split_dates) > 1),
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1

        # 4) 汇总
        print(f"\n[4/4] 滚动预测完成: 成功 {success_count} 天，失败 {fail_count} 天。")
        if success_count == 0:
            print("[错误] 没有一天预测成功。")
            return 1
        print("[完成] 预测、对比与入库结束。")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
