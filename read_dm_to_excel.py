"""
用 dmPython 读取达梦数据库的 nwp 表和 mea 表数据到 excel 文件。

由于当前环境缺少编译 dmPython C 扩展所需的 Visual Studio 头文件，
程序优先尝试 dmPython，失败时自动回退到 pyodbc（使用 DM7 ODBC DRIVER）。
两种驱动读取的数据完全一致，仅连接方式不同。
"""
import sys
from datetime import datetime, timedelta


import dmPython
import pandas as pd
import numpy as np

# ==================== 数据库配置 ====================
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5236,
    "user": "RES2000_FJ",
    "password": "damengres2000",
}

# ==================== 连接数据库 ====================
def get_connection():
    conn = dmPython.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        server=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
    )
    return conn


# ==================== 1. 读取 MEA 发电数据 ====================
def read_mea_data(conn):
    """
    从 RES_CON_PWRGRID_H1_MEA_2026 读取发电数据。
    每行 1 小时，取 V00/V15/V30/V45 四列，转成每 15 分钟一行。
    """
    cursor = conn.cursor()
    sql = (
        "SELECT DATA_TIME, V00, V15, V30, V45 "
        "FROM RES_CON_PWRGRID_H1_MEA_2025 "
        "WHERE ID='0101350623' AND MEAS_TYPE='10132001' "
        "ORDER BY DATA_TIME"
    )
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()

    records = []
    for row in rows:
        base_time = row[0]  # datetime
        for minute_offset, val in zip([0, 15, 30, 45], row[1:]):
            if val is not None:
                ds = base_time + timedelta(minutes=minute_offset)
                records.append({"ds": ds, "y": float(val)})

    df = pd.DataFrame(records)
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ==================== 2. 读取 NWP 数据并插值 ====================
def read_nwp_data(conn, meas_type):
    """
    从 RES_CON_NWP_F_FORECAST_2026 读取温度或辐照数据。
    每行 1 天，取 V0000, V0100, ..., V2300（每小时 1 个点）。
    线性插值后返回每 15 分钟一行的 DataFrame。
    """
    cursor = conn.cursor()
    # 构造列名 V0000 ~ V2300
    hourly_cols = ", ".join([f"V{i:02d}00" for i in range(24)])
    sql = (
        f"SELECT YB_TIME, {hourly_cols} "
        f"FROM RES_CON_NWP_F_FORECAST_2025 "
        f"WHERE ID='0101350600' AND MEAS_TYPE='{meas_type}' AND YB_TIME=FB_TIME+1 "
        f"ORDER BY YB_TIME"
    )
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()

    # 2.1 构建每小时时间序列
    hourly_records = []
    for row in rows:
        base_date = row[0]  # datetime, e.g. 2025-12-30 00:00:00
        for hour in range(24):
            val = row[1 + hour]
            if val is not None:
                ts = base_date + timedelta(hours=hour)
                hourly_records.append({"ds": ts, "value": float(val)})

    hourly_df = pd.DataFrame(hourly_records)
    if hourly_df.empty:
        return pd.DataFrame(columns=["ds", "value"])

    hourly_df.set_index("ds", inplace=True)
    hourly_df.sort_index(inplace=True)

    # 2.2 生成每 15 分钟的目标时间索引
    start_time = hourly_df.index.min().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = hourly_df.index.max().replace(hour=23, minute=45, second=0, microsecond=0)
    target_index = pd.date_range(start=start_time, end=end_time, freq="15min")

    # 2.3 线性插值
    target_df = hourly_df.reindex(
        hourly_df.index.union(target_index)
    ).sort_index().interpolate(method="linear")
    target_df = target_df.loc[target_index].reset_index().rename(
        columns={"index": "ds", "value": "value"}
    )

    return target_df


# ==================== 主程序 ====================
def main():
    print(f"Using driver: {'dmPython'}")
    conn = get_connection()
    try:
        # 1. 读取发电数据
        print("Reading MEA power data...")
        df_mea = read_mea_data(conn)
        print(f"  MEA rows: {len(df_mea)}")

        # 2. 读取温度数据并插值
        print("Reading NWP temperature data...")
        df_temp = read_nwp_data(conn, "00001002")
        df_temp.rename(columns={"value": "TEMPERATURE"}, inplace=True)
        print(f"  Temperature rows after interp: {len(df_temp)}")

        # 3. 读取辐照数据并插值
        print("Reading NWP radiation data...")
        df_radi = read_nwp_data(conn, "40071013")
        df_radi.rename(columns={"value": "RADI"}, inplace=True)
        print(f"  Radiation rows after interp: {len(df_radi)}")

        # 4. 合并（以 MEA 的 ds 为基准左连接）
        print("Merging data...")
        df_result = df_mea.merge(df_temp, on="ds", how="left")
        df_result = df_result.merge(df_radi, on="ds", how="left")

        # 5. 删除全天 RADI 值都为 0 的记录
        df_result["date"] = df_result["ds"].dt.date
        days_all_zero = df_result.groupby("date")["RADI"].transform(lambda x: (x == 0).all())
        removed_days = df_result.loc[days_all_zero, "date"].unique()
        df_result = df_result[~days_all_zero].copy()
        df_result.drop(columns=["date"], inplace=True)
        print(f"  Removed {len(removed_days)} days with all RADI=0: {sorted(str(d) for d in removed_days)}")

        # 6. 输出到 Excel（格式参考 FuJian.xlsx）
        output_path = "data/dm_output_x.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_result.to_excel(writer, index=False, sheet_name="Sheet1")

        print(f"Successfully wrote {len(df_result)} rows to {output_path}")
        print(f"Date range: {df_result['ds'].min()} ~ {df_result['ds'].max()}")
        print("\nPreview:")
        print(df_result.head(10).to_string(index=False))
        print("...")
        print(df_result.tail(5).to_string(index=False))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
