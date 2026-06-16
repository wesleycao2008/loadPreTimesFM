#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 Chronos-2 和 TimesFM 2.5 的预测误差"""

import pandas as pd
import numpy as np

# 读取两个算法的预测结果
chronos = pd.read_csv('output/load_forecast_chronos_10days.csv')
timesfm = pd.read_csv('output/load_forecast_10days.csv')

# 计算 Chronos-2 误差指标
actual = chronos['LOAD_ACTUAL'].values
pred_chronos = chronos['LOAD_PRED'].values
mae_chronos = np.mean(np.abs(actual - pred_chronos))
rmse_chronos = np.sqrt(np.mean((actual - pred_chronos) ** 2))
mape_chronos = np.mean(np.abs((actual - pred_chronos) / actual)) * 100

# 计算 TimesFM 误差指标
pred_timesfm = timesfm['LOAD_PRED'].values
mae_timesfm = np.mean(np.abs(actual - pred_timesfm))
rmse_timesfm = np.sqrt(np.mean((actual - pred_timesfm) ** 2))
mape_timesfm = np.mean(np.abs((actual - pred_timesfm) / actual)) * 100

print("=" * 60)
print("兰州负荷预测 10 天（240 小时）误差对比")
print("=" * 60)
print("预测期间: 2024-07-25 ~ 2024-08-03")
print("=" * 60)
print(f"{'指标':<15} {'Chronos-2':<15} {'TimesFM 2.5':<15} {'更优者':<10}")
print("-" * 60)
winner_mae = "Chronos-2" if mae_chronos < mae_timesfm else "TimesFM"
winner_rmse = "Chronos-2" if rmse_chronos < rmse_timesfm else "TimesFM"
winner_mape = "Chronos-2" if mape_chronos < mape_timesfm else "TimesFM"
print(f"{'MAE (MW)':<15} {mae_chronos:<15.2f} {mae_timesfm:<15.2f} {winner_mae:<10}")
print(f"{'RMSE (MW)':<15} {rmse_chronos:<15.2f} {rmse_timesfm:<15.2f} {winner_rmse:<10}")
print(f"{'MAPE (%)':<15} {mape_chronos:<15.2f} {mape_timesfm:<15.2f} {winner_mape:<10}")
print("=" * 60)

# 计算改进幅度
print("\n改进幅度分析:")
if mae_chronos < mae_timesfm:
    print(f"  Chronos-2 的 MAE 比 TimesFM 低 {mae_timesfm - mae_chronos:.2f} MW ({(mae_timesfm - mae_chronos)/mae_timesfm*100:.1f}%)")
else:
    print(f"  TimesFM 的 MAE 比 Chronos-2 低 {mae_chronos - mae_timesfm:.2f} MW ({(mae_chronos - mae_timesfm)/mae_chronos*100:.1f}%)")

if mape_chronos < mape_timesfm:
    print(f"  Chronos-2 的 MAPE 比 TimesFM 低 {mape_timesfm - mape_chronos:.2f}%")
else:
    print(f"  TimesFM 的 MAPE 比 Chronos-2 低 {mape_chronos - mape_timesfm:.2f}%")