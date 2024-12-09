# データ分析および描画を行うスクリプト

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider

import datetime

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

target = "jpy"

if target == "btc":
    target_cols = ["close"]
    data_loader = GluontsDataProvider(
        file_path=f"dataset/btc.csv",
        index_col="timeOpen",
        target_cols=target_cols,
        prediction_length=30,
        context_length=30,
        freq="D",
        train_start_date=datetime.datetime(2000, 1, 1),
        test_start_date=datetime.datetime(2023, 1, 1),
        scaler_flag=False
    )
elif target == "jpy":
    target_cols = ["Price"]
    data_loader = GluontsDataProvider(
        file_path=f"dataset/usd_jpy.csv",
        index_col="Date",
        target_cols=target_cols,
        prediction_length=30,
        context_length=30,
        freq="D",
        train_start_date=datetime.datetime(1972, 1, 1),
        test_start_date=datetime.datetime(2023, 1, 1),
        scaler_flag=False
    )

data_array = np.array([])

#data_array = np.append(data_array, data_loader.train[target_cols[0]].values)
data_array = np.append(data_array, data_loader.test[target_cols[0]].values[30:])

print("ave", np.mean(data_array))
print("var", np.var(data_array))
print("std", np.std(data_array))
print("max", np.max(data_array))
print("min", np.min(data_array))

print("len", len(data_array))

fig, ax = plt.subplots()

#ax.plot(data_loader.train.index, data_loader.train[target_cols[0]].values, label="train", color="blue")
ax.plot(data_loader.test.index[30:], data_loader.test[target_cols[0]].values[30:], label="test", color="blue")

#ax.legend()
ax.set_xlabel("Date")
#ax.set_ylabel("Price($)")

ax.xaxis.set_major_locator(mdates.YearLocator(1))

fig.savefig("output.png")