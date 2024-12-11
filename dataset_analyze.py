# データ分析および描画を行うスクリプト

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider

import datetime

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

target = "btc"

test_start_date = datetime.datetime.strptime("2023-12-01", "%Y-%m-%d")
train_num = 180
test_num = 30

if target == "btc":
    target_cols = ["close"]
    data_loader = GluontsDataProvider(
        file_path=f"dataset/btc.csv",
        index_col="timeOpen",
        target_cols=target_cols,
        prediction_length=30,
        context_length=30,
        freq="D",
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
        scaler_flag=False
    )

data_loader.update_date_by_index(test_start_date, train_num, test_num)

data_array = np.array([])

#data_array = np.append(data_array, data_loader.train[target_cols[0]].values)
data_array = np.append(data_array, data_loader.test[target_cols[0]].values[30:])

print("平均:", np.mean(data_array).round(2))
#print("var", np.var(data_array))
print("標準偏差", np.std(data_array).round(2))
print("最高値", np.max(data_array).round(2))
print("最低値", np.min(data_array).round(2))

print("len", len(data_array))

fig, ax = plt.subplots()

ax.plot(data_loader.test.index[:30], data_loader.test[target_cols[0]].values[:30], label="input", color="red")
ax.plot(data_loader.test.index[30:], data_loader.test[target_cols[0]].values[30:], label="target", color="blue")

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price($)")

#ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_locator(mdates.MonthLocator())

fig.savefig("output_test.png")
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(data_loader.train.index, data_loader.train[target_cols[0]].values, label="train", color="red")

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price($)")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

fig.savefig("output_train.png")