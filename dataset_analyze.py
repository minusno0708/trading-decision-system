# データ分析および描画を行うスクリプト

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider
from data_provider.custom_data_provider import CustomDataProvider

import datetime

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

target = "jpy"

test_start_date = datetime.datetime.strptime("2023-01-02", "%Y-%m-%d")
train_data_length = 2000
val_data_length = 180
test_data_length = 360

if target == "btc":
    target_cols = ["close"]
    data_loader = CustomDataProvider(
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
    data_loader = CustomDataProvider(
        file_path=f"dataset/usd_jpy.csv",
        index_col="Date",
        target_cols=target_cols,
        prediction_length=30,
        context_length=30,
        freq="D",
        scaler_flag=False
    )
elif target == "wth":
    test_start_date = datetime.datetime.strptime("2020-12-01", "%Y-%m-%d")
    # "p (mbar),T (degC),Tpot (K),Tdew (degC),rh (%),VPmax (mbar),VPact (mbar),VPdef (mbar),sh (g/kg),H2OC (mmol/mol),rho (g/m**3),wv (m/s),max. wv (m/s),wd (deg),rain (mm),raining (s),SWDR (W/m�),PAR (�mol/m�/s),max. PAR (�mol/m�/s),Tlog (degC),OT"
    target_cols = ["p (mbar)"]
    data_loader = CustomDataProvider(
        file_path=f"dataset/weather.csv",
        index_col="date",
        target_cols=target_cols,
        prediction_length=30,
        context_length=30,
        freq="D",
        scaler_flag=False
    )

data_loader.update_date_by_index(test_start_date, train_data_length, test_data_length, val_data_length)

data_array = np.array([])

print(data_loader.test.columns)

#data_array = np.append(data_array, data_loader.train[target_cols[0]].values)
data_array = np.append(data_array, data_loader.test[target_cols[0]].values[30:])

print("平均:", np.mean(data_array).round(2))
#print("var", np.var(data_array))
print("標準偏差", np.std(data_array).round(2))
print("最高値", np.max(data_array).round(2))
print("最低値", np.min(data_array).round(2))

print("len", len(data_array))

context_length = 100
prediction_length = 30

fig, ax = plt.subplots()

target_index = 200
ax.plot(data_loader.test.index[target_index:target_index+context_length], data_loader.test[target_cols[0]].values[target_index:target_index+context_length], label="row data", color="red")
ax.plot(data_loader.test.index[target_index:target_index+context_length], data_loader.test["Price_mov_line_5"].values[target_index:target_index+context_length], label="average(5)", color="blue")
#ax.plot(data_loader.test.index[target_index:target_index+context_length], data_loader.test["Price_mov_line_10"].values[target_index-1:target_index+context_length-1], label="ave10", color="green")
#ax.plot(data_loader.test.index[target_index+context_length-1:target_index+context_length+prediction_length], data_loader.test[target_cols[0]].values[target_index+context_length-1:target_index+context_length+prediction_length], label="target", color="blue")

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")

#ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_locator(mdates.MonthLocator())

fig.savefig("output_test.png")
plt.close(fig)

fig, ax = plt.subplots()

target_index = 1600
ax.plot(data_loader.train.index[target_index:target_index+context_length], data_loader.train[target_cols[0]].values[target_index:target_index+context_length], label="input", color="red")
#ax.plot(data_loader.train.index[target_index+context_length-1:target_index+context_length+prediction_length], data_loader.train[target_cols[0]].values[target_index+context_length-1:target_index+context_length+prediction_length], label="target", color="blue")

#ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

fig.savefig("output_train.png")

plt.close(fig)

fig, ax = plt.subplots()

ax.plot(data_loader.train.index, data_loader.train[target_cols[0]].values, label="train", color="blue")
ax.plot(data_loader.val.index, data_loader.val[target_cols[0]].values, label="val", color="green")
ax.plot(data_loader.test.index, data_loader.test[target_cols[0]].values, label="test", color="red")

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.xaxis.set_major_locator(mdates.YearLocator(1))

fig.savefig("output_total.png")