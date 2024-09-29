import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_provider.data_loader import DataLoader

import torch
from model.DeepAR import Model

np.random.seed(0)
torch.manual_seed(0)

from strutegy import *


input_length = 30
output_length = 7

trade_rate = 0.01
start_yen = 100000


class Log:
    def __init__(self):
        self.index = []
        self.values = []

    def append(self, index, value):
        self.index.append(index)
        self.values.append(value)

    def plot(self, name):
        plt.plot(self.index, self.values)
        plt.savefig(f"output/images/{name}.png")
        plt.clf()
        plt.close()

class Asset:
    def __init__(self, name):
        self.name = name
        self.possession = 0

    def buy(self, amount):
        self.possession += amount
    
    def sell(self, amount):
        self.possession -= amount

class Assets:
    def __init__(self, assets):
        for asset in assets:
            setattr(self, asset, Asset(asset))

    def get(self, asset, amount):
        self.__dict__[asset].buy(amount)
        
    def trade(self, from_asset, to_asset, rate, amount):
        if self.__dict__[from_asset].possession < amount:
            amount = self.__dict__[from_asset].possession

        self.__dict__[from_asset].sell(amount)
        self.__dict__[to_asset].buy(amount * rate)

class TradeRate:
    def __init__(self, base_asset, target_assets):
        self.base_asset = base_asset
        self.target_assets = target_assets

        self.rates = np.zeros(len(target_assets))

    def update(self, rates):
        self.rates = rates

    def buy(self, target_asset):
        return 1 / self.rates[self.target_assets.index(target_asset)]

    def sell(self, target_asset):
        return self.rates[self.target_assets.index(target_asset)]

def main():
    base_asset = "yen"
    target_assets = ["btc"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, start_yen)

    rate = TradeRate(base_asset, target_assets)

    data_loader = DataLoader(output_length)
    _, raw_data = data_loader.load("btc.csv", False)
    _, test_data = data_loader.load("btc.csv")

    model = Model(input_length, output_length)
    model.load("output/models/btc_model")

    rate_log = Log()
    yen_log = Log()
    btc_log = Log()

    for d in range(0, len(raw_data) - input_length - output_length):
        target_scaled_data = test_data.iloc[range(d, input_length + d), [0]]
        target_raw_data = raw_data.iloc[range(d, input_length + d), [0]]
        tommorow_data = raw_data.iloc[range(input_length + d, input_length + d + 1), [0]]

        # 仮想通貨の価格を更新
        rate.update([
            target_raw_data.values[-1][0]
        ])

        # 明日以降の価格を予測
        forecasts = model.forecast(target_scaled_data)

        # 取引開始
        history_values = data_loader.inverse_transform(target_scaled_data.values.reshape(-1))[0]
        forecast_values = data_loader.inverse_transform(forecasts[0].mean)[0]

        action = diff_next_mean(history_values, forecast_values)

        if action == "buy":
            assets.trade("yen", "btc", rate.buy("btc"), assets.yen.possession * trade_rate)
        elif action == "sell":
            assets.trade("btc", "yen", rate.sell("btc"), assets.btc.possession * trade_rate)
        else:
            pass

        # 結果を出力
        current_date = target_raw_data.index[-1]
        print(f"date: {current_date}, action: {action}, result: yen {assets.yen.possession}, btc {assets.btc.possession}")

        # ログを保存
        rate_log.append(current_date, rate.rates[0])
        yen_log.append(current_date, assets.yen.possession)
        btc_log.append(current_date, assets.btc.possession)
        

    rate_log.plot("rate")
    yen_log.plot("yen")
    btc_log.plot("btc")

    assets.trade("btc", "yen", rate.sell("btc"), assets.btc.possession)

    print(f"result: yen {assets.yen.possession}, btc {assets.btc.possession}")


if __name__ == '__main__':
    main()

    