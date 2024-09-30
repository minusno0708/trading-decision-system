import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_provider.data_loader import DataLoader
import matplotlib.dates as mdates

import torch
from model.DeepAR import Model


from strutegy import *


input_length = 30
output_length = 7

trade_rate = 0.1
start_yen = 100000

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_flag = True


class Log:
    def __init__(self):
        self.index = []
        self.values = []

    def append(self, index, value):
        self.index.append(index)
        self.values.append(value)

    def plot(self, name, title = None):
        fig, ax = plt.subplots()
        ax.plot(self.index, self.values)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        if title:
            plt.title(title)
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
    train_data, test_data = data_loader.load(f"{target_assets[0]}.csv")

    model = Model(input_length, output_length)
    if train_flag:
        model.train(train_data)
    else:
        model.load(f"output/models/{target_assets[0]}_model")

    rate_log = Log()
    yen_log = Log()
    crypto_log = Log()

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

        #action = diff_next_mean(history_values, forecast_values)
        action = random_decision()
        #action = all_lose(history_values, tommorow_data.values[0][0])

        if action == "buy":
            assets.trade("yen", "btc", rate.buy("btc"), assets.yen.possession * trade_rate)
        elif action == "sell":
            assets.trade("btc", "yen", rate.sell("btc"), assets.btc.possession * trade_rate)
        else:
            pass

        #print(f"action: {action}, current: {history_values[-1]}, forecast: {forecast_values[0]}")

        # 結果を出力
        current_date = target_raw_data.index[-1]
        #print(f"date: {current_date}, action: {action}, result: yen {assets.yen.possession}, btc {assets.btc.possession}")

        # ログを保存
        rate_log.append(current_date, rate.rates[0])
        yen_log.append(current_date, assets.yen.possession)
        crypto_log.append(current_date, assets.btc.possession)
        

    rate_log.plot(f"{target_assets[0]}-rate", "BTC/JPY Rate")
    yen_log.plot(f"{target_assets[0]}-yen", "Yen Possession")
    crypto_log.plot(f"{target_assets[0]}-{target_assets[0]}", "BTC Possession")

    assets.trade(f"{target_assets[0]}", "yen", rate.sell(f"{target_assets[0]}"), assets.btc.possession)

    print(f"result: yen {assets.yen.possession}, {target_assets[0]} {assets.btc.possession}")

if __name__ == '__main__':
    main()
    