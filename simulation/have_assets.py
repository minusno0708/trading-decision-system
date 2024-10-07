import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_provider.data_loader import DataLoader
from model.DeepAR import Model
from strutegy import *


input_length = 30
output_length = 1

trade_rate = 0.1
start_dollar = 1000

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

decision_method = "all_lose"

train_flag = False

output_dir = "output"

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
        plt.savefig(f"{output_dir}/images/{name}.png")
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
    actions_result = {
        "buy_win": 0,
        "buy_lose": 0,
        "sell_win": 0,
        "sell_lose": 0,
    }

    base_asset = "dollar"
    target_assets = ["btc"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, start_dollar)

    rate = TradeRate(base_asset, target_assets)

    data_loader = DataLoader(output_length)
    _, raw_data = data_loader.load("btc.csv", False)
    train_data, test_data = data_loader.load(f"{target_assets[0]}.csv")

    model = Model(input_length, output_length)
    if train_flag:
        model.train(train_data)
        model.save(f"{output_dir}/models/{target_assets[0]}_model")
    else:
        model.load(f"{output_dir}/models/{target_assets[0]}_model")

    rate_log = Log()
    dollar_log = Log()
    crypto_log = Log()
    total_log = Log()

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

        if decision_method == "diff_next_mean":
            action = diff_next_mean(history_values, forecast_values)
        elif decision_method == "random":
            action = random_decision()
        elif decision_method == "all_win":
            action = all_win(history_values, tommorow_data.values[0][0])
        elif decision_method == "all_lose":
            action = all_lose(history_values, tommorow_data.values[0][0])
        elif decision_method == "all_buy":
            action = "buy"
        elif decision_method == "all_sell":
            action = "sell"
        elif decision_method == "cross_action":
            if not "action" in locals():
                action = "sell"
            elif action == "buy":
                action = "sell"
            elif action == "sell":
                action = "buy"

        current_date = target_raw_data.index[-1]

        if action == "buy":
            assets.trade("dollar", "btc", rate.buy("btc"), assets.dollar.possession * trade_rate)
        elif action == "sell":
            assets.trade("btc", "dollar", rate.sell("btc"), assets.btc.possession * trade_rate)
        else:
            pass

        if action == "buy":
            if history_values[-1] < tommorow_data.values[0][0]:
                actions_result["buy_win"] += 1
            elif history_values[-1] > tommorow_data.values[0][0]:
                actions_result["buy_lose"] += 1
            else:
                print("Error")
        elif action == "sell":
            if history_values[-1] > tommorow_data.values[0][0]:
                actions_result["sell_win"] += 1
            elif history_values[-1] < tommorow_data.values[0][0]:
                actions_result["sell_lose"] += 1
            else:
                print("Error")

        # ログを保存
        rate_log.append(current_date, rate.rates[0])
        dollar_log.append(current_date, assets.dollar.possession)
        crypto_log.append(current_date, assets.btc.possession)
        total_log.append(current_date, assets.dollar.possession + assets.btc.possession * rate.rates[0])
        

    rate_log.plot(f"{target_assets[0]}-rate", "btc/USD Rate")
    dollar_log.plot(f"{target_assets[0]}-dollar", "dollar Possession")
    crypto_log.plot(f"{target_assets[0]}-{target_assets[0]}", "btc Possession")
    total_log.plot(f"{target_assets[0]}-total", "Total Possession")

    assets.trade(f"{target_assets[0]}", "dollar", rate.sell(f"{target_assets[0]}"), assets.btc.possession)

    print(f"result: dollar {assets.dollar.possession}, {target_assets[0]} {assets.btc.possession}")

    print(f"actions_result: {actions_result}")

if __name__ == '__main__':
    print("desicion method: ", decision_method)
    main()
    