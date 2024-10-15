import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_provider.data_loader import DataLoader
from model.DeepAR import Model
from strutegy import *

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

decision_method = "diff_next_mean"

train_flag = True

output_dir = "output"

input_length = 30
output_length = 7

dollor_sell_amount = 100
btc_sell_amount = 0

atempt_name = decision_method + "-" + str(seed)

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

class Bank:
    def __init__(self):
        self.account = 0

    def put(self, amount):
        self.account += amount
    
    def out(self, amount):
        self.account -= amount

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
        "buy_true": 0,
        "buy_false": 0,
        "sell_true": 0,
        "sell_false": 0,
    }

    base_asset = "dollar"
    target_assets = ["btc"]

    bank = Bank()
    btc_hold = 0
    btc_shorted = 0

    log = Log()

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
        elif decision_method == "all_true":
            action = all_true(history_values, tommorow_data.values[0][0])
        elif decision_method == "all_false":
            action = all_false(history_values, tommorow_data.values[0][0])
        elif decision_method == "all_buy":
            action = "buy"
        elif decision_method == "all_sell":
            action = "sell"
        else:
            print("method not found")
            return

        current_date = target_raw_data.index[-1]

        # 保持しているビットコインを売却
        bank.put(btc_hold * rate.sell("btc"))
        btc_hold = 0
        # 空売りしたビットコイン分のドルを返却
        bank.out(btc_shorted * rate.sell("btc"))
        btc_shorted = 0

        log.append(current_date, bank.account)

        # 取引
        if action == "buy":
            # 100ドル分のビットコインを購入
            bank.out(dollor_sell_amount)
            btc_hold = dollor_sell_amount * rate.buy("btc")
        elif action == "sell":
            # 100ドル分のBTCを空売り
            btc_sell_amount = dollor_sell_amount * rate.buy("btc")
            bank.put(btc_sell_amount * rate.sell("btc"))
            btc_shorted = btc_sell_amount
        else:
            pass

        # 行動とその結果を記録
        if action == "buy":
            if history_values[-1] < tommorow_data.values[0][0]:
                actions_result["buy_true"] += 1
            elif history_values[-1] > tommorow_data.values[0][0]:
                actions_result["buy_false"] += 1
            else:
                print("Error")
        elif action == "sell":
            if history_values[-1] > tommorow_data.values[0][0]:
                actions_result["sell_true"] += 1
            elif history_values[-1] < tommorow_data.values[0][0]:
                actions_result["sell_false"] += 1
            else:
                print("Error")

    bank.put(btc_hold * rate.sell("btc"))
    bank.out(btc_shorted * rate.sell("btc"))
    log.append(current_date, bank.account)

    log.plot("bank-" + atempt_name, "損益額($)")
    with open(f"{output_dir}/bank.txt", "a") as f:
        f.write(f"action: {action}, seed: {seed}\n")
        f.write("bank: " + str(bank.account) + "\n")
        f.write("actions_result: " + str(actions_result) + "\n")
    print(f"action: {action}, seed: {seed}")
    print("bank: ", bank.account)
    print("actions_result: ", actions_result)

if __name__ == '__main__':
    print("desicion method: ", decision_method)
    for i in range(5):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        main()
    