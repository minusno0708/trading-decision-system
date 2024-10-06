import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from data_provider.data_loader import DataLoader
import matplotlib.dates as mdates

import torch

from model.DeepAR import Model
from strutegy import *

import seaborn as sns


input_length = 30
output_length = 1

trade_rate = 0.1
start_dollar = 1000

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

decision_method = "random"

train_flag = False

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
    base_asset = "dollar"
    target_assets = ["btc"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, start_dollar)

    rate = TradeRate(base_asset, target_assets)

    data_loader = DataLoader(output_length)
    _, raw_data = data_loader.load("btc.csv", False)
    train_data, test_data = data_loader.load(f"{target_assets[0]}.csv")

    for d in range(0, len(raw_data) - input_length - output_length):
        target_scaled_data = test_data.iloc[range(d, input_length + d), [0]]
        target_raw_data = raw_data.iloc[range(d, input_length + d), [0]]

        # 仮想通貨の価格を更新
        rate.update([
            target_raw_data.values[-1][0]
        ])

        # 取引開始
        history_values = data_loader.inverse_transform(target_scaled_data.values.reshape(-1))[0]
      
        action = random_decision()

        if action == "buy":
            assets.trade("dollar", "btc", rate.buy("btc"), assets.dollar.possession * trade_rate)
        elif action == "sell":
            assets.trade("btc", "dollar", rate.sell("btc"), assets.btc.possession * trade_rate)
        else:
            pass
        

    assets.trade(f"{target_assets[0]}", "dollar", rate.sell(f"{target_assets[0]}"), assets.btc.possession)

    return assets.dollar.possession + assets.btc.possession * rate.rates[0]

from collections import Counter

bins = 50
def formatter(x, pos):
    return f"{int(x * bins)}"

if __name__ == '__main__':
    simulation_flag = False

    if simulation_flag:
        result = []
        for i in range(10000):
            print(i)
            random.seed(i)
            result.append(main())
        
        print(result)
        with open("output/random_result.txt", "w") as f:
            for r in result:
                f.write(f"{r}\n")
    else:
        result = []
        with open("output/random_result.txt", "r") as f:
            for line in f:
                result.append(float(line.strip()))

    distribution = Counter(int((r // bins)) for r in result)
    print(distribution)

    for rounded_value, count in sorted(distribution.items()):
        plt.bar(rounded_value, count, color="blue")

    plt.gca().xaxis.set_major_formatter(FuncFormatter(formatter))
    #plt.xticks(range(min(distribution.keys()), max(distribution.keys()) + 1, 1))

    plt.ylabel("Count")
    plt.xlabel("Dollar")

    plt.savefig("output/images/btc_random_hist.png")
    plt.close()

    sns.histplot(result, bins=100)
    plt.savefig("output/images/btc_random_hist2.png")
    