import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_provider.data_loader import DataLoader

from model.DeepAR import Model

trade_rate = 0.1

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
        
    def exchange(self, from_asset, to_asset, rate, amount):
        if self.__dict__[from_asset].possession < amount:
            amount = self.__dict__[from_asset].possession

        self.__dict__[from_asset].sell(amount)
        self.__dict__[to_asset].buy(amount * rate)

class ExchangeRate:
    def __init__(self, base_asset, target_assets):
        self.base_asset = base_asset
        self.target_assets = target_assets

        self.rates = np.zeros(len(target_assets))

    def update(self, rates):
        self.rates = rates

    def buy(self, target_asset):
        return self.rates[self.target_assets.index(target_asset)]

    def sell(self, target_asset):
        return 1 / self.rates[self.target_assets.index(target_asset)]


def is_buy_signal(today_price, tomorrow_price):
    return today_price < tomorrow_price

def is_sell_signal(today_price, tomorrow_price):
    return today_price > tomorrow_price

def day_trade(assets: Assets, rate: ExchangeRate, model: Model, data_loader: DataLoader, data: pd.DataFrame):
    rate.update(
        data_loader.inverse_transform(data.values[-1])[0]
    )

    forecasts = model.forecast(data)

    today_price = data.values[-1][0]
    tomorrow_price = forecasts[0].mean[0]

    if is_buy_signal(today_price, tomorrow_price):
        assets.exchange("yen", "btc", rate.buy("btc"), assets.yen.possession * trade_rate)
    elif is_sell_signal(today_price, tomorrow_price):
        assets.exchange("btc", "yen", rate.sell("btc"), assets.btc.possession * trade_rate)

def main():
    base_asset = "yen"
    target_assets = ["btc"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, 10000)

    rate = ExchangeRate(base_asset, target_assets)

    input_length = 30
    output_length = 7

    data_loader = DataLoader(output_length)
    _, test_data = data_loader.load("btc.csv")

    model = Model(input_length, output_length)
    model.load("output/models/model.pth")

    rate_log = Log()
    yen_log = Log()
    btc_log = Log()

    for d in range(0, len(test_data) - input_length - output_length):
        target_data = test_data.iloc[range(d, input_length + d), [0]]
        day_trade(assets, rate, model, data_loader, target_data)

        today_date = target_data.index[-1]

        print(f"{today_date}: yen {assets.yen.possession}, btc {assets.btc.possession}")

        rate_log.append(today_date, rate.rates[0])
        yen_log.append(today_date, assets.yen.possession)
        btc_log.append(today_date, assets.btc.possession)
        

    rate_log.plot("rate")
    yen_log.plot("yen")
    btc_log.plot("btc")


if __name__ == '__main__':
    main()

    