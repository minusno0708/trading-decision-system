import numpy as np

from data_provider.data_loader import DataLoader

from model.DeepAR import Model

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

    model = Model(input_length, output_length).load("output/models/model.pth")

    target_data = test_data.iloc[range(0, input_length), [0]]

    print(target_data.values[-1])
    print(data_loader.inverse_transform(target_data.values[-1]))

    rate.update([0.01])

    print(assets.yen.possession)
    print(assets.btc.possession)

    assets.exchange("yen", "btc", rate.buy("btc"), 5000)

    print(assets.yen.possession)
    print(assets.btc.possession)

    rate.update([0.009])
    assets.exchange("btc", "yen", rate.sell("btc"), 50)

    print(assets.yen.possession)
    print(assets.btc.possession)

if __name__ == '__main__':
    main()

    