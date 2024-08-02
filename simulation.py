import numpy as np

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
    target_assets = ["dollar"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, 10000)

    rate = ExchangeRate(base_asset, target_assets)
    rate.update([0.01])

    print(assets.yen.possession)
    print(assets.dollar.possession)

    assets.exchange("yen", "dollar", rate.buy("dollar"), 5000)

    print(assets.yen.possession)
    print(assets.dollar.possession)

    rate.update([0.009])
    assets.exchange("dollar", "yen", rate.sell("dollar"), 50)

    print(assets.yen.possession)
    print(assets.dollar.possession)

if __name__ == '__main__':
    main()

    