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

def main():
    base_asset = "yen"
    target_assets = ["dollar"]

    assets = Assets([base_asset] + target_assets)
    assets.get(base_asset, 10000)

    print(assets.yen.possession)
    print(assets.dollar.possession)

    assets.exchange("yen", "dollar", 0.01, 5000)

    print(assets.yen.possession)
    print(assets.dollar.possession)

    assets.exchange("dollar", "yen", 101, 50)

    print(assets.yen.possession)
    print(assets.dollar.possession)

if __name__ == '__main__':
    main()

    