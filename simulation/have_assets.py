import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import torch

from datetime import datetime

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_provider.data_loader import DataLoader
from data_provider.custom_data_provider import CustomDataProvider

from model.custom import Model

from strutegy import *

input_length = 30
output_length = 1

trade_rate = 0.1
start_dollar = 1000

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

decision_method = "diff_next_mean"

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

def main(
    seed,
    output_path,
    exp_name,
    data_path,
    index_col,
    target_cols,
    train_start_date,
    train_end_date,
    test_start_date,
    test_end_date,
    train_data_length,
    test_data_length,
    val_data_length,
    split_type,
    prediction_length,
    context_length,
    epochs,
    num_batches,
    num_parallel_samples,
    is_pre_scaling,
    is_model_scaling,
    add_time_features,
    add_extention_features
):
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

    data_loader = CustomDataProvider(
        file_path=data_path,
        index_col=index_col,
        target_cols=target_cols,
        prediction_length=prediction_length,
        context_length=context_length,
        freq="D",
        scaler_flag=is_pre_scaling,
    )

    if split_type == "datetime":
        data_loader.update_date(train_start_date, train_end_date, test_start_date, test_end_date)
    elif split_type == "index":
        data_loader.update_date_by_index(test_start_date, train_data_length, test_data_length, val_data_length)

    model = Model(
        context_length=context_length,
        prediction_length=prediction_length,
        freq="D",
        epochs=epochs,
        num_parallel_samples=num_parallel_samples,
        target_dim=len(target_cols),
        is_scaling=is_model_scaling,
        add_time_features=add_time_features,
        add_extention_features=add_extention_features,
    )

    if train_flag:
        train_loss, val_loss, minimal_val_loss = model.train(data_loader.train_dataset(batch_size=num_batches, is_shuffle=False), data_loader.val_dataset(batch_size=1, is_shuffle=False))
    else:
        model.load()

    rate_log = Log()
    dollar_log = Log()
    crypto_log = Log()
    total_log = Log()

    print(f"start: dollar {assets.dollar.possession}, {target_assets[0]} {assets.btc.possession}")

    for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(data_loader.test_dataset(batch_size=1, is_shuffle=False)):
        # 価格を予測
        forecasts, loss = model.make_evaluation_predictions(input_x, target_x, time_features, extention_features)

        if is_pre_scaling:
            input_x_inverse = []
            target_x_inverse = []

            for c, col in enumerate(target_cols):
                forecasts[c].inverse_transform(data_loader.scaler[col])
                input_x_inverse.append(data_loader.inverse_transform(input_x[:,:,c].detach().numpy().squeeze(0), col).squeeze(0))
                target_x_inverse.append(data_loader.inverse_transform(target_x[:,:,c].detach().numpy().squeeze(0), col).squeeze(0))

            input_x = np.array(input_x_inverse)
            target_x = np.array(target_x_inverse)
        else:
            input_x = input_x.squeeze(0).permute(1, 0).detach().numpy()
            target_x = target_x.squeeze(0).permute(1, 0).detach().numpy()

        current_rate = input_x[0][-1]

        # 仮想通貨の価格を更新
        rate.update([
            current_rate,
        ])

        # 取引開始
        history_values = input_x[0]
        future_values = target_x[0]
        forecast_values = forecasts[0].mean

        if decision_method == "diff_next_mean":
            action = diff_next_mean(history_values, forecast_values)
        elif decision_method == "random":
            action = random_decision()
        elif decision_method == "all_win":
            action = all_true(history_values, future_values[0])
        elif decision_method == "all_lose":
            action = all_false(history_values, future_values[0])
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

        if action == "buy":
            assets.trade("dollar", "btc", rate.buy("btc"), assets.dollar.possession * trade_rate)
        elif action == "sell":
            assets.trade("btc", "dollar", rate.sell("btc"), assets.btc.possession * trade_rate)
        else:
            pass

        if action == "buy":
            if history_values[-1] < future_values[0]:
                actions_result["buy_win"] += 1
            elif history_values[-1] > future_values[0]:
                actions_result["buy_lose"] += 1
            else:
                print("Error")
        elif action == "sell":
            if history_values[-1] > future_values[0]:
                actions_result["sell_win"] += 1
            elif history_values[-1] < future_values[0]:
                actions_result["sell_lose"] += 1
            else:
                print("Error")

        # ログを保存
        rate_log.append(current_rate, rate.rates[0])
        dollar_log.append(current_rate, assets.dollar.possession)
        crypto_log.append(current_rate, assets.btc.possession)
        total_log.append(current_rate, assets.dollar.possession + assets.btc.possession * rate.rates[0])
        

    rate_log.plot(f"{target_assets[0]}-rate", "btc/USD Rate")
    dollar_log.plot(f"{target_assets[0]}-dollar", "dollar Possession")
    crypto_log.plot(f"{target_assets[0]}-{target_assets[0]}", "btc Possession")
    total_log.plot(f"{target_assets[0]}-total", "Total Possession")

    assets.trade(f"{target_assets[0]}", "dollar", rate.sell(f"{target_assets[0]}"), assets.btc.possession)

    print(f"result: dollar {assets.dollar.possession}, {target_assets[0]} {assets.btc.possession}")

    print(f"actions_result: {actions_result}")

def int2bool(v):
    v = int(v)
    if v == 0:
        return False
    else:
        return True

def str2datetime(v):
    if len(v.split("T")) == 2:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
    else:
        return datetime.strptime(v, "%Y-%m-%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="output/forecast")
    parser.add_argument("--exp_name", type=str, default="exp_default")

    parser.add_argument("--data_path", type=str, default="dataset/btc.csv")
    parser.add_argument("--index_col", type=str, default="timeOpen")
    parser.add_argument("--target_cols", type=str, default=["close"])

    parser.add_argument("--train_start_date", type=str2datetime, default=None)
    parser.add_argument("--train_end_date", type=str2datetime, default=None)
    parser.add_argument("--test_start_date", type=str2datetime, default=None)
    parser.add_argument("--test_end_date", type=str2datetime, default=None)
    
    parser.add_argument("--train_data_length", type=int, default=60)
    parser.add_argument("--test_data_length", type=int, default=30)
    parser.add_argument("--val_data_length", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="datetime")

    parser.add_argument("--prediction_length", type=int, default=30)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--num_parallel_samples", type=int, default=1000)

    parser.add_argument("--is_pre_scaling", type=int2bool, default=True)

    parser.add_argument("--is_model_scaling", type=int2bool, default=False)
    parser.add_argument("--add_time_features", type=int2bool, default=False)
    parser.add_argument("--add_extention_features", type=int2bool, default=False)

    args = parser.parse_args()

    seed = args.seed

    if not os.path.exists(f"{args.output_path}/images/{args.exp_name}"):
        os.makedirs(f"{args.output_path}/images/{args.exp_name}")

    if not os.path.exists(f"{args.output_path}/logs"):
        os.makedirs(f"{args.output_path}/logs")

    if args.split_type == "datetime":
        if (args.train_end_date - args.train_start_date).days < args.context_length + args.prediction_length:
            raise ValueError(f"train data must be longer than {args.context_length + args.prediction_length}: {args.train_end_date - args.train_start_date}")

        if (args.test_end_date - args.test_start_date).days < args.prediction_length:
            raise ValueError(f"test data must be longer than {args.prediction_length}: {args.test_end_date - args.test_start_date}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(
        seed=seed,
        output_path=args.output_path,
        exp_name=args.exp_name,
        data_path=args.data_path,
        index_col=args.index_col,
        target_cols=args.target_cols.split(","),
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        train_data_length=args.train_data_length,
        test_data_length=args.test_data_length,
        val_data_length=args.val_data_length,
        split_type=args.split_type,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        epochs=args.epochs,
        num_batches=args.num_batches,
        num_parallel_samples=args.num_parallel_samples,
        is_pre_scaling=args.is_pre_scaling,
        is_model_scaling=args.is_model_scaling,
        add_time_features=args.add_time_features,
        add_extention_features=args.add_extention_features,
    )
    