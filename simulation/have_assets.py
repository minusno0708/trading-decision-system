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

from simulation.strutegy import *
from simulation.assets import *

from args_parser import parse_args

input_length = 30
output_length = 1

trade_rate = 0.1
start_dollar = 1000

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

    for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(data_loader.test_dataset(batch_size=1, is_shuffle=False)):
        if is_pre_scaling:
            input_x_inverse = []

            for c, col in enumerate(target_cols):
                input_x_inverse.append(data_loader.inverse_transform(input_x[:,:,c].detach().numpy().squeeze(0), col).squeeze(0))

            input_x = np.array(input_x_inverse)
        else:
            input_x = input_x.squeeze(0).permute(1, 0).detach().numpy()

        current_rate = input_x[0][-1]

        # 仮想通貨の価格を更新
        rate.update([
            current_rate,
        ])

        break

    assets.trade(base_asset, target_assets[0], rate.buy(target_assets[0]), assets[base_asset].possession * 0.5)

    print(f"start: {base_asset} {assets[base_asset].possession}, {target_assets[0]} {assets[target_assets[0]].possession}")

    strutegy = Strategy(decision_method)

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

        action = strutegy.decide_action(history_values, forecast_values, future_values)

        if action == "buy":
            assets.trade(base_asset, target_assets[0], rate.buy(target_assets[0]), assets[base_asset].possession * trade_rate)
        elif action == "sell":
            assets.trade(target_assets[0], base_asset, rate.sell(target_assets[0]), assets[target_assets[0]].possession * trade_rate)
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
        dollar_log.append(current_rate, assets[base_asset].possession)
        crypto_log.append(current_rate, assets[target_assets[0]].possession)
        total_log.append(current_rate, assets[base_asset].possession + assets[target_assets[0]].possession * rate.rates[0])
        

    rate_log.plot(f"{target_assets[0]}-rate", f"{target_assets[0]}/USD Rate")
    dollar_log.plot(f"{target_assets[0]}-{base_asset}", f"{base_asset} Possession")
    crypto_log.plot(f"{target_assets[0]}-{target_assets[0]}", f"{target_assets[0]} Possession")
    total_log.plot(f"{target_assets[0]}-total", "Total Possession")

    assets.trade(f"{target_assets[0]}", base_asset, rate.sell(f"{target_assets[0]}"), assets[target_assets[0]].possession)

    print(f"result: {base_asset} {assets[base_asset].possession}, {target_assets[0]} {assets[target_assets[0]].possession}")

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
    args = parse_args()

    seed = args.seed

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
    