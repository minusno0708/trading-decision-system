import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

import torch
import mxnet as mx

import datetime

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_provider.data_loader import DataLoader
from data_provider.custom_data_provider import CustomDataProvider

from model.custom import Model

from simulation.strutegy import *
from simulation.assets import *

from args_parser import parse_args

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
mx.random.seed(seed)

decision_method = "diff_next_mean"

train_flag = False

output_dir = "output"

input_length = 30
output_length = 30

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
        "buy_true": 0,
        "buy_false": 0,
        "sell_true": 0,
        "sell_false": 0,
    }

    base_asset = "dollar"
    target_assets = ["btc"]

    start_dollar = 0

    bank = Bank()
    bank.put(start_dollar)

    btc_hold = 0
    btc_shorted = 0

    log = Log()

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

        current_date = history_values[-1]

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
            if history_values[-1] < future_values[0]:
                actions_result["buy_true"] += 1
            elif history_values[-1] > future_values[0]:
                actions_result["buy_false"] += 1
            else:
                print("Error")
        elif action == "sell":
            if history_values[-1] > future_values[0]:
                actions_result["sell_true"] += 1
            elif history_values[-1] < future_values[0]:
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
    