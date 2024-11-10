import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider
from model import Model

import numpy as np
import pandas as pd
import torch
import mxnet as mx

import datetime
import math
import statistics

from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

import properscoring as ps

input_length = 30
output_length = 30

train_flag = True
evaluation_mode = [
    #"backtest",
    #"crps",
    #"updown",
    "diff_price",
]

model_type = "torch"
model_name = "deepar"

seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
mx.random.seed(seed)

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/images/{name}.png")

    plt.clf()
    plt.close()

def draw_predict_graph(input_data: pd.DataFrame, forecasts: list, correct_data: pd.DataFrame, crypto: str, num: int):
    fig, ax = plt.subplots()

    ax.plot(input_data.index, input_data.values, label="input")
    ax.plot(correct_data.index, correct_data.values, label="correct")
    

    forecasts[0].plot(intervals=(0.3, 0.8))
    ax.plot(forecasts[0].index, forecasts[0].median, label="forecast-median")
    ax.plot(forecasts[0].index, forecasts[0].mean, label="forecast-mean")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.legend()
    plt.xlabel("Date")
    plt.savefig(f"output/images/forecast/{num}-{crypto}-forecast.png")

def rmse(y_true, y_pred):
    return np.sqrt((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def likelihood(y_true, y_pred_prob):
    ave = statistics.mean(y_pred_prob)
    var = statistics.pvariance(y_pred_prob)

    return pow(2 * math.pi * var, -1/2) * math.exp(-math.pow((y_true - ave), 2) / (2 * var))


def log_likelihood(y_true, y_pred_prob):
    loss_sum = []
    for i in range(len(y_true)):
        lh = likelihood(y_true[i], y_pred_prob[:, i])
        if lh == 0:
            lh = 1e-10
        loss_sum.append(math.log(lh))

    return np.mean(loss_sum)

def main(train_start_year: int = 2010, test_start_year: int = 2023):
    crypto = "btc"

    data_loader = GluontsDataProvider(
        file_path=f"dataset/{crypto}.csv",
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        scaler_flag=True
    )

    crps_error = np.array([])
    mean_price_diff = np.empty((0, output_length))
    mean_correct = np.array([])

    forecast_updown = np.array([])
    correct_updown = np.array([])

    model = Model(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        epochs=1,
        num_parallel_samples=1000,
        model_name=model_name,
        model_type=model_type
    )

    # training
    if train_flag:
        train_log = model.train(data_loader.train_dataset())
        model.save(f"output/models/{crypto}_model")
        print(train_log)
    else:
        model.load(f"output/models/{crypto}_model")

    # evaluation
    if "backtest" in evaluation_mode:
        agg_metrics, item_metrics = model.backtest(data_loader.train_dataset())

        print("Train")
        print("agg metrics:", agg_metrics)
        print("item_metrics:", item_metrics)

        agg_metrics, item_metrics = model.backtest(data_loader.test_dataset())

        print("Test")
        print("agg metrics:", agg_metrics)
        print("item_metrics:", item_metrics)

    for i in range(data_loader.test_length()):
        if i % 100 == 0:
            print(f"Forecasting {i} Days")

        target_data, correct_data = data_loader.test_prediction_data(i)
        correct_data = data_loader.listdata_values(correct_data)

        forecasts = model.forecast(target_data)

        # CRPS 誤差の計算
        if "crps" in evaluation_mode:  
            for d in range(output_length):
                crps_error = np.append(crps_error, ps.crps_ensemble(correct_data[d], forecasts[0].samples[:, d]))

        # 実際の価格との差分の計算
        if "diff_price" in evaluation_mode:
            correct_inverse = data_loader.inverse_transform(correct_data)[0]
            mean_inverse = data_loader.inverse_transform(forecasts[0].mean)[0]

            mean_price_diff = np.append(mean_price_diff, [rmse(correct_inverse, mean_inverse)], axis=0)

        # 正解と予測の上下の判定
        if "updown" in evaluation_mode:
            past_day_value = data_loader.listdata_values(target_data)[-1]

            correct_flag = past_day_value < correct_data[0]
            correct_updown = np.append(correct_updown, correct_flag)

            forecast_mean_flag = past_day_value < forecasts[0].mean[0]
            forecast_updown = np.append(forecast_updown, forecast_mean_flag)

            mean_correct = np.append(mean_correct, correct_flag == forecast_mean_flag)
    
    if "crps" in evaluation_mode:  
        print(f"CRPS Loss: {crps_error.mean()}")

    if "diff_price" in evaluation_mode:
        print(f"MeanPriceDiff: {mean_price_diff.mean(axis=0)}")

    if "updown" in evaluation_mode:
        print(f"MeanCorrect: {mean_correct.mean()}")
        print(f"CorrectUp Rate: {correct_updown.mean()}")
        print(f"ForecastUp Rate: {forecast_updown.mean()}")


if __name__ == "__main__":
    main()
