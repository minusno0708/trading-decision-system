import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader import DataLoader
from model.DeepAR import TorchModel as Model
#from model.DeepAR import MxModel as Model

import numpy as np
import pandas as pd
import torch
import mxnet as mx

import datetime
import math
import statistics

input_length = 30
output_length = 30

train_flag = False

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

    return pow(2 * math.pi * math.pow(var, 2), -1/2) * math.exp(-math.pow((y_true - ave), 2) / (2 * pow(var, 2)))


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

    data_loader = DataLoader(input_length)
    train_data, test_data = data_loader.load(f"{crypto}.csv", True, datetime.datetime(train_start_year, 1, 1), datetime.datetime(test_start_year, 1, 1))

    loss = np.array([])
    mean_price_diff = np.empty((0, output_length))
    mean_correct = np.array([])

    forecast_updown = np.array([])
    correct_updown = np.array([])

    model = Model(input_length, output_length)

    if train_flag:
        model.train(train_data)
        model.save(f"output/models/{crypto}_model")
    else:
        model.load(f"output/models/{crypto}_model")

    for i in range(0, len(test_data) - input_length - output_length):

        target_data = test_data.iloc[range(i, input_length + i), [0]]
        correct_data = test_data.iloc[range(i + input_length, input_length + i + output_length), [0]]

        forecasts = model.forecast(target_data)

        loss = np.append(loss, log_likelihood(correct_data.values.flatten(), forecasts[0].samples))
        
        if i % 50 == 0:
            print(f"Forecasting: {i}")
            draw_predict_graph(target_data, forecasts, correct_data, crypto, i)

        correct_inverse = data_loader.inverse_transform(correct_data.values.flatten())[0]
        mean_inverse = data_loader.inverse_transform(forecasts[0].mean)[0]

        mean_price_diff = np.append(mean_price_diff, [rmse(correct_inverse, mean_inverse)], axis=0)

        correct_flag = target_data.values[-1] < correct_data.values.flatten()[0]
        mean_flag = target_data.values[-1] < forecasts[0].mean[0]

        correct_updown = np.append(correct_updown, correct_flag)
        forecast_updown = np.append(forecast_updown, mean_flag)

        mean_correct = np.append(mean_correct, correct_flag == mean_flag)
    
    """
    with open("test.txt", "a") as f:
        f.write(f"Train: {train_start_year}, Test: {test_start_year}\n")
        
        f.write(f"MeanLoss: {mean_price_diff.mean(axis=0)}\n")
        f.write(f"MeanCorrect: {mean_correct.mean()}\n")
        f.write(f"CorrectUp Rate: {correct_updown.mean()}\n")
        f.write(f"ForecastUp Rate: {forecast_updown.mean()}\n")

        f.write("\n\n")
    """
    print(f"loss: {loss.mean()}\n")
    print(f"MeanPriceDiff: {mean_price_diff.mean(axis=0)}\n")
    print(f"MeanCorrect: {mean_correct.mean()}\n")
    print(f"CorrectUp Rate: {correct_updown.mean()}\n")
    print(f"ForecastUp Rate: {forecast_updown.mean()}\n")

    print("success")


if __name__ == "__main__":
    main()
    """
    year_term = [2010, 2024]
    train_year = 2
    test_year = 1
    for i in range(year_term[0], year_term[1] - test_year):
        print(f"Train: {i}, Test: {i + train_year}")
        main(i, i + train_year)
    """
