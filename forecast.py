import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader import DataLoader
from model.DeepAR import Model

import numpy as np
import pandas as pd
import torch

import datetime

input_length = 30
output_length = 7

train_flag = True

seed = 0

np.random.seed(seed)
torch.manual_seed(seed)

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/images/{name}.png")

    plt.clf()
    plt.close()

def draw_predict_graph(input_data: pd.DataFrame, forecasts: list, correct_data: pd.DataFrame, crypto: str):
    fig, ax = plt.subplots()

    ax.plot(input_data.index, input_data.values, label="input")
    ax.plot(correct_data.index, correct_data.values, label="correct")
    

    forecasts[0].plot()
    ax.plot(forecasts[0].index, forecasts[0].median, label="forecast-median")
    ax.plot(forecasts[0].index, forecasts[0].mean, label="forecast-mean")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.legend()
    plt.xlabel("Date")
    plt.savefig(f"output/images/{crypto}-forecast.png")

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

if __name__ == "__main__":
    crypto = "btc"

    data_loader = DataLoader(output_length)
    train_data, test_data = data_loader.load(f"{crypto}.csv", True, datetime.datetime(2021, 1, 1), datetime.datetime(2023, 1, 1))

    mean_loss = np.array([])
    median_loss = np.array([])

    mean_correct = np.array([])
    median_correct = np.array([])

    updown_trend = np.array([0, 0])

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

        correct_inverse = data_loader.inverse_transform(correct_data.values.flatten())[0][0]
        mean_inverse = data_loader.inverse_transform(forecasts[0].mean)[0][0]
        median_inverse = data_loader.inverse_transform(forecasts[0].median)[0][0]

        mean_loss = np.append(mean_loss, rmse(correct_inverse, mean_inverse))
        median_loss = np.append(median_loss, rmse(correct_inverse, median_inverse))
        if correct_inverse < mean_inverse:
            updown_trend[0] += 1
        else:
            updown_trend[1] += 1

        correct_flag = target_data.values[-1] < correct_data.values.flatten()[0]
        mean_flag = target_data.values[-1] < forecasts[0].mean[0]
        median_flag = target_data.values[-1] < forecasts[0].median[0]

        if correct_flag == mean_flag:
            mean_correct = np.append(mean_correct, 1)
        else:
            mean_correct = np.append(mean_correct, 0)

        if correct_flag == median_flag:
            median_correct = np.append(median_correct, 1)
        else:
            median_correct = np.append(median_correct, 0)
        
    print(f"UpDown: {updown_trend}")
    print(f"MeanLoss: {mean_loss.mean()}")
    print(f"MeanCorrect: {mean_correct.mean()}")
    print(f"MedianLoss: {median_loss.mean()}")
    print(f"MedianCorrect: {median_correct.mean()}")

    print("success")
