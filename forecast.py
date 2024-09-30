import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader import DataLoader
from model.DeepAR import Model

import numpy as np
import torch

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

if __name__ == "__main__":
    crypto = "btc"

    data_loader = DataLoader(output_length)
    train_data, test_data = data_loader.load(f"{crypto}.csv")

    draw_graph(list(train_data.index), list(train_data["close"]), f"{crypto}-train-data")
    draw_graph(list(test_data.index), list(test_data["close"]), f"{crypto}-test-data")

    model = Model(input_length, output_length)

    if train_flag:
        model.train(train_data)
        model.save(f"output/models/{crypto}_model")
    else:
        model.load(f"output/models/{crypto}_model")

    i = 0

    target_data = test_data.iloc[range(i, input_length + i), [0]]
    correct_data = test_data.iloc[range(input_length + i, input_length + output_length + i), [0]]

    forecasts = model.forecast(target_data)

    # 予測結果を描画
    input_data = test_data.iloc[range(i, input_length + i + 1), [0]]

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

    print("success")
