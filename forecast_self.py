import numpy as np
import pandas as pd
import torch

import random
import datetime

from data_provider.data_loader import DataLoader
from data_provider.pytorch_data_provider import PytorchDataProvider

from model.pytorch import Model

import matplotlib.pyplot as plt

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    print(seed)
    input_length = 30
    output_length = 30
    train_start_year = 2010
    test_start_year = 2023
    is_pre_scaling = True

    data_loader = PytorchDataProvider(
        file_path=f"dataset/btc.csv",
        index_col="timeOpen",
        target_cols=["close"],
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        scaler_flag=is_pre_scaling
    )

    model = Model(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        epochs=1000
    )

    model.train(data_loader.train_dataset(batch_size=32, is_shuffle=True))

    for i, (start_date, input_x, target_x, time_feature) in enumerate(data_loader.train_dataset(batch_size=1, is_shuffle=False)):
        if i % 100 == 0:
            print(f"forecasting {i}th data")
            mean, var = model.forecast(input_x)

            target_x = target_x.detach().numpy().reshape(-1)
            mean = mean.reshape(-1)

            fig, ax = plt.subplots()

            ax.plot(target_x, label="target")
            ax.plot(mean, label="forecast")

            plt.legend()

            plt.savefig(f"output/images/self_forecast/train1000_{i}_{seed}.png")


if __name__ == "__main__":
    for s in range(0, 5):
        seed = s
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        main()