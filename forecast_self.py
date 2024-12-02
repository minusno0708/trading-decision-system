import numpy as np
import pandas as pd
import torch

import random
import datetime

from data_provider.data_loader import DataLoader
from data_provider.pytorch_data_provider import PytorchDataProvider

from model.pytorch import Model
from evaluator import Evaluator
from logger import Logger

import matplotlib.pyplot as plt

import os

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

output_path = "output/images/self_forecast"
exp_name = "exp_1000_2017"

if not os.path.exists(f"{output_path}/{exp_name}"):
    os.makedirs(f"{output_path}/{exp_name}")

def main():
    input_length = 30
    output_length = 30
    train_start_year = 2017
    test_start_year = 2023
    is_pre_scaling = True

    logger = Logger(f"self_{exp_name}")
    logger.log("Start Self Forecasting, Seed: " + str(seed))
    logger.timestamp()


    data_loader = PytorchDataProvider(
        file_path=f"dataset/btc.csv",
        index_col="timeOpen",
        target_cols=["close"],
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        scaler_flag=is_pre_scaling,
    )

    model = Model(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        epochs=1000,
        num_parallel_samples=1000,
    )

    evaluator = Evaluator()

    train_loss = model.train(data_loader.train_dataset(batch_size=64, is_shuffle=True))

    logger.log("Train Loss")
    logger.log(train_loss)

    fig, ax = plt.subplots()

    ax.plot(train_loss)

    plt.savefig(f"{output_path}/{exp_name}/loss_{seed}.png")

    print("Forecast Train Data")
    logger.log("Forecast Train Data")

    loss_arr = np.array([])

    for i, (start_date, input_x, target_x, time_feature) in enumerate(data_loader.train_dataset(batch_size=1, is_shuffle=False)):
        forecasts, loss = model.make_evaluation_predictions(input_x, target_x)
        loss_arr = np.append(loss_arr, loss)

        if i % 100 == 0:
            #metrics = evaluator.evaluate(forecasts, target_x.numpy().reshape(-1))

            print(f"forecasting {i}th data, loss: {loss}")
            logger.log(f"forecasting {i}th data, loss: {loss}")

            input_x = input_x.detach().numpy().reshape(-1)
            target_x = target_x.detach().numpy().reshape(-1)
            true_mean = forecasts.distribution["mean"].reshape(-1)
            samples_mean = forecasts.values.mean
            median = forecasts.values.median
            quantile_10 = forecasts.values.quantile(0.1)
            quantile_90 = forecasts.values.quantile(0.9)

            fig, ax = plt.subplots()

            ax.plot(range(input_length), input_x, label="input")
            ax.plot(range(input_length, input_length + output_length), target_x, label="target")
            ax.plot(range(input_length, input_length + output_length), true_mean, label="true_mean")
            ax.plot(range(input_length, input_length + output_length), samples_mean, label="samples_mean")
            ax.plot(range(input_length, input_length + output_length), median, label="median")
            ax.fill_between(
                range(input_length, input_length + output_length),
                quantile_10,
                quantile_90,
                alpha=0.3
            )

            ax.set_xticks([])

            plt.ylim(data_loader.min("train") - 0.5, data_loader.max("train") + 0.5)
            plt.legend()
            
            plt.savefig(f"{output_path}/{exp_name}/train_{i}_{seed}.png")

    print("End Train Forecasting")
    logger.log("End Train Forecasting")

    print("Train Loss:", np.mean(loss_arr))
    logger.log("Train Loss: " + str(np.mean(loss_arr)))

    print("Forecast Test Data")
    logger.log("Forecast Test Data")

    loss_arr = np.array([])

    for i, (start_date, input_x, target_x, time_feature) in enumerate(data_loader.test_dataset(batch_size=1, is_shuffle=False)):
        forecasts, loss = model.make_evaluation_predictions(input_x, target_x)
        loss_arr = np.append(loss_arr, loss)

        if i % 100 == 0:
            print(f"forecasting {i}th data, loss: {loss}")
            logger.log(f"forecasting {i}th data, loss: {loss}")

            input_x = input_x.detach().numpy().reshape(-1)
            target_x = target_x.detach().numpy().reshape(-1)
            true_mean = forecasts.distribution["mean"].reshape(-1)
            samples_mean = forecasts.values.mean
            median = forecasts.values.median
            quantile_10 = forecasts.values.quantile(0.1)
            quantile_90 = forecasts.values.quantile(0.9)

            fig, ax = plt.subplots()

            ax.plot(range(input_length), input_x, label="input")
            ax.plot(range(input_length, input_length + output_length), target_x, label="target")
            ax.plot(range(input_length, input_length + output_length), true_mean, label="true_mean")
            ax.plot(range(input_length, input_length + output_length), samples_mean, label="samples_mean")
            ax.plot(range(input_length, input_length + output_length), median, label="median")
            ax.fill_between(
                range(input_length, input_length + output_length),
                quantile_10,
                quantile_90,
                alpha=0.3
            )

            ax.set_xticks([])

            plt.ylim(data_loader.min("test") - 0.5, data_loader.max("test") + 0.5)
            plt.legend()

            plt.savefig(f"{output_path}/{exp_name}/test_{i}_{seed}.png")

    print("End Test Forecasting")
    logger.log("End Test Forecasting")

    print("Test Loss:", np.mean(loss_arr))
    logger.log("Test Loss: " + str(np.mean(loss_arr)))

    logger.log("End Self Forecasting")


if __name__ == "__main__":
    for s in range(5):
        seed = s
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        main()