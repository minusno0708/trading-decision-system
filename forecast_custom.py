import argparse

import numpy as np
import pandas as pd
import torch

import random
import datetime

from data_provider.data_loader import DataLoader
from data_provider.custom_data_provider import CustomDataProvider

from model.custom import Model
from evaluator import Evaluator
from logger import Logger

import matplotlib.pyplot as plt

import os

def main(
    seed,
    output_path,
    exp_name,
    data_path,
    index_col,
    target_cols,
    train_start_year,
    test_start_year,
    prediction_length,
    context_length,
    epochs,
    num_batches,
    num_parallel_samples,
    is_pre_scaling,
):

    logger = Logger(exp_name, f"{output_path}/logs")
    logger.log("Start Self Forecasting, Seed: " + str(seed))
    logger.timestamp()


    data_loader = CustomDataProvider(
        file_path=data_path,
        index_col=index_col,
        target_cols=target_cols,
        prediction_length=prediction_length,
        context_length=context_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        scaler_flag=is_pre_scaling,
    )

    model = Model(
        context_length=context_length,
        prediction_length=prediction_length,
        freq="D",
        epochs=epochs,
        num_parallel_samples=num_parallel_samples,
    )

    evaluator = Evaluator()

    train_loss, val_loss = model.train(data_loader.train_dataset(batch_size=num_batches, is_shuffle=True), data_loader.test_dataset(batch_size=1, is_shuffle=False))

    logger.log("Train Loss")
    logger.log(train_loss)

    logger.log("Val Loss")
    logger.log(val_loss)

    fig, ax = plt.subplots()

    ax.plot(train_loss, label="train")
    ax.plot(val_loss, label="val")

    plt.legend()

    plt.savefig(f"{output_path}/images/{exp_name}/loss_{seed}.png")

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
            true_mean = forecasts[0].distribution["mean"].reshape(-1)
            samples_mean = forecasts[0].mean
            median = forecasts[0].median
            quantile_10 = forecasts[0].quantile(0.1)
            quantile_90 = forecasts[0].quantile(0.9)

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
            
            plt.savefig(f"{output_path}/images/{exp_name}/train_{i}_{seed}.png")

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
            true_mean = forecasts[0].distribution["mean"].reshape(-1)
            samples_mean = forecasts[0].mean
            median = forecasts[0].median
            quantile_10 = forecasts[0].quantile(0.1)
            quantile_90 = forecasts[0].quantile(0.9)

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

            plt.savefig(f"{output_path}/images/{exp_name}/test_{i}_{seed}.png")

    print("End Test Forecasting")
    logger.log("End Test Forecasting")

    print("Test Loss:", np.mean(loss_arr))
    logger.log("Test Loss: " + str(np.mean(loss_arr)))

    logger.log("End Self Forecasting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="output/forecast")
    parser.add_argument("--exp_name", type=str, default="exp_default")

    parser.add_argument("--data_path", type=str, default="dataset/btc.csv")
    parser.add_argument("--index_col", type=str, default="timeOpen")
    parser.add_argument("--target_cols", type=str, default=["close"])

    parser.add_argument("--train_start_year", type=int, default=2000)
    parser.add_argument("--test_start_year", type=int, default=2023)

    parser.add_argument("--prediction_length", type=int, default=30)
    parser.add_argument("--context_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--num_parallel_samples", type=int, default=1000)

    parser.add_argument("--is_pre_scaling", type=bool, default=True)

    args = parser.parse_args()

    seed = args.seed

    if not os.path.exists(f"{args.output_path}/images/{args.exp_name}"):
        os.makedirs(f"{args.output_path}/images/{args.exp_name}")

    if not os.path.exists(f"{args.output_path}/logs"):
        os.makedirs(f"{args.output_path}/logs")

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
        train_start_year=args.train_start_year,
        test_start_year=args.test_start_year,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        epochs=args.epochs,
        num_batches=args.num_batches,
        num_parallel_samples=args.num_parallel_samples,
        is_pre_scaling=args.is_pre_scaling,
    )