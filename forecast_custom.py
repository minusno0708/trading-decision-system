import argparse

import numpy as np
import pandas as pd
import torch

import random
from datetime import datetime

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
    train_start_date,
    train_end_date,
    test_start_date,
    test_end_date,
    train_data_length,
    test_data_length,
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
        scaler_flag=is_pre_scaling,
    )

    if split_type == "datetime":
        data_loader.update_date(train_start_date, train_end_date, test_start_date, test_end_date)
    elif split_type == "index":
        data_loader.update_date_by_index(test_start_date, train_data_length, test_data_length)

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

    evaluator = Evaluator()

    train_loss, val_loss, minimal_val_loss = model.train(data_loader.train_dataset(batch_size=num_batches, is_shuffle=False), data_loader.test_dataset(batch_size=1, is_shuffle=False))

    logger.log("Train Loss")
    logger.log(train_loss)

    logger.log("Val Loss")
    logger.log(val_loss)

    logger.log("Minimal Val Loss")
    logger.log(f"Epoch: {minimal_val_loss['epoch']}, Loss: {minimal_val_loss['loss']}")

    # ロスの推移をプロット
    fig, ax = plt.subplots()

    ax.plot(train_loss, label="train")
    ax.plot(val_loss, label="val")

    plt.legend()

    plt.savefig(f"{output_path}/images/{exp_name}/loss_{seed}.png")
    plt.close(fig)

    print("Forecast Test Data")
    logger.log("Forecast Test Data")

    loss_arr = np.array([])

    for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(data_loader.test_dataset(batch_size=1, is_shuffle=False)):
        print(start_date)
        
        forecasts, loss = model.make_evaluation_predictions(input_x, target_x, time_features, extention_features)
        loss_arr = np.append(loss_arr, loss)

        if is_pre_scaling:
            forecasts[0].inverse_transform(data_loader.scaler[target_cols[0]])
            input_x = data_loader.inverse_transform(input_x.detach().numpy().reshape(-1), target_cols[0]).reshape(-1)
            target_x = data_loader.inverse_transform(target_x.detach().numpy().reshape(-1), target_cols[0]).reshape(-1)
        else:
            input_x = input_x.detach().numpy().reshape(-1)
            target_x = target_x.detach().numpy().reshape(-1)

        metrics = evaluator.evaluate(forecasts[0], target_x)
        print(metrics)

        baseline = np.array([input_x[-1]] * prediction_length)
        baseline_rmse = evaluator.rmse(baseline, target_x)
        print("Baseline RMSE:", baseline_rmse)

        if i % 1 == 0:
            print(f"forecasting {i}th data, loss: {loss}")
            logger.log(f"forecasting {i}th data, loss: {loss}")

            #true_mean = forecasts[0].distribution["mean"].reshape(-1)
            samples_mean = forecasts[0].mean
            median = forecasts[0].median
            quantile_10 = forecasts[0].quantile(0.1)
            quantile_30 = forecasts[0].quantile(0.3)
            quantile_70 = forecasts[0].quantile(0.7)
            quantile_90 = forecasts[0].quantile(0.9)

            fig, ax = plt.subplots()

            ax.plot(range(context_length), input_x, label="input")
            ax.plot(range(context_length, context_length + prediction_length), target_x, label="target")
            #ax.plot(range(context_length, context_length + prediction_length), true_mean, label="true_mean")
            ax.plot(range(context_length, context_length + prediction_length), samples_mean, label="mean")
            ax.plot(range(context_length, context_length + prediction_length), median, label="median")
            ax.fill_between(
                range(context_length, context_length + prediction_length),
                quantile_10,
                quantile_90,
                alpha=0.3,
            )
            ax.fill_between(
                range(context_length, context_length + prediction_length),
                quantile_30,
                quantile_70,
                alpha=0.5,
            )

            ax.set_xticks([])

            #plt.ylim(data_loader.min("test") - 0.5, data_loader.max("test") + 0.5)
            plt.legend()

            plt.savefig(f"{output_path}/images/{exp_name}/test_{i}_{seed}.png")
            plt.close(fig)

    print("End Test Forecasting")
    logger.log("End Test Forecasting")

    print("Test Loss:", np.mean(loss_arr))
    logger.log("Test Loss: " + str(np.mean(loss_arr)))

    logger.log("End Self Forecasting")

def int2bool(v):
    v = int(v)
    if v == 0:
        return False
    else:
        return True

def str2datetime(v):
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