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

    is_training = True
    is_feat_metrics = True

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

    if is_training:
        train_loss, val_loss, minimal_val_loss = model.train(data_loader.train_dataset(batch_size=num_batches, is_shuffle=False), data_loader.val_dataset(batch_size=1, is_shuffle=False))

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
    else:
        model.load()

    print("Forecast Test Data")
    logger.log("Forecast Test Data")

    loss_arr = np.array([])

    evaluator = Evaluator(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])

    today_line_rmse_arr = np.array([])
    ave_line_rmse_arr = np.array([])

    today_line_mse_arr = np.array([])
    ave_line_mse_arr = np.array([])

    today_line_mae_arr = np.array([])
    ave_line_mae_arr = np.array([])

    if is_feat_metrics:
        feat_evaluator = []
        feat_today_line_rmse_arr = []
        feat_ave_line_rmse_arr = []
        for c in range(len(target_cols)):
            feat_evaluator.append(Evaluator(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]))

            feat_today_line_rmse_arr.append(np.array([]))
            feat_ave_line_rmse_arr.append(np.array([]))

    for i, (start_date, input_x, target_x, time_features, extention_features) in enumerate(data_loader.test_dataset(batch_size=1, is_shuffle=False)):
        forecasts, loss = model.make_evaluation_predictions(input_x, target_x, time_features, extention_features)
        
        loss_arr = np.append(loss_arr, loss)

        if is_pre_scaling:
            input_x_inverse = []
            target_x_inverse = []

            for c, col in enumerate(target_cols):
                forecasts[c].inverse_transform(data_loader.scaler[col])
                input_x_inverse.append(data_loader.inverse_transform(input_x[:,:,0].detach().numpy().squeeze(0), target_cols[0]).squeeze(0))
                target_x_inverse.append(data_loader.inverse_transform(target_x[:,:,0].detach().numpy().squeeze(0), target_cols[0]).squeeze(0))

            input_x = np.array(input_x_inverse)
            target_x = np.array(target_x_inverse)
        else:
            input_x = input_x.squeeze(0).permute(1, 0).detach().numpy()
            target_x = target_x.squeeze(0).permute(1, 0).detach().numpy()

        for c in range(len(target_cols)):
            metrics = evaluator.evaluate(forecasts[c], target_x[c])

            today_line = np.array([input_x[c, -1]] * prediction_length)
            today_line_rmse = evaluator.rmse(today_line, target_x[c])
            today_line_rmse_arr = np.append(today_line_rmse_arr, today_line_rmse)

            today_line_mse = evaluator.mse(today_line, target_x[c])
            today_line_mse_arr = np.append(today_line_mse_arr, today_line_mse)

            today_line_mae = evaluator.mae(today_line, target_x[c])
            today_line_mae_arr = np.append(today_line_mae_arr, today_line_mae)

            ave_line = np.array([input_x[c].mean()] * prediction_length)
            ave_line_rmse = evaluator.rmse(ave_line, target_x[c])
            ave_line_rmse_arr = np.append(ave_line_rmse_arr, ave_line_rmse)

            ave_line_mse = evaluator.mse(ave_line, target_x[c])
            ave_line_mse_arr = np.append(ave_line_mse_arr, ave_line_mse)

            ave_line_mae = evaluator.mae(ave_line, target_x[c])
            ave_line_mae_arr = np.append(ave_line_mae_arr, ave_line_mae)

            if is_feat_metrics:
                feat_metrics = feat_evaluator[c].evaluate(forecasts[c], target_x[c])

                feat_today_line_rmse_arr[c] = np.append(feat_today_line_rmse_arr[c], today_line_rmse)
                feat_ave_line_rmse_arr[c] = np.append(feat_ave_line_rmse_arr[c], ave_line_rmse)

        if i % 1000 == 0:
            print(f"forecasting {i}th data, date: {start_date}, loss: {loss}")
            logger.log(f"forecasting {i}th data, date: {start_date}, loss: {loss}")

            print(evaluator.mean())

            target_x = np.append(input_x[:, -1:], target_x, axis=1)

            samples_mean = forecasts[0].mean
            median = forecasts[0].median
            quantile_10 = forecasts[0].quantile(0.1)
            quantile_30 = forecasts[0].quantile(0.3)
            quantile_70 = forecasts[0].quantile(0.7)
            quantile_90 = forecasts[0].quantile(0.9)

            samples_mean = np.append(input_x[0, -1:], samples_mean)
            median = np.append(input_x[0, -1:], median)
            quantile_10 = np.append(input_x[0, -1:], quantile_10)
            quantile_30 = np.append(input_x[0, -1:], quantile_30)
            quantile_70 = np.append(input_x[0, -1:], quantile_70)
            quantile_90 = np.append(input_x[0, -1:], quantile_90)

            prediction_range = range(context_length - 1, context_length + prediction_length)
        
            fig, ax = plt.subplots()

            ax.plot(range(context_length), input_x[0], label="input", color="red")
            ax.plot(prediction_range, target_x[0], label="target", color="blue")
            ax.plot(prediction_range, samples_mean, label="mean", color="green")
            ax.fill_between(
                prediction_range,
                quantile_10,
                quantile_90,
                alpha=0.3,
            )
            ax.fill_between(
                prediction_range,
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

    for c, col in enumerate(target_cols):
        print("精度評価[{0}]".format(col))
        print(feat_evaluator[c].mean())
        logger.log("精度評価[{0}]".format(col))
        logger.log(evaluator.mean())

        print("前日価格比較[{0}]".format(col))
        print("rmse:" + str(feat_today_line_rmse_arr[c].mean()))
        logger.log("前日価格比較[{0}]".format(col))
        logger.log("rmse:" + str(feat_today_line_rmse_arr[c].mean()))

        print("平均価格比較[{0}]".format(col))
        print("rmse:" + str(feat_ave_line_rmse_arr[c].mean()))
        logger.log("平均価格比較[{0}]".format(col))
        logger.log("rmse:" + str(feat_ave_line_rmse_arr[c].mean()))

    print("精度評価")
    print(evaluator.mean())
    logger.log("精度評価")
    logger.log(evaluator.mean())

    print("前日価格比較")
    print("rmse:" + str(today_line_rmse_arr.mean()))
    print("mse:" + str(today_line_mse_arr.mean()))
    print("mae:" + str(today_line_mae_arr.mean()))
    logger.log("前日価格比較")
    logger.log("rmse:" + str(today_line_rmse_arr.mean()))
    logger.log("mse:" + str(today_line_mse_arr.mean()))
    logger.log("mae:" + str(today_line_mae_arr.mean()))

    print("平均価格比較")
    print("rmse:" + str(ave_line_rmse_arr.mean()))
    print("mse:" + str(ave_line_mse_arr.mean()))
    print("mae:" + str(ave_line_mae_arr.mean()))
    logger.log("平均価格比較")
    logger.log("rmse:" + str(ave_line_rmse_arr.mean()))
    logger.log("mse:" + str(ave_line_mse_arr.mean()))
    logger.log("mae:" + str(ave_line_mae_arr.mean()))
    
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
    if len(v.split("T")) == 2:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
    else:
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
    parser.add_argument("--val_data_length", type=int, default=0)
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