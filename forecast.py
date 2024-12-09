import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import torch
import mxnet as mx

import random
import datetime
import math
import statistics

from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

import properscoring as ps

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider

from model.gluonts import Model

from logger import Logger
import os

is_training = True
is_pre_scaling = True

evaluation_mode = [
    #"backtest",
    #"crps",
    #"updown",
    #"diff_price",
    "plot_forecast"
]

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/images/{name}.png")

    plt.clf()
    plt.close()

def count_duplicates(path: str, index: int = 0):
    if os.path.exists(f"{path}_{index}.png"):
        return count_duplicates(path, index + 1)
    else:
        return index

def draw_predict_graph(
        forecasts: list,
        target_data_date: list,
        target_data: list,
        correct_data_date: list,
        correct_data: list,
        graph_name: str,
        graph_num: str,
        path = "output/images/gluonts_forecast",
        ylim = None
    ):

    if not os.path.exists(f"{path}/{graph_name}"):
        os.makedirs(f"{path}/{graph_name}")

    dup_num = count_duplicates(f"{path}/{graph_name}/{graph_num}")
    graph_num = f"{graph_num}_{dup_num}"

    interval = len(target_data_date) // 3 * 2

    fig, ax = plt.subplots()

    ax.plot(target_data_date, target_data, label="input")
    ax.plot(correct_data_date, correct_data, label="correct")
    

    forecasts[0].plot(intervals=(0.3, 0.8))
    ax.plot(correct_data_date, forecasts[0].median, label="forecast-median")
    ax.plot(correct_data_date, forecasts[0].mean, label="forecast-mean")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend()
    plt.xlabel("Date")
    plt.savefig(f"{path}/{graph_name}/{graph_num}.png")

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

def main(
        experiment_name: str = "forecast-deepar",
        model_name: str = "deepar",
        model_type: str = "torch",
        input_length: int = 30,
        output_length: int = 30,
        train_start_year: int = 2010,
        test_start_year: int = 2023,
        seed: int = 0
    ):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)

    crypto = "btc"
    logger = Logger(experiment_name)
    logger.timestamp()
    logger.log(f"Seed: {seed}")

    data_loader = GluontsDataProvider(
        file_path=f"dataset/{crypto}.csv",
        index_col="timeOpen",
        target_cols=["close"],
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        train_end_date=datetime.datetime(test_start_year-1, 12, 31),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        test_end_date=datetime.datetime(2024, 12, 31),
        scaler_flag=is_pre_scaling
    )

    # 評価用の変数
    crps_error = np.array([])
    mean_price_diff = np.empty((0, output_length))

    updown_dates = [0, 4, 9, 14, 19, 24, 29]

    mean_correct = {
        updown_dates[0]: np.array([]),
        updown_dates[1]: np.array([]),
        updown_dates[2]: np.array([]),
        updown_dates[3]: np.array([]),
        updown_dates[4]: np.array([]),
        updown_dates[5]: np.array([]),
        updown_dates[6]: np.array([]
        ),
    }
    forecast_updown = {
        updown_dates[0]: np.array([]),
        updown_dates[1]: np.array([]),
        updown_dates[2]: np.array([]),
        updown_dates[3]: np.array([]),
        updown_dates[4]: np.array([]),
        updown_dates[5]: np.array([]),
        updown_dates[6]: np.array([]
        ),
    }
    correct_updown = {
        updown_dates[0]: np.array([]),
        updown_dates[1]: np.array([]),
        updown_dates[2]: np.array([]),
        updown_dates[3]: np.array([]),
        updown_dates[4]: np.array([]),
        updown_dates[5]: np.array([]),
        updown_dates[6]: np.array([]
        ),
    }

    model = Model(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        epochs=100,
        num_parallel_samples=1000,
        model_name=model_name,
        model_type=model_type
    )

    # training
    if is_training:
        model.train(data_loader.train_dataset())
        model.save(f"output/models/{crypto}_model")
    else:
        model.load(f"output/models/{crypto}_model")

    # evaluation
    if "backtest" in evaluation_mode:
        logger.log("Backtest")
        """
        agg_metrics, item_metrics = model.backtest(data_loader.train_dataset())

        logger.log("Train")
        logger.log("--- agg metrics ---")
        logger.log(agg_metrics)
        print("item_metrics:", item_metrics)
        """

        agg_metrics, item_metrics = model.backtest(data_loader.test_dataset())

        backtest_metrics = {}
        for key in agg_metrics:
            backtest_metrics[key] = np.array([])
        """
        #logger.log("Test")
        logger.set_tag("backtest")
        logger.log(agg_metrics)
        print("item_metrics:", item_metrics)

        logger.log("\n")
        """
    

    for i in range(data_loader.data_length('train')):
        if "backtest" in evaluation_mode:
            agg_metrics, item_metrics = model.backtest(data_loader.evaluation_data(i))
            for key in agg_metrics:
                backtest_metrics[key] = np.append(backtest_metrics[key], agg_metrics[key])



        target_data, correct_data = data_loader.prediction_data(i, 'train')
        correct_data_values = data_loader.listdata_values(correct_data)

        forecasts = model.forecast(target_data)

        if i % 100 == 0:
            print(f"Forecasting {i} Days")
            if "plot_forecast" in evaluation_mode:
                draw_predict_graph(
                    forecasts=forecasts,
                    target_data_date=data_loader.listdata_dates(target_data),
                    target_data=data_loader.listdata_values(target_data),
                    correct_data_date=data_loader.listdata_dates(correct_data),
                    correct_data=data_loader.listdata_values(correct_data),
                    graph_name="train_" + experiment_name,
                    graph_num=i,
                    ylim=(data_loader.min("train") - 0.5, data_loader.max("train") + 0.5)
                )

        # CRPS 誤差の計算
        if "crps" in evaluation_mode:  
            for d in range(output_length):
                crps_error = np.append(crps_error, ps.crps_ensemble(correct_data_values[d], forecasts[0].samples[:, d]))

        # 実際の価格との差分の計算
        if "diff_price" in evaluation_mode:
            if is_pre_scaling:
                correct_inverse = data_loader.inverse_transform(correct_data_values)[0]
                mean_inverse = data_loader.inverse_transform(forecasts[0].mean)[0]
            else:
                correct_inverse = correct_data_values
                mean_inverse = forecasts[0].mean

            mean_price_diff = np.append(mean_price_diff, [rmse(correct_inverse, mean_inverse)], axis=0)

        # 正解と予測の上下の判定
        if "updown" in evaluation_mode:
            past_day_value = data_loader.listdata_values(target_data)[-1]

            for d in updown_dates:
                correct_flag = past_day_value < correct_data_values[d]
                correct_updown[d] = np.append(correct_updown[d], correct_flag)

                forecast_mean_flag = past_day_value < forecasts[0].mean[d]
                forecast_updown[d] = np.append(forecast_updown[d], forecast_mean_flag)

                mean_correct[d] = np.append(mean_correct[d], correct_flag == forecast_mean_flag)
    
    if "backtest" in evaluation_mode:
        logger.set_tag("backtest")
        for key in backtest_metrics:
            logger.log(f"{key}: {backtest_metrics[key].mean()}")
        logger.log("\n")

    if "crps" in evaluation_mode:
        logger.set_tag("crps")
        logger.log(crps_error.mean())
        logger.log("\n")

    if "diff_price" in evaluation_mode:
        logger.set_tag("mean_price_diff")
        logger.log(mean_price_diff.mean(axis=0))
        logger.log("\n")

    if "updown" in evaluation_mode:
        for d in updown_dates:
            logger.log(f"Day: {d}")

            logger.log(f"MeanCorrect: {mean_correct[d].mean()}")
            logger.log(f"CorrectUp Rate: {correct_updown[d].mean()}")
            logger.log(f"ForecastUp Rate: {forecast_updown[d].mean()}")
            logger.log("\n")

    logger.log("Evaluation Done")
    logger.log("--------------------\n")

    for i in range(data_loader.data_length('test')):
        if "backtest" in evaluation_mode:
            agg_metrics, item_metrics = model.backtest(data_loader.evaluation_data(i))
            for key in agg_metrics:
                backtest_metrics[key] = np.append(backtest_metrics[key], agg_metrics[key])



        target_data, correct_data = data_loader.prediction_data(i, 'test')
        correct_data_values = data_loader.listdata_values(correct_data)

        forecasts = model.forecast(target_data)

        if i % 100 == 0:
            print(f"Forecasting {i} Days")
            if "plot_forecast" in evaluation_mode:
                draw_predict_graph(
                    forecasts=forecasts,
                    target_data_date=data_loader.listdata_dates(target_data),
                    target_data=data_loader.listdata_values(target_data),
                    correct_data_date=data_loader.listdata_dates(correct_data),
                    correct_data=data_loader.listdata_values(correct_data),
                    graph_name="test_" + experiment_name,
                    graph_num=i,
                    ylim=(data_loader.min("test") - 0.5, data_loader.max("test") + 0.5)
                )


if __name__ == "__main__":
    exp = ["deepar", "torch"]
    
    for s in range(5):

        main(
            experiment_name=f"exp1210_{s}",
            model_name=exp[0],
            model_type=exp[1],
            input_length=30,
            output_length=30,
            train_start_year=2019,
            test_start_year=2023,
            seed=s
        )

