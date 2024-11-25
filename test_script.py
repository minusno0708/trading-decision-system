import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

from data_provider.data_loader import DataLoader
from data_provider.gluonts_data_provider import GluontsDataProvider

from model import Model

from logger import Logger
import os

input_length = 30
output_length = 30

data_loader = GluontsDataProvider(
    file_path=f"dataset/btc.csv",
    index_col="timeOpen",
    target_cols=["close"],
    prediction_length=input_length,
    context_length=output_length,
    freq="D",
    train_start_date=datetime.datetime(2010, 1, 1),
    test_start_date=datetime.datetime(2023, 1, 1),
    scaler_flag=False
)

model = Model(
    context_length=input_length,
    prediction_length=output_length,
    freq="D",
    epochs=10,
    num_parallel_samples=1000,
    model_name="deepar",
    model_type="torch"
)

model.train(data_loader.train_dataset())

test_dataset = data_loader.test_dataset()

forecast_it, ts_it = make_evaluation_predictions(
    test_dataset, predictor=model.model, num_samples=1000
)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(test_dataset))