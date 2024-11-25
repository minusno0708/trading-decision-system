import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader import DataLoader
from model import Model
from logger import Logger

import numpy as np
import pandas as pd
import torch
import mxnet as mx

import datetime
import math
import statistics

from gluonts.evaluation import Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor

import properscoring as ps

input_length = 30
output_length = 30

train_flag = True

model_type = "torch"
model_name = "deepar"

def main(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Logger("exp_follow")
    logger.log("electricity experiment with deepar")

    logger.timestamp()
    logger.log(f"Seed: {seed}")

    prediction_length = 100
    meta, train_ds, test_ds = get_dataset("electricity")
    
    model = Model(prediction_length=prediction_length, context_length=prediction_length, freq="h", model_type=model_type, model_name=model_name, epochs=100, num_parallel_samples=1000)
    model.train(train_ds)

    agg_metrics, item_metrics = model.backtest(
        dataset = train_ds,
    )

    logger.log("Test backtest")
    logger.log(agg_metrics)

    logger.log("Evaluation Done")
    logger.log("--------------------")

if __name__ == "__main__":
    for i in range(5,10):
        main(i)
