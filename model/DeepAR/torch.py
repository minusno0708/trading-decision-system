import pandas as pd

import torch
import torch.nn as nn

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator as TorchDeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor

from gluonts.evaluation import Evaluator

from pathlib import Path
import os

torch.set_float32_matmul_precision("high")

class Model:
    def __init__(self, input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length

        self.model = TorchDeepAREstimator(
            prediction_length=output_length,
            context_length=input_length,
            freq="D",
            trainer_kwargs={"max_epochs": 100},
            num_layers = 2,
            hidden_size = 40,
            lr = 0.001,
            weight_decay = 1e-08,
            dropout_rate = 0.1,
            patience = 10,
            num_feat_dynamic_real = 0,
            num_feat_static_cat = 0,
            num_feat_static_real = 0,
            scaling = True,
            num_parallel_samples = 1000,
            batch_size = 32,
            num_batches_per_epoch = 50,
        )

    def train(self, train_data: pd.DataFrame):
        dataset = ListDataset(
            [{"start": train_data.index[0], "target": train_data["close"]}],
            freq="1D",
        )

        self.model = self.model.train(dataset)

    def forecast(self, test_data: pd.DataFrame):
        dataset = ListDataset(
            [{"start": test_data.index[0], "target": test_data["close"]}],
            freq="1D",
        )

        forecasts = list(self.model.predict(dataset))

        return forecasts

    def evaluate(self, test_data: pd.DataFrame):
        dataset = ListDataset(
            [{"start": test_data.index[0], "target": test_data["close"]}],
            freq="1D",
        )

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.model,
            num_samples=1000
        )

        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset))
        return agg_metrics, item_metrics

    def backtest(self, test_data: pd.DataFrame):
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        test_dataset = ListDataset(
            [{"start": test_data.index[0], "target": test_data["close"]}],
            freq="1D",
        )

        agg_metrics, item_metrics = backtest_metrics(
            test_dataset=test_dataset,
            predictor=self.model,
            evaluator=evaluator,
        )

        return agg_metrics, item_metrics

    def save(self, str_path: str):
        path = Path(str_path)

        if not os.path.exists(path):
            os.makedirs(path)

        self.model.serialize(path)

    def load(self, str_path: str):
        path = Path(str_path)
        self.model = Predictor.deserialize(path)