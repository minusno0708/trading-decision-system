import pandas as pd

import torch

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

from pathlib import Path
import os

torch.set_float32_matmul_precision("high")

class Model:
    def __init__(self, input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length

        self.model = DeepAREstimator(
            prediction_length=output_length,
            context_length=input_length,
            freq="D",
            trainer_kwargs={"max_epochs": 10},
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
            num_parallel_samples = 100,
            batch_size = 32,
            num_batches_per_epoch = 50,
        )

    def train(self, train_data: pd.DataFrame):
        dataset = ListDataset(
            [{"start": train_data.index[0], "target": train_data["close"]}],
            freq="D",
        )

        self.model = self.model.train(dataset)

    def forecast(self, test_data: pd.DataFrame):
        dataset = ListDataset(
            [{"start": test_data.index[0], "target": test_data["close"]}],
            freq="D",
        )

        forecasts = list(self.model.predict(dataset))

        return forecasts

    def save(self, str_path: str):
        path = Path(str_path)

        if not os.path.exists(path):
            os.makedirs(path)

        self.model.serialize(path)

    def load(self, str_path: str):
        path = Path(str_path)
        self.model = Predictor.deserialize(path)
