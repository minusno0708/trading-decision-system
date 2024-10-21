import pandas as pd

import torch

from gluonts.dataset.common import ListDataset
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.mx.trainer import Trainer

from pathlib import Path
import os

class Model:
    def __init__(self, input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length

        trainer = Trainer(
            epochs=50,
            learning_rate=0.001,
        )

        self.model = SimpleFeedForwardEstimator(
            prediction_length=output_length,
            context_length=input_length,
            trainer=trainer,
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
