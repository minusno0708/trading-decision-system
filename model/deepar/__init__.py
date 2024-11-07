import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor

from gluonts.evaluation import Evaluator

from pathlib import Path
import os

from model.deepar.torch import model as torch_model
from model.deepar.mxnet import model as mx_model

class Model:
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            epochs: int = 100,
            num_parallel_samples: int = 1000,
            model_type="torch"
        ):
        self.context_length = context_length
        self.prediction_length = prediction_length

        if model_type == "torch":
            self.model = torch_model(context_length, prediction_length, epochs, num_parallel_samples)
        elif model_type == "mxnet":
            self.model = mx_model(context_length, prediction_length, epochs, num_parallel_samples)

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

    # テストデータの末尾の部分を予測した結果の評価
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

    # テストデータ全体の評価
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

