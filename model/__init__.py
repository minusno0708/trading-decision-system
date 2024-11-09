import pandas as pd


from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor

from gluonts.evaluation import Evaluator

from pathlib import Path
import os

from model.deepar import model as deepar_model
from model.deepvar import model as deepvar_model
from model.transformer import model as transformer_model
from model.itransformer import model as itransformer_model

class Model:
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            freq: str = "D",
            epochs: int = 100,
            num_parallel_samples: int = 1000,
            model_name="deepar",
            model_type="torch"
        ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq

        if model_name == "deepar":
            self.model = deepar_model(
                context_length=context_length,
                prediction_length=prediction_length,
                freq=freq,
                epochs=epochs,
                num_parallel_samples=num_parallel_samples,
                model_type=model_type
            )
        elif model_name == "deepvar":
            self.model = deepvar_model(
                context_length=context_length,
                prediction_length=prediction_length,
                freq=freq,
                epochs=epochs,
                num_parallel_samples=num_parallel_samples
            )
        elif model_name == "transformer":
            self.model = transformer_model(
                context_length=context_length,
                prediction_length=prediction_length,
                freq=freq,
                epochs=epochs,
                num_parallel_samples=num_parallel_samples
            )
        elif model_name == "itransformer":
            self.model = itransformer_model(
                context_length=context_length,
                prediction_length=prediction_length,
                freq=freq,
                epochs=epochs,
                num_parallel_samples=num_parallel_samples,
                model_type=model_type
            )
        else:
            raise ValueError(f"{model_name} is invalid model type")

    def train(self, dataset: ListDataset):
        self.model = self.model.train(dataset)

    def forecast(self, dataset: ListDataset):
        return list(self.model.predict(dataset))

    # テストデータの末尾の部分を予測した結果の評価
    def evaluate(self, dataset: ListDataset):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=self.model,
            num_samples=1000
        )

        evaluator = Evaluator()
        agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset))
       
        return agg_metrics, item_metrics

    # テストデータ全体の評価
    def backtest(self, dataset: ListDataset):
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        agg_metrics, item_metrics = backtest_metrics(
            test_dataset=dataset,
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

