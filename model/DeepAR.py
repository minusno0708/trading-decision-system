import pandas as pd

from model.DeepARTorch import Model as TorchModel
from model.DeepARMxnet import Model as MxModel

class Model:
    def __init__(self, input_length: int, output_length: int, model_type="torch"):
        self.input_length = input_length
        self.output_length = output_length

        if model_type == "torch":
            self.model = TorchModel(input_length, output_length)
        elif model_type == "mxnet":
            self.model = MxModel(input_length, output_length)

    def train(self, train_data: pd.DataFrame):
        self.model.train(train_data)

    def forecast(self, test_data: pd.DataFrame):
        return self.model.forecast(test_data)

    def evaluate(self, test_data: pd.DataFrame):
        return self.model.evaluate(test_data)

    def save(self, str_path: str):
        self.model.save(str_path)

    def load(self, str_path: str):
        self.model.load(str_path)

