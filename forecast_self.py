import numpy as np
import pandas as pd
import torch

import random
import datetime

from data_provider.data_loader import DataLoader
from data_provider.pytorch_data_provider import PytorchDataProvider

from model.pytorch import Model

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    input_length = 30
    output_length = 30
    train_start_year = 2010
    test_start_year = 2023
    is_pre_scaling = True

    data_loader = PytorchDataProvider(
        file_path=f"dataset/btc.csv",
        index_col="timeOpen",
        target_cols=["close"],
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        train_start_date=datetime.datetime(train_start_year, 1, 1),
        test_start_date=datetime.datetime(test_start_year, 1, 1),
        scaler_flag=is_pre_scaling
    )

    model = Model(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        epochs=100,
        num_parallel_samples=100
    )

    print('success')

if __name__ == "__main__":
    main()