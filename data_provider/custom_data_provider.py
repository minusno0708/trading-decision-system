import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import datetime

from data_provider.data_loader import DataLoader as SelfDataLoader

class CustomDataset(Dataset):
    def __init__(self, data, time, context_length, prediction_length, batch_size):
        self.data = data
        self.time_str = time.astype(str)
        self.time_feature = self.convert_time(time)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size


    def convert_time(self, time):
        dt = time.astype("M8[ms]").astype(datetime.datetime)

        time_feature = []

        for d in dt:
            time_feature.append([
                d.year / 2100,
                (d.month - 1) / 11,
                (d.day - 1) / 30,
            ])

        return time_feature

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length

    def __getitem__(self, idx):
        start_date = self.time_str[idx+self.context_length]
        input_x = torch.tensor(self.data[idx:idx+self.context_length]).reshape(-1, self.context_length).float()
        target_x = torch.tensor(self.data[idx+self.context_length:idx+self.context_length+self.prediction_length]).reshape(-1, self.prediction_length).float()

        time_feature = torch.tensor(self.time_feature[idx:idx+self.context_length+self.prediction_length]).float()

        return start_date, input_x, target_x, time_feature

class CustomDataProvider(SelfDataLoader):
    def train_dataset(self, batch_size, is_shuffle=True):
        torch_dataset = CustomDataset(self.train["close"].values, self.train.index.values, self.context_length, self.prediction_length, batch_size)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=is_shuffle)

    def test_dataset(self, batch_size=1, is_shuffle=False):
        torch_dataset = CustomDataset(self.test["close"].values, self.test.index.values, self.context_length, self.prediction_length, batch_size)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=is_shuffle)
