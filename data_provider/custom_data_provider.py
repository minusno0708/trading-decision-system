import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import datetime

from data_provider.data_loader import DataLoader as SelfDataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset, target_cols, extention_cols, context_length, prediction_length, batch_size):
        date = dataset.index.values

        self.data = dataset[target_cols[0]].values
        self.date_str = date.astype(str)

        self.time_features = self.time_to_feature(date)
        self.extention_features = dataset[extention_cols].values

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size

    def time_to_feature(self, date):
        dt = date.astype("M8[ms]").astype(datetime.datetime)

        time_feature = []

        for d in dt:
            time_feature.append([
                d.year / 2100, # year
                (d.month - 1) / 11, # month
                (d.day - 1) / 30, # day of month
                d.weekday() / 6, # day of week
            ])

        return np.array(time_feature)

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length + 1

    def __getitem__(self, idx):
        start_date = self.date_str[idx+self.context_length]
        input_x = torch.tensor(np.array([self.data[idx:idx+self.context_length]])).float()
        target_x = torch.tensor(np.array([self.data[idx+self.context_length:idx+self.context_length+self.prediction_length]])).float()

        time_features = torch.tensor(self.time_features[idx+self.context_length:idx+self.context_length+self.prediction_length]).float()
        extention_features = torch.tensor(self.extention_features[idx:idx+self.context_length]).float()

        return start_date, input_x, target_x, time_features, extention_features

class CustomDataProvider(SelfDataLoader):
    def custom_dataset(self, dataset, batch_size):
        return CustomDataset(dataset, self.target_cols, self.extention_cols, self.context_length, self.prediction_length, batch_size)

    def train_dataset(self, batch_size, is_shuffle=True):
        torch_dataset = self.custom_dataset(self.train, batch_size)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=is_shuffle)

    def test_dataset(self, batch_size=1, is_shuffle=False):
        torch_dataset = self.custom_dataset(self.test, batch_size)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=is_shuffle)
