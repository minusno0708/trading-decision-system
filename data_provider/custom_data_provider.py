import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import datetime

from data_provider.data_loader import DataLoader as SelfDataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset, target_cols, extention_cols, context_length, prediction_length, batch_size):
        date = dataset.index.values

        self.data = dataset[target_cols].values
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
            pd_time = pd.Timestamp(d)
            time_feature.append([
                #pd_time.year / 2100, # year
                #(pd_time.month - 1) / 11, # month
                #(pd_time.day - 1) / 30, # day of month
                #pd_time.weekday() / 6, # day of week
                (pd_time.hour*60 + pd_time.minute) / (24*60) - 0.5, # hour
                pd_time.hour / 23 - 0.5, # hour
                pd_time.dayofweek / 6 - 0.5, # day of week
                (pd_time.day - 1) / 30 - 0.5, # day of month
                (pd_time.dayofyear - 1) / 364 - 0.5, # day of year
            ])

        return np.array(time_feature)

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length + 1

    def __getitem__(self, idx):
        start_date = self.date_str[idx+self.context_length]
        input_x = torch.tensor(np.array([self.data[idx:idx+self.context_length]])).float().squeeze(0)
        target_x = torch.tensor(np.array([self.data[idx+self.context_length:idx+self.context_length+self.prediction_length]])).float().squeeze(0)

        time_features = torch.tensor(self.time_features[idx:idx+self.context_length]).float()
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
    
    def val_dataset(self, batch_size=1, is_shuffle=False):
        torch_dataset = self.custom_dataset(self.val, batch_size)
        return DataLoader(torch_dataset, batch_size=batch_size, shuffle=is_shuffle)
