import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import os
import datetime

class DataLoader:
    def __init__(self,
        file_path: str,
        index_col: str,
        target_cols: list,
        prediction_length: int,
        context_length: int,
        freq: str = "D",
        train_start_date: datetime.datetime = "2000-01-01",
        test_start_date: datetime.datetime = "2023-01-01",
        scaler_flag: bool = True,
    ):
        self.file_path = file_path
        self.index_col = index_col
        self.target_cols = target_cols
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.train_start_date = train_start_date
        self.test_start_date = test_start_date
        self.scaler_flag = scaler_flag

        self.scaler = {}
        for col in self.target_cols:
            self.scaler[col] = StandardScaler()

        self.load()

    def load(self):
        # データの読み込み
        df_row = pd.read_csv(self.file_path)

        # 不要な列を削除
        target_columns = [self.index_col] + self.target_cols
        df_row = df_row[target_columns]

        # 時刻をdatetime型に変換
        if self.file_path == "dataset/usd_jpy.csv":
            df_row[self.index_col] = pd.to_datetime(df_row[self.index_col], format="%m/%d/%Y")
        else:
            df_row[self.index_col] = pd.to_datetime(df_row[self.index_col], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # 日付順にソート
        df_row = df_row.sort_values(self.index_col)
        df_row = df_row.set_index(self.index_col)

        # データの範囲を選択
        df_row = df_row[df_row.index >= self.train_start_date]

        # 値を標準化
        if self.scaler_flag:
            for col in self.target_cols:
                self.scaler[col].fit(df_row[df_row.index < self.test_start_date][col].values.reshape(-1, 1))
                df_row[col] = self.scaler[col].transform(df_row[col].values.reshape(-1, 1))

        # データを分割
        train_data = df_row[df_row.index < self.test_start_date]
        test_data = df_row[df_row.index >= self.test_start_date - datetime.timedelta(days=self.prediction_length)]

        self.train = train_data
        self.test = test_data 

    def inverse_transform(self, values: np.ndarray, col: str = "close") -> np.ndarray:
        return self.scaler[col].inverse_transform([values])

    def max(self, label: str = "all"):
        if label == "train":
            return self.train.max().values[0]
        elif label == "test":
            return self.test.max().values[0]
        else:
            train_max = self.train.max().values[0]
            test_max = self.test.max().values[0]
            return train_max if train_max > test_max else test_max

    def min(self, label: str = "all"):
        if label == "train":
            return self.train.min().values[0]
        elif label == "test":
            return self.test.min().values[0]
        else:
            train_min = self.train.min().values[0]
            test_min = self.test.min().values[0]
            return train_min if train_min < test_min else test_min
