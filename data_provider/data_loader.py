import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
import datetime

class DataLoader:
    def __init__(self,
        file_path: str,
        index_col: str,
        target_cols: list,
        prediction_length: int,
        context_length: int,
        freq: str,
        scaler_flag: bool,
    ):
        self.file_path = file_path
        self.index_col = index_col
        self.target_cols = target_cols
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.scaler_flag = scaler_flag
        self.nums_moving_average = [5, 25, 75]

        self.scaler = {}
        for col in self.target_cols:
            self.scaler[col] = StandardScaler()

        self.extention_cols = []

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
        elif self.file_path == "dataset/weather.csv":
            df_row[self.index_col] = pd.to_datetime(df_row[self.index_col], format="%Y-%m-%d %H:%M:%S")
        else:
            df_row[self.index_col] = pd.to_datetime(df_row[self.index_col], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # 日付順にソート
        df_row = df_row.sort_values(self.index_col)
        df_row = df_row.set_index(self.index_col)

        # 移動平均を追加
        for num in self.nums_moving_average:
            for col in self.target_cols:
                extention_name = f"{col}_mov_line_{num}"
                df_row[extention_name] = df_row[col].rolling(window=num).mean()
                self.extention_cols.append(extention_name)

        self.df_row = df_row
        self.date = df_row.index

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

    def split_dataset(self):
        # データを分割
        train_data = self.df_row[(self.df_row.index >= self.train_start_date) & (self.df_row.index <= self.train_end_date)].copy()
        test_data = self.df_row[(self.df_row.index >= self.test_start_date) & (self.df_row.index <= self.test_end_date)].copy()

        # 値を標準化
        if self.scaler_flag:
            for col in self.target_cols:
                self.scaler[col].fit(train_data[col].values.reshape(-1, 1))
                train_data[col] = self.scaler[col].transform(train_data[col].values.reshape(-1, 1))
                test_data[col] = self.scaler[col].transform(test_data[col].values.reshape(-1, 1))

                for extention_name in self.extention_cols:
                    train_data[extention_name] = self.scaler[col].transform(train_data[extention_name].values.reshape(-1, 1))
                    test_data[extention_name] = self.scaler[col].transform(test_data[extention_name].values.reshape(-1, 1))

        self.train = train_data
        self.test = test_data 

        self.print_datainfo("train")
        self.print_datainfo("test")

    def split_dataset_val(self, val_num):
        # データを分割
        train_data = self.df_row[(self.df_row.index >= self.train_start_date) & (self.df_row.index <= self.train_end_date)].copy()
        test_data = self.df_row[(self.df_row.index >= self.test_start_date) & (self.df_row.index <= self.test_end_date)].copy()
        val_data = self.df_row[(self.df_row.index >= self.val_start_date) & (self.df_row.index <= self.val_end_date)].copy()

        # 値を標準化
        if self.scaler_flag:
            for col in self.target_cols:
                self.scaler[col].fit(train_data[col].values.reshape(-1, 1))
                train_data[col] = self.scaler[col].transform(train_data[col].values.reshape(-1, 1))
                test_data[col] = self.scaler[col].transform(test_data[col].values.reshape(-1, 1))
                val_data[col] = self.scaler[col].transform(val_data[col].values.reshape(-1, 1))

                for extention_name in self.extention_cols:
                    train_data[extention_name] = self.scaler[col].transform(train_data[extention_name].values.reshape(-1, 1))
                    test_data[extention_name] = self.scaler[col].transform(test_data[extention_name].values.reshape(-1, 1))
                    val_data[extention_name] = self.scaler[col].transform(val_data[extention_name].values.reshape(-1, 1))

        self.train = train_data
        self.val = val_data
        self.test = test_data 

        self.print_datainfo("train")
        self.print_datainfo("val")
        self.print_datainfo("test")

    def update_date(self, train_start_date, train_end_date, test_start_date, test_end_date):
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        self.split_dataset()

    def update_date_by_index(self, test_start_date, train_num, test_num, val_num=0):
        target_index = self.search_date_index(test_start_date)

        self.test_start_date = self.date[target_index - self.context_length]
        self.test_end_date = self.date[target_index + test_num + self.prediction_length - 2]
        self.val_end_date = self.date[target_index - 1]
        self.val_start_date = self.date[target_index - val_num - self.context_length - self.prediction_length + 1]
        if val_num > 0:
            self.train_end_date = self.date[target_index - val_num - self.prediction_length]
            self.train_start_date = self.date[target_index - train_num - val_num - self.context_length - self.prediction_length*2 + 2]
        else:
            self.train_end_date = self.date[target_index - 1]
            self.train_start_date = self.date[target_index - train_num - self.context_length - self.prediction_length + 1]

        if val_num > 0:
            self.split_dataset_val(val_num)
        else:
            self.split_dataset()

    def search_date_index(self, date):
        return self.df_row.index.get_loc(date)

    def is_date_in_dataset(self, date):
        return date in self.df_row.index

    def print_datainfo(self, label="train"):
        if label == "train":
            print(f"train date, start: {self.train_start_date}, end: {self.train_end_date}")
            print("train date num: ", len(self.train)-self.context_length-self.prediction_length + 1)
        elif label == "test":
            print(f"test date, start: {self.test_start_date}, end: {self.test_end_date}")
            print("test date num: ", len(self.test)-self.context_length-self.prediction_length + 1)
        elif label == "val":
            print(f"val date, start: {self.val_start_date}, end: {self.val_end_date}")
            print("val date num: ", len(self.val)-self.context_length-self.prediction_length + 1)
        
