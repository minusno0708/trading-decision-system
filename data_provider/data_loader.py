import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import os
import datetime

class DataLoader:
    def __init__(self,
        file_path: str,
        prediction_length: int,
        train_start_date: datetime.datetime = "2000-01-01",
        test_start_date: datetime.datetime = "2023-01-01",
        scaler_flag: bool = True,
    ):
        self.prediction_length = prediction_length

        self.scaler = StandardScaler()

        self.load(
            file_path=file_path,
            train_start_date=train_start_date,
            test_start_date=test_start_date,
            scaler_flag=scaler_flag
        )

    def load(self, 
        file_path: str,
        train_start_date: datetime.datetime,
        test_start_date: datetime.datetime,
        scaler_flag: bool,
    ):
        # データの読み込み
        df_row = pd.read_csv(file_path)

        # 不要な列を削除
        target_columns = ["timeOpen", "close"]
        df_row = df_row[target_columns]

        # 時刻をdatetime型に変換
        df_row["timeOpen"] = pd.to_datetime(df_row["timeOpen"], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # 日付順にソート
        df_row = df_row.sort_values("timeOpen")
        df_row = df_row.set_index("timeOpen")

        # データの範囲を選択
        df_row = df_row[df_row.index >= train_start_date]

        # 値を標準化
        if scaler_flag:
            df_row["close"] = self.scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

        # データを分割
        train_data = df_row[df_row.index < test_start_date]
        test_data = df_row[df_row.index >= test_start_date - datetime.timedelta(days=self.prediction_length)]

        self.train = train_data
        self.test = test_data 

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform([values])
