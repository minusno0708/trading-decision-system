import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import os

current_dir = os.path.dirname(__file__)
file_dir = "../dataset/"

class DataLoader:
    def __init__(self, prediction_length: int):
        self.prediction_length = prediction_length

        self.scaler = StandardScaler()

    def load(self, file_name: str) -> list[pd.DataFrame, pd.DataFrame]:
        file_path = current_dir + "/" + file_dir + file_name

        df_row = pd.read_csv(file_path)

        # 不要な列を削除
        target_columns = ["timeOpen", "close"]
        df_row = df_row[target_columns]

        # 時刻をdatetime型に変換
        df_row["timeOpen"] = pd.to_datetime(df_row["timeOpen"], format="%Y-%m-%dT%H:%M:%S.%fZ")

        # 日付順にソート
        df_row = df_row.sort_values("timeOpen")
        df_row = df_row.set_index("timeOpen")

        # 値を標準化
        df_row["close"] = self.scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

        # データを分割
        rate = 0.8
        n_train = int(len(df_row) * rate)

        train_data = df_row.iloc[:n_train + self.prediction_length]
        test_data = df_row.iloc[n_train:]

        return train_data, test_data 

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform([values])
