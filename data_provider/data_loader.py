import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import os
import datetime

current_dir = os.path.dirname(__file__)
file_dir = "../dataset/"

class DataLoader:
    def __init__(self, prediction_length: int):
        self.prediction_length = prediction_length

        self.scaler = StandardScaler()

    def load(self, file_name: str, scaler_flag = True) -> list[pd.DataFrame, pd.DataFrame]:
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
        if scaler_flag:
            df_row["close"] = self.scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

        # データを分割
        # 2023年1月1日から予測を開始するように設定
        split_date = datetime.datetime(2023, 1, 1)
        train_data = df_row[df_row.index < split_date]
        test_data = df_row[df_row.index >= split_date - datetime.timedelta(days=31)]

        return train_data, test_data 

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform([values])
