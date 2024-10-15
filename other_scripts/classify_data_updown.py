# 時系列データを読み込み、上昇・下降の割合を計算するスクリプト

import os
import datetime

import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)
file_dir = "../dataset/"

def load(file_name: str) -> pd.DataFrame:
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

    return df_row

def choose_year(data: pd.DataFrame, target_year: int) -> pd.DataFrame:
    return data[data.index.year == target_year]

def main():
    data = load("btc.csv")
    first_year = data.index[0].year
    last_year = data.index[-1].year

    for year in range(first_year, last_year + 1):
        data_year = choose_year(data, year)

        updown = {
            "up": 0,
            "up0-1": 0,
            "up1-10": 0,
            "up10-100": 0,
            "up100-1000": 0,
            "up1000-": 0,
            "down": 0,
            "down0-1": 0,
            "down1-10": 0,
            "down10-100": 0,
            "down100-1000": 0,
            "down1000-": 0,
        }
        ave_up = np.array([])
        ave_down = np.array([])

        max_up = 0
        max_down = 0

        for i in range(len(data_year) - 1):
            current = data_year.iloc[i]["close"]
            next = data_year.iloc[i + 1]["close"]

            diff = next - current

            if diff > max_up:
                max_up = diff
            if diff < max_down:
                max_down = diff

            if diff > 0:
                ave_up = np.append(ave_up, diff)
                updown["up"] += 1
                if diff < 1:
                    updown["up0-1"] += 1
                elif diff < 10:
                    updown["up1-10"] += 1
                elif diff < 100:
                    updown["up10-100"] += 1
                elif diff < 1000:
                    updown["up100-1000"] += 1
                else:
                    updown["up1000-"] += 1
            else:
                ave_down = np.append(ave_down, diff)
                updown["down"] += 1
                if diff > -1:
                    updown["down0-1"] += 1
                elif diff > -10:
                    updown["down1-10"] += 1
                elif diff > -100:
                    updown["down10-100"] += 1
                elif diff > -1000:
                    updown["down100-1000"] += 1
                else:
                    updown["down1000-"] += 1
        print(f"{year}: {len(data_year)}")
        up_rate = updown["up"] / (updown["up"] + updown["down"])
        down_rate = updown["down"] / (updown["up"] + updown["down"])
        print(f"up: {up_rate}")
        print(f"down: {down_rate}")
        

if __name__ == "__main__":
    main()
