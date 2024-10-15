# データ描画用のスクリプト

import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

current_dir = os.path.dirname(__file__)
file_dir = "../dataset/"

output_dir = "../output/raw_data/"

cryptos = {
    "BitCoin": "btc",
    "Ethereum": "eth",
    "Tether": "usdt",
    "Binance Coin": "bnb",
    "Solana": "sol",
    "USD Coin": "usdc",
    "XRP": "xrp",
    "TONCoin": "ton",
    "Dogecoin": "doge",
    "Cardano": "ada",
}


def load(file_name: str) -> list[pd.DataFrame, pd.DataFrame]:
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

    train = df_row[df_row.index < "2023-01-01"]
    test = df_row[df_row.index >= "2023-01-01"]
    #df_row = df_row[df_row.index >= "2023-01-01"]

    return train, test

def main():
    for crypto in cryptos:
        train_data, test_data = load(cryptos[crypto] + ".csv")

        start_year = train_data.index[0].year
        end_year = test_data.index[-1].year

        #print(f"{crypto}: {data.index[0]} - {data.index[-1]}")

        year_interval = (end_year - start_year) // 10 + 1

        fig, ax = plt.subplots()

        ax.plot(train_data.index, train_data["close"], label="train")
        ax.plot(test_data.index, test_data["close"], label="test")

        ax.xaxis.set_major_locator(mdates.YearLocator(year_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.legend()

        plt.xlabel("Year")
        plt.ylabel("Price(Dollar)")
        plt.title(crypto)
        plt.savefig(current_dir + "/" + output_dir + crypto + ".png")


if __name__ == "__main__":
    main()
