import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
file_dir = "../dataset/"

output_dir = "../output/raw_data/"

cryptos = {
    "BitCoin": "btc",
    "Ethereum": "eth",
    "Tether": "tet",
    "Binance Coin": "bnb",
    "Solana": "sol",
    "USD Coin": "usdc",
    "XRP": "xrp",
    "TONCoin": "ton",
    "Dogecoin": "doge",
    "Cardano": "car"
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

    return df_row

def main():
    for crypto in cryptos:
        data = load(cryptos[crypto] + ".csv")

        plt.plot(data.index, data["close"])
        plt.xlabel("Date")
        plt.ylabel("Price(Yen)")
        plt.title(crypto)
        plt.savefig(current_dir + "/" + output_dir + crypto + ".png")

        plt.clf()
        plt.close()


if __name__ == "__main__":
    main()
