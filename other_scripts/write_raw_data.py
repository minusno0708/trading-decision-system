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

    return df_row

def main():
    for crypto in cryptos:
        data = load(cryptos[crypto] + ".csv")

        start_year = data.index[0].year
        end_year = data.index[-1].year

        print(f"{crypto}: {data.index[0]} - {data.index[-1]}")

        year_interval = (end_year - start_year) // 10 + 1

        fig, ax = plt.subplots()

        ax.plot(data.index, data["close"])

        ax.xaxis.set_major_locator(mdates.YearLocator(year_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.xlabel("Year")
        plt.ylabel("Price(Yen)")
        plt.title(crypto)
        plt.savefig(current_dir + "/" + output_dir + crypto + ".png")


if __name__ == "__main__":
    main()
