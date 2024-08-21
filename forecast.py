import pandas as pd

data_path = "dataset/btc.csv"

def data_loader(file_path: str) -> pd.DataFrame:
    df_row = pd.read_csv(file_path)

    # 不要な列を削除
    target_columns = ["timeOpen", "close"]
    df_row = df_row[target_columns]

    # 時刻をdatetime型に変換
    df_row["timeOpen"] = pd.to_datetime(df_row["timeOpen"], format="%Y-%m-%dT%H:%M:%S.%fZ")

    # 日付順にソート
    df_row = df_row.sort_values("timeOpen")

    return df_row

if __name__ == "__main__":
    data = data_loader(data_path)
    print(data.head())