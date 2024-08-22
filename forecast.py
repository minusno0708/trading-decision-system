import pandas as pd

from sklearn.preprocessing import StandardScaler

data_path = "dataset/btc.csv"

def data_loader(file_path: str) -> [pd.DataFrame, pd.DataFrame]:
    df_row = pd.read_csv(file_path)

    # 不要な列を削除
    target_columns = ["timeOpen", "close"]
    df_row = df_row[target_columns]

    # 時刻をdatetime型に変換
    df_row["timeOpen"] = pd.to_datetime(df_row["timeOpen"], format="%Y-%m-%dT%H:%M:%S.%fZ")

    # 日付順にソート
    df_row = df_row.sort_values("timeOpen")
    df_row = df_row.set_index("timeOpen")

    # 値を正規化
    scaler = StandardScaler()
    df_row["close"] = scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

    # データを分割
    rate = 0.8
    n_train = int(len(df_row) * rate)
    n_test = len(df_row) - n_train

    train_data = df_row.iloc[:n_train]
    test_data = df_row.iloc[n_train:]

    return train_data, test_data

if __name__ == "__main__":
    train, test = data_loader(data_path)
    print(train.head())
    print(test.head())