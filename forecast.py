import pandas as pd
import matplotlib.pyplot as plt

from model.DeepAR import Model

from sklearn.preprocessing import StandardScaler

data_path = "dataset/btc.csv"

input_length = 30
output_length = 7

train_flag = False

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/images/{name}.png")

    plt.clf()
    plt.close()

def data_loader(file_path: str) -> list[pd.DataFrame, pd.DataFrame]:
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
    scaler = StandardScaler()
    df_row["close"] = scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

    # データを分割
    rate = 0.8
    n_train = int(len(df_row) * rate)

    train_data = df_row.iloc[:n_train + output_length]
    test_data = df_row.iloc[n_train:]

    return train_data, test_data    

if __name__ == "__main__":
    train_data, test_data = data_loader(data_path)

    draw_graph(list(train_data.index), list(train_data["close"]), "train_data")
    draw_graph(list(test_data.index), list(test_data["close"]), "test_data")

    model = Model(input_length, output_length)

    if train_flag:
        model.train(train_data)
        model.save("output/models/model.pth")
    else:
        model.load("output/models/model.pth")

    i = 0

    target_data = test_data.iloc[range(i, input_length + i), [0]]
    correct_data = test_data.iloc[range(input_length + i, input_length + output_length + i), [0]]

    forecasts = model.forecast(target_data)

    # 予測結果を描画
    plt.plot(target_data.index, target_data.values, label="target")
    plt.plot(correct_data.index, correct_data.values, label="correct")

    forecasts[0].plot()

    plt.legend()
    plt.savefig("output/images/forecast.png")

    print("success")
