import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator

data_path = "dataset/btc.csv"

prediction_length = 36

def do_deepar(data: pd.DataFrame):
    data = ListDataset(
        [
            {
                "start": data.index[0],
                "target": data.close[:data.index[0]]
            }
        ]
    )

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

    # 値を標準化
    scaler = StandardScaler()
    df_row["close"] = scaler.fit_transform(df_row["close"].values.reshape(-1, 1))

    # データを分割
    rate = 0.8
    n_train = int(len(df_row) * rate)

    train_data = df_row.iloc[:n_train + prediction_length]
    test_data = df_row.iloc[n_train:]

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = data_loader(data_path)

    model = DeepAREstimator(
        prediction_length=prediction_length,
        freq="D",
        trainer_kwargs={"max_epochs": 5}
    ).train(PandasDataset(train_data, target="close"))
    
    forecasts = list(model.predict(PandasDataset(test_data, target="close")))

    plt.plot(test_data["close"], color="black")
    for forecast in forecasts:
        forecast.plot()
    plt.legend(["True values"], loc="upper left", fontsize="xx-large")
