import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch

from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions

torch.set_float32_matmul_precision("high")

data_path = "dataset/btc.csv"

input_length = 30
output_length = 7

train_flag = False

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/{name}.png")

def save_model(model: DeepAREstimator, name: str):
    torch.save(model, f"model/{name}.pth")

def load_model(name: str) -> DeepAREstimator:
    model = torch.load(f"model/{name}.pth")
    return model

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

    train_data = df_row.iloc[:n_train + output_length]
    test_data = df_row.iloc[n_train:]

    return train_data, test_data

def train_model(train_data: pd.DataFrame) -> DeepAREstimator:
    dataset = ListDataset(
        [{"start": train_data.index[0], "target": train_data["close"]}],
        freq="D",
    )

    return DeepAREstimator(
        context_length=input_length,
        prediction_length=output_length,
        freq="D",
        trainer_kwargs={"max_epochs": 5},
    ).train(dataset)

def forecast(model: DeepAREstimator, test_data: pd.DataFrame) -> [list, list]:
    dataset = ListDataset(
        [{"start": test_data.index[0], "target": test_data["close"]}],
        freq="D",
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=model,
        num_samples=100,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss
    

if __name__ == "__main__":
    train_data, test_data = data_loader(data_path)

    draw_graph(list(train_data.index), list(train_data["close"]), "train_data")


    if train_flag:
        model = train_model(train_data)
        save_model(model, "model")
    else:
        model = load_model("model")

    forecasts, tss = forecast(model, test_data)

    draw_graph(list(test_data.index), list(test_data["close"]), "test_data")

    print("success")

