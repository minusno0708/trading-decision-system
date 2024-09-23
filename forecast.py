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

    plt.clf()
    plt.close()

def save_model(model: DeepAREstimator, name: str):
    torch.save(model, f"model/{name}.pth")

def load_model(name: str) -> DeepAREstimator:
    model = torch.load(f"model/{name}.pth")
    return model

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

def train_model(train_data: pd.DataFrame) -> DeepAREstimator:
    dataset = ListDataset(
        [{"start": train_data.index[0], "target": train_data["close"]}],
        freq="D",
    )

    return DeepAREstimator(
        prediction_length=output_length,
        context_length=input_length,
        freq="D",
        trainer_kwargs={"max_epochs": 10},
        num_layers = 2,
        hidden_size = 40,
        lr = 0.001,
        weight_decay = 1e-08,
        dropout_rate = 0.1,
        patience = 10,
        num_feat_dynamic_real = 0,
        num_feat_static_cat = 0,
        num_feat_static_real = 0,
        scaling = True,
        num_parallel_samples = 100,
        batch_size = 32,
        num_batches_per_epoch = 50,
    ).train(dataset)

def forecast(model: DeepAREstimator, test_data: pd.DataFrame) -> list:
    dataset = ListDataset(
        [{"start": test_data.index[0], "target": test_data["close"]}],
        freq="D",
    )

    forecasts = list(model.predict(dataset))

    return forecasts
    

if __name__ == "__main__":
    train_data, test_data = data_loader(data_path)

    #draw_graph(list(train_data.index), list(train_data["close"]), "train_data")
    #draw_graph(list(test_data.index), list(test_data["close"]), "test_data")

    if train_flag:
        model = train_model(train_data)
        save_model(model, "model")
    else:
        model = load_model("model")

    i = 51

    target_data = test_data.iloc[range(i, input_length + i), [0]]
    correct_data = test_data.iloc[range(input_length + i, input_length + output_length + i), [0]]

    forecasts = forecast(model, target_data)

    # 予測結果を描画
    plt.plot(target_data.index, target_data.values, label="target")
    plt.plot(correct_data.index, correct_data.values, label="correct")

    forecasts[0].plot()

    plt.legend()
    plt.savefig("output/forecast.png")

    print("success")
