import matplotlib.pyplot as plt

from data_provider.data_loader import DataLoader
from model.DeepAR import Model

input_length = 30
output_length = 7

train_flag = False

def draw_graph(x_data: list, y_data: list, name: str):
    plt.plot(x_data, y_data)
    plt.savefig(f"output/images/{name}.png")

    plt.clf()
    plt.close()

if __name__ == "__main__":
    data_loader = DataLoader(output_length)
    train_data, test_data = data_loader.load("btc.csv")

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
