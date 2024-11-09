from model.deepar.torch import model as torch_model
from model.deepar.mxnet import model as mx_model

def model(
        context_length: int,
        prediction_length: int,
        freq: str,
        epochs: int,
        num_parallel_samples: int,
        model_type="torch"
    ):
    if model_type == "torch":
        model = torch_model(context_length, prediction_length, freq, epochs, num_parallel_samples)
    elif model_type == "mxnet":
        model = mx_model(context_length, prediction_length, freq, epochs, num_parallel_samples)

    return model