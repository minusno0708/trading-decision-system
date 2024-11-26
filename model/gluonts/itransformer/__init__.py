import torch

from gluonts.torch.model.i_transformer import ITransformerEstimator

torch.set_float32_matmul_precision("high")

def model(
        context_length: int,
        prediction_length: int,
        freq: str,
        epochs: int,
        num_parallel_samples: int,
        model_type="torch"
    ):
    return ITransformerEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
        num_parallel_samples=num_parallel_samples,
        trainer_kwargs={"max_epochs": epochs},
    )