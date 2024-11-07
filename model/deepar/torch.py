import torch

from gluonts.torch import DeepAREstimator as DeepAREstimator

torch.set_float32_matmul_precision("high")

def model(
        context_length: int,
        prediction_length: int,
        epochs: int,
        num_parallel_samples: int,
    ):
    return DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq="D",
        trainer_kwargs={"max_epochs": epochs},
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
        num_parallel_samples = num_parallel_samples,
        batch_size = 32,
        num_batches_per_epoch = 50,
    )