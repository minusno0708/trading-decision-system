import pandas as pd

from gluonts.mx.model.deepar import DeepAREstimator as MxDeepAREstimator
from gluonts.mx.trainer import Trainer

def model(
        context_length: int,
        prediction_length: int,
        epochs: int,
        num_parallel_samples: int,
    ):

    trainer = Trainer(
        epochs=epochs,
    )

    return MxDeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_parallel_samples=num_parallel_samples,
        cell_type="lstm",
        trainer=trainer,
        freq="D",
    )
