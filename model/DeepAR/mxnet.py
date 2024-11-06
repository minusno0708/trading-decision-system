import pandas as pd

from gluonts.mx.model.deepar import DeepAREstimator as MxDeepAREstimator
from gluonts.mx.trainer import Trainer


def model(input_length: int, output_length: int):
    trainer = Trainer(
        epochs=50,
    )

    return MxDeepAREstimator(
        prediction_length=output_length,
        context_length=input_length,
        trainer=trainer,
        freq="D",
    )
