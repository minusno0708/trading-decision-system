from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.trainer import Trainer

def model(
        context_length: int,
        prediction_length: int,
        freq: str,
        epochs: int,
        num_parallel_samples: int,
    ):
    return TransformerEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_parallel_samples=num_parallel_samples,
        freq=freq,
        trainer=Trainer(
            epochs=epochs,
        ),
    )