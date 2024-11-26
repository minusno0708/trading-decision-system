from gluonts.mx.model.deepvar import DeepVAREstimator
from gluonts.mx.trainer import Trainer

def model(
        context_length: int,
        prediction_length: int,
        freq: str,
        epochs: int,
        num_parallel_samples: int,
    ):
    return DeepVAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_parallel_samples=num_parallel_samples,
        cell_type="lstm",
        trainer=Trainer(
            epochs=epochs,
        ),
        freq=freq,
    )
