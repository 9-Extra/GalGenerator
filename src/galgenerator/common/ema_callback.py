from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_multi_avg_fn


class EMAWeightAveraging(WeightAveraging):
    """
    EMA (Exponential Moving Average) callback for Lightning.

    Usage:
        trainer = Trainer(callbacks=[EMAWeightAveraging(decay=0.999)])

    This callback automatically:
    - Updates EMA weights after each training step
    - Swaps to EMA weights during validation/prediction
    - Restores training weights after validation/prediction
    - Persists EMA weights to the model at the end of training
    - Saves/loads EMA state in checkpoints
    """

    def __init__(self, decay: float = 0.999):
        super().__init__(
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=True,
        )
