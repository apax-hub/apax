import jax
from jax import tree_util


@jax.jit
def tree_ema(new, average, decay):
    """One exponential-moving-average step: decay * average + (1 - decay) * new."""
    return tree_util.tree_map(lambda n, a: decay * a + (1 - decay) * n, new, average)


class EMAParameters:
    """Exponential moving average of the model parameters, updated after every optimizer step.
    Validation and checkpoints use the averaged parameters.

    Parameters
    ----------
    ema_start : int, default = 0
        Epoch from which to average. Before it, the average follows the raw parameters.
    decay : float, default = 0.999
        Per-step weight of the running average.
    """

    def __init__(self, ema_start: int, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_start = ema_start
        self.ema_params = None

    def update(self, params, epoch):
        if epoch < self.ema_start or self.ema_params is None:
            self.ema_params = params
        else:
            self.ema_params = tree_ema(params, self.ema_params, self.decay)
