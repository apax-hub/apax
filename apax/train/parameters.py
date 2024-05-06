import jax


@jax.jit
def tree_ema(tree1, tree2, alpha):
    """Exponential moving average of two pytrees."""
    ema = jax.tree_map(lambda a, b: alpha * a + (1 - alpha) * b, tree1, tree2)
    return ema


class EMAParameters:
    """Handler for tracking an exponential moving average of model parameters.
    The EMA parameters are used in the valitaion loop.

    Parameters
    ----------
    ema_start : int, default = 1
        Epoch at which to start averaging models.
    alpha : float, default = 0.9
        How much of the new model to use. 1.0 would mean no averaging, 0.0 no updates.
    """

    def __init__(self, ema_start: int, alpha: float = 0.9) -> None:
        self.alpha = alpha
        self.ema_start = ema_start
        self.ema_params = None

    def update(self, opt_params, epoch):
        if epoch > self.ema_start:
            self.ema_params = tree_ema(opt_params, self.ema_params, self.alpha)
        else:
            self.ema_params = opt_params
