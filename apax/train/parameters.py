import jax


@jax.jit
def tree_ema(tree1, tree2, alpha):
    ema = jax.tree_map(lambda a, b: alpha * a + (1 - alpha) * b, tree1, tree2)
    return ema


class EMAParameters:
    def __init__(self, ema_start, alpha) -> None:
        self.alpha = alpha
        self.ema_start = ema_start
        self.ema_params = None

    def update(self, opt_params, epoch):
        if epoch > self.ema_start:
            self.ema_params = tree_ema(opt_params, self.ema_params, self.alpha)
        else:
            self.ema_params = opt_params
