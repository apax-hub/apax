import jax
import jax.numpy as jnp


@jax.jit
def tree_ema(tree1, tree2, alpha):
    """Exponential moving average of two pytrees."""
    ema = jax.tree_map(lambda a, b: alpha * a + (1 - alpha) * b, tree1, tree2)
    return ema

@jax.jit
def tree_sum(pytrees):
    summed_tree = jax.tree_util.tree_map(lambda *x: sum(x), *pytrees)
    return summed_tree

def tree_div(pytree, n):
    tree = jax.tree_util.tree_map(lambda x: x/n, pytree)
    return tree


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


class SWAParameters:
    """Handler for tracking an exponential moving average of model parameters.
    The EMA parameters are used in the valitaion loop.

    Parameters
    ----------
    ema_start : int, default = 1
        Epoch at which to start averaging models.
    alpha : float, default = 0.9
        How much of the new model to use. 1.0 would mean no averaging, 0.0 no updates.
    """

    def __init__(self, swa_start: int, period: int) -> None:
        self.swa_start = swa_start
        self.period = period
        self.params = None
        self.count = 0

    def update(self, params, epoch):
        epoch = epoch + 1
        has_started = epoch >= self.swa_start
        should_collect = (epoch - self.swa_start) % self.period == 0
        if has_started and should_collect:
            if self.count == 0:
                self.params = params
            else:
                print(epoch, self.count)
                self.params = tree_sum([self.params, params])
            self.count += 1 
        
            # snapshot = self.params["params"]["energy_model"]["atomistic_model"]["descriptor"]["radial_fn"]["atomic_type_embedding"][0,0,0,0]
            # snapshot1 = params["params"]["energy_model"]["atomistic_model"]["descriptor"]["radial_fn"]["atomic_type_embedding"][0,0,0,0]
            # print(snapshot, snapshot1)

    def average_weights(self):
        avg_weights = tree_div(self.params, self.count)
        # snapshot = avg_weights["params"]["energy_model"]["atomistic_model"]["descriptor"]["radial_fn"]["atomic_type_embedding"][0,0,0,0]
        # print(snapshot)
        # print("count", self.count)
        return avg_weights
