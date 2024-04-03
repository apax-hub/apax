from apax import ops


def swish(x):
    out = 1.6765324703310907 * ops.swish(x)
    return out


def inverse_softplus(x):
    return ops.log(ops.exp(x) - 1.0)
