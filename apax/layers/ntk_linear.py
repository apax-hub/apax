import flax.linen as nn
import jax.numpy as jnp

from apax.utils.convert import str_to_dtype


class NTKLinear(nn.Module):
    """Linear layer with activation.
    Corresponds to an NTK layer with "normal" and "zeros" for w and b initialization
    and "use_ntk" set to True.
    """

    units: int
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    dtype: str = "fp32"

    @nn.compact
    def __call__(self, inputs):
        dtype = str_to_dtype(self.dtype)
        inputs = inputs.astype(dtype)

        if self.w_init == "normal":
            w_initializer = nn.initializers.normal(1.0, dtype=dtype)
        elif self.w_init == "lecun":
            w_initializer = nn.initializers.lecun_normal(dtype=dtype)
        else:
            raise ValueError(f"Unknown weight initializer: {self.w_init}.")

        if self.b_init == "normal":
            b_initializer = nn.initializers.normal(1.0, dtype=dtype)
        elif self.b_init == "zeros":
            b_initializer = nn.initializers.constant(0.0, dtype=dtype)
        else:
            raise ValueError(f"Unknown bias initializer: {self.b_init}.")
        w = self.param("w", w_initializer, (inputs.shape[0], self.units), dtype)
        b = self.param("b", b_initializer, [self.units], dtype)

        wx = jnp.dot(inputs, w)

        if self.use_ntk:
            bias_factor = 0.1
            weight_factor = jnp.sqrt(1.0 / inputs.shape[0])
            prediction = weight_factor * wx + bias_factor * b
        else:
            prediction = wx + b

        assert prediction.dtype == dtype
        return prediction
