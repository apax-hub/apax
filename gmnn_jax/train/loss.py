import dataclasses

import jax.numpy as jnp


def weighted_squared_error(label, prediction, divisor=1):
    return jnp.sum((label - prediction) ** 2 / divisor)

@dataclasses.dataclass
class Loss:
    key: str
    loss_type: str
    weight: float = 1.0


    def __call__(self, prediction: dict, label: dict)-> float:
        divisor = 1.0
        if self.loss_type == "structures":
            divisor = label["n_atoms"] ** 2
        elif self.loss_type == "vibrations":
            divisor = label["n_atoms"]

        
        loss = weighted_squared_error(label[self.key], prediction[self.key], divisor=1)
        # TODO add stress multiplication with cell volume as dataset.map
        

        return self.weight * loss

