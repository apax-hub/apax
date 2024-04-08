import torch.nn as nn
from apax.nn.impl.activation import swish


class SwishT(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(self, x):
        h = swish(x)
        return h
