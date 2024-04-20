import torch.nn as nn
import torch


class SwishT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        h = 1.6765324703310907 * torch.nn.functional.silu(x)
        return h
