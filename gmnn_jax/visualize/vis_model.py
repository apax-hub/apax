import haiku as hk
from rich import print

def model_tabular(model, sample_input):
    print(hk.experimental.tabulate(model)(sample_input))
