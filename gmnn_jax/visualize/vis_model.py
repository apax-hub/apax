import haiku as hk
from rich import print


def model_tabular(model, R, Z, idx):
    columns = (
        "module",
        "owned_params",
        "output",
        "params_size",
        "params_bytes",
    )  # , "config", "input"
    # We drop config and input since they make the table too wide to fit on a screen

    table = hk.experimental.tabulate(model, columns=columns)(R, Z, idx)
    print(table)
