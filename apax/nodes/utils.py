import logging

import matplotlib.pyplot as plt
import numpy as np


def get_flat_data_from_dict(data: dict, silent_ignore: bool = False) -> list:
    """Flatten a dictionary of lists into a single list.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    silent_ignore : bool, optional
        If True, the function will return the input if it is not a
        dictionary. If False, it will raise a TypeError.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_flat_data_from_dict(data)
        [1, 2, 3, 4, 5, 6]
    """
    if not isinstance(data, dict):
        if silent_ignore:
            return data
        else:
            raise TypeError(f"data must be a dictionary and not {type(data)}")

    flat_data = []
    for x in data.values():
        flat_data.extend(x)
    return flat_data


def plot_with_uncertainty(value, ylabel: str, xlabel: str, x=None, **kwargs) -> dict:
    """Parameters
    ----------
    value: data of shape (n, m) where n is the number of ensembles.
    x: optional x values of shape (m,)

    Returns
    -------

    """
    if isinstance(value, dict):
        data = value
    else:
        data = {
            "mean": np.mean(value, axis=0),
            "std": np.std(value, axis=0),
            "max": np.max(value, axis=0),
            "min": np.min(value, axis=0),
        }

    fig, ax = plt.subplots(**kwargs)
    if x is None:
        x = np.arange(len(data["mean"]))
    ax.fill_between(
        x,
        data["mean"] + data["std"],
        data["mean"] - data["std"],
        facecolor="lightblue",
    )
    if "max" in data:
        ax.plot(x, data["max"], linestyle="--", color="darkcyan")
    if "min" in data:
        ax.plot(x, data["min"], linestyle="--", color="darkcyan")
    ax.plot(x, data["mean"], color="black")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax, data


def check_duplicate_keys(dict_a: dict, dict_b: dict, log: logging.Logger) -> None:
    """Check if a key of dict_a is present in dict_b and then log a warning."""
    for key in dict_a:
        if key in dict_b:
            log.warning(
                f"Found <{key}> in given config file. Please be aware that <{key}>"
                " will be overwritten by MLSuite!"
            )
