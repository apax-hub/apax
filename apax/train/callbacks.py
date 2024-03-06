import collections
import csv
import logging

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, TensorBoard

log = logging.getLogger(__name__)


class CSVLoggerApax(CSVLogger):
    def __init__(self, filename, separator=",", append=False):
        super().__init__(filename, separator=",", append=False)

    def on_test_batch_begin(self, batch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["batch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"batch": batch})
        row_dict.update((key, handle_value(logs.get(key, "NA"))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


def initialize_callbacks(callback_configs, model_version_path):
    log.info("Initializing Callbacks")

    dummy_model = tf.keras.Model()
    callback_dict = {
        "csv": {
            "class": CSVLoggerApax,
            "log_path": model_version_path / "log.csv",
            "path_arg_name": "filename",
            "kwargs": {"append": True},
            "model": dummy_model,
        },
        "tensorboard": {
            "class": TensorBoard,
            "log_path": model_version_path,
            "path_arg_name": "log_dir",
            "kwargs": {},
            "model": dummy_model,
        },
    }

    callback_configs = [config.name for config in callback_configs]
    if "csv" in callback_configs and "tensorboard" in callback_configs:
        csv_idx, tb_idx = callback_configs.index("csv"), callback_configs.index(
            "tensorboard"
        )
        msg = (
            "Using both csv and tensorboard callbacks is not supported at the moment."
            " Only the first of the two will be used."
        )
        print("Warning: " + msg)
        log.warning(msg)
        if csv_idx < tb_idx:
            callback_configs.pop(tb_idx)
        else:
            callback_configs.pop(csv_idx)

    callbacks = []
    for callback_config in callback_configs:
        callback_info = callback_dict[callback_config]

        path_arg_name = callback_info["path_arg_name"]
        path = {path_arg_name: callback_info["log_path"]}

        kwargs = callback_info["kwargs"]
        callback = callback_info["class"](**path, **kwargs)
        callback.set_model(callback_info["model"])
        callbacks.append(callback)

    return tf.keras.callbacks.CallbackList([callback])
