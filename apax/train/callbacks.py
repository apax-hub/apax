import collections
import csv
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger, TensorBoard

from apax.config.train_config import Config

try:
    from apax.train.mlflow import MLFlowLogger
except ImportError:
    MLFlowLogger = None

log = logging.getLogger(__name__)


class CallbackCollection:
    def __init__(self, callbacks: list) -> None:
        self.callbacks = callbacks

    def on_train_begin(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def on_train_batch_begin(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch)

    def on_train_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_train_batch_end(batch)

    def on_epoch_end(self, epoch, logs):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_test_batch_end(self, batch, logs=None):
        for cb in self.callbacks:
            cb.on_test_batch_end(batch, logs)


def format_str(k):
    return f"{k:.5f}"


class CSVLoggerApax(CSVLogger):
    def __init__(self, filename, separator=",", append=False):
        super().__init__(filename, separator=separator, append=append)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(format_str, k))}]"'
            else:
                return format_str(k)

        if self.keys is None:
            self.keys = sorted(logs.keys())
            # When validation_freq > 1, `val_` keys are not in first epoch logs
            # Add the `val_` keys so that its part of the fieldnames of writer.
            val_keys_found = False
            for key in self.keys:
                if key.startswith("val_"):
                    val_keys_found = True
                    break
            if not val_keys_found:
                self.keys.extend(["val_" + k for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs.get(key, "NA"))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(format_str, k))}]"'
            else:
                return format_str(k)

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


def initialize_callbacks(config: Config, model_version_path: Path):
    callback_configs = config.callbacks
    log.info("Initializing Callbacks")

    dummy_model = tf.keras.Model()
    dummy_model.compile(loss="mse", optimizer="adam")
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
            "write_graph": False,
        },
        "mlflow": {
            "class": MLFlowLogger,
            "log_path": model_version_path,
            "path_arg_name": "log_dir",
            "kwargs": {"run_name": config.data.experiment},
        },
    }

    callbacks = []
    for callback_config in callback_configs:
        if callback_config.name == "mlflow":
            callback = MLFlowLogger(
                experiment=callback_config.experiment, run_name=config.data.experiment
            )
        else:
            callback_info = callback_dict[callback_config.name]

            path_arg_name = callback_info["path_arg_name"]
            path = {path_arg_name: callback_info["log_path"]}

            kwargs = callback_info["kwargs"]
            callback = callback_info["class"](**path, **kwargs)
            callback.set_model(callback_info["model"])
        callbacks.append(callback)

    return CallbackCollection(callbacks)
