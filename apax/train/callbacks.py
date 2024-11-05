import collections
import csv
import logging
from pathlib import Path
import csv
import threading
import queue
import time
import os

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
                return f"\"[{', '.join(map(format_str, k))}]\""
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
                return f"\"[{', '.join(map(format_str, k))}]\""
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


class ThreadedCSVLogger:
    def __init__(self, filepath, flush_interval=1.0):
        """
        Initialize the ThreadedCSVLogger.
        
        :param filepath: Path to the CSV file where logs will be saved.
        :param fieldnames: List of fieldnames (keys) for the CSV file. If not provided,
                           fieldnames will be inferred from the first logged metrics.
        :param flush_interval: Time interval (in seconds) to flush the log to the file.
        """
        self.filepath = filepath
        self.flush_interval = flush_interval
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread)
        self.csv_file = None
        self.writer = None
        self.headers_written = False
        self.last_epoch = -1  # Track the last logged epoch

        # Check if the file already exists and recover the last epoch
        if os.path.exists(self.filepath):
            self._recover_last_epoch()

        # Start the background thread for logging
        self.thread.start()
    
    def set_model(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def _recover_last_epoch(self):
        """Recover the last logged epoch from the CSV file if it exists."""
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row and 'epoch' in last_row:
                    self.last_epoch = int(last_row['epoch'])
                self.fieldnames = reader.fieldnames
                self.headers_written = True  # Headers were already written
        except (csv.Error, FileNotFoundError):
            pass  # Handle potential CSV read errors if the file is corrupted or does not exist

    def on_epoch_end(self, epoch, logs):
        """
        Log a dictionary of metrics to the CSV file asynchronously.
        
        :param metrics_dict: Dictionary containing metric names as keys and their values.
        """
        self.queue.put((epoch, logs))

    def _writer_thread(self):
        """
        Internal method that runs in a separate thread, writing metrics to the CSV file.
        """
        # Open the file in append mode
        with open(self.filepath, mode='a', newline='') as csv_file:
            self.csv_file = csv_file
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)

            # Write the headers if needed
            if not self.headers_written and self.fieldnames:
                self.writer.writeheader()
                self.headers_written = True

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    # Try to get a batch of logs from the queue (wait max flush_interval)
                    epoch, logs = self.queue.get(timeout=self.flush_interval)
                    metrics = logs.copy()
                    metrics["epoch"] = epoch
                    self._write_row(metrics)
                except queue.Empty:
                    pass  # Timeout reached, no new logs, but continue running

                # Optionally flush the file
                self.csv_file.flush()

    def _write_row(self, metrics_dict):
        """
        Write a single row of metrics to the CSV file.
        
        :param metrics_dict: Dictionary containing metric names and values.
        """
        fieldnames = list(metrics_dict.keys())
        self.writer.fieldnames = fieldnames

        # Write headers once we have the fieldnames
        if not self.headers_written:
            self.writer.writeheader()
            self.headers_written = True
        
        self.writer.writerow(metrics_dict)

    def on_train_end(self, logs=None):
        """
        Signal the logger to stop and wait for the background thread to finish.
        """
        self.stop_event.set()
        self.thread.join()









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
        "tcsv": {
            "class": ThreadedCSVLogger,
            "log_path": model_version_path / "log.csv",
            "kwargs": {},
            "model": None,
            "path_arg_name": "filepath",
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
