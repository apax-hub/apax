import collections
import csv
import logging
import typing as t
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, CSVLogger, TensorBoard

from apax.config.optuna_config import get_pruner
from apax.config.train_config import Config

try:
    from apax.train.mlflow import MLFlowLogger
except ImportError:
    MLFlowLogger = None

try:
    import optuna
except ImportError:
    optuna = None

if t.TYPE_CHECKING:
    import optuna

log = logging.getLogger(__name__)


class CallbackCollection:
    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = callbacks

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_test_batch_end(batch, logs)


def format_str(k: Any) -> str:
    return f"{k:.5f}"


class CSVLoggerApax(CSVLogger):
    def __init__(
        self, filename: str, separator: str = ",", append: bool = False
    ) -> None:
        super().__init__(filename, separator=separator, append=append)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}

        def handle_value(k: Any) -> Any:
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

    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}

        def handle_value(k: Any) -> Any:
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


class KerasPruningCallback(Callback):
    """Adapted from https://optuna-integration.readthedocs.io/en/latest/_modules/optuna_integration/keras/keras.html#KerasPruningCallback
    Keras callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    keras/keras_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy.

    Args:
        study_id:
            id number of study
        trial_id:
            id of current trial
        study_log_file:
            path to log file
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
        interval:
            Check if trial should be pruned every n-th epoch. By default ``interval=1`` and
            pruning is performed after every epoch. Increase ``interval`` to run several
            epochs faster before applying pruning.
    """  # noqa: E501

    def __init__(
        self,
        study_name: str,
        trial_id: int,
        study_log_file: t.Union[str, Path],
        monitor: str = "val_loss",
        interval: int = 1,
    ) -> None:
        super().__init__()

        if optuna is None:
            raise ImportError(f"{self.__name__} requires optuna, but could not import it")

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(study_log_file))
        )
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=None, pruner=None
        )

        if "pruner" in study.user_attrs:
            pruner_class = get_pruner(study.user_attrs["pruner"]["name"])
            study.pruner = pruner_class(**study.user_attrs["pruner"]["kwargs"])
        else:
            study.pruner = None

        self._monitor = monitor
        self._interval = interval
        self._trial = optuna.trial.Trial(study, trial_id)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if epoch % self._interval != 0:
            return

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self._monitor)
            )
            log.warning(message)
            return

        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            log.info(message)
            raise optuna.TrialPruned(message)


def initialize_callbacks(config: Config, model_version_path: Path) -> CallbackCollection:
    callback_configs = config.callbacks
    log.info("Initializing Callbacks")

    dummy_model = tf.keras.Model()
    dummy_model.compile(loss="mse", optimizer="adam")
    callback_dict: Dict[str, Dict[str, Any]] = {
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

    callbacks: List[Callback] = []
    for callback_config in callback_configs:
        if callback_config.name == "mlflow":
            callback = MLFlowLogger(
                experiment=callback_config.experiment, run_name=config.data.experiment
            )
        elif callback_config.name == "pruning":
            callback = KerasPruningCallback(
                study_name=callback_config.study_name,
                trial_id=callback_config.trial_id,
                study_log_file=callback_config.study_log_file,
                monitor=callback_config.monitor,
                interval=callback_config.interval,
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
