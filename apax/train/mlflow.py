from typing import Any, Dict, Optional

import mlflow


class MLFlowLogger:
    def __init__(
        self, experiment: Optional[str] = None, run_name: Optional[str] = None
    ) -> None:
        """
        Initialize the MLFlow logger.

        :param log_dir: The directory where MLFlow logs will be stored.
        :param experiment_name: Name of the experiment in MLFlow.
        :param run_name: Name of the run to be logged in MLFlow.
        """
        mlflow.login()
        if experiment is not None:
            mlflow.set_experiment(experiment)

        # Start a new MLFlow run
        self.run = mlflow.start_run(run_name=run_name)

    def set_model(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics at the end of each epoch."""
        if logs is None:
            return
        mlflow.log_metrics(logs, step=epoch, synchronous=False)

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """
        End the current MLFlow run.
        """
        _ = logs
        mlflow.end_run()
