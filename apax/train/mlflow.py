import mlflow


class MLFlowLogger:
    def __init__(self, experiment=None, run_name=None):
        """
        Initialize the MLFlow logger.

        :param log_dir: The directory where MLFlow logs will be stored.
        :param experiment_name: Name of the experiment in MLFlow.
        :param run_name: Name of the run to be logged in MLFlow.
        """
        mlflow.login()
        mlflow.set_experiment(experiment)

        # Start a new MLFlow run
        self.run = mlflow.start_run(run_name=run_name)

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

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs is None:
            return
        mlflow.log_metrics(logs, step=epoch, synchronous=False)

    def on_train_end(self, logs=None):
        """
        End the current MLFlow run.
        """
        mlflow.end_run()
