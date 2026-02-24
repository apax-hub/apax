from apax.config.optuna_config import OptunaConfig, get_pruner

x = get_pruner("MedianPruner")
x.