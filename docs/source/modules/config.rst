Configuration
=============

:mod:`apax.config`

Training Configuration
----------------------

.. autoclass:: apax.config.train_config.Config
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.OptimizerConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.DataConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.DatasetConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.CachedDataset
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.PBDDataset
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.OTFDataset
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.MetricsConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.LossConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.TrainProgressbarConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.CSVCallback
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.TBCallback
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.MLFlowCallback
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.KerasPruningCallback
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.TransferLearningConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.train_config.WeightAverage
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields


Model Configuration
----------------------

.. autoclass:: apax.config.model_config.GaussianBasisConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.BesselBasisConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.FullEnsembleConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.ShallowEnsembleConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.Correction
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.ZBLRepulsion
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.ExponentialRepulsion
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.LatentEwald
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.PropertyHead
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.GMNNConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.EquivMPConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.model_config.So3kratesConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields



Molecular Dynamics Configuration
--------------------------------

.. autoclass:: apax.config.md_config.MDConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.md_config.NPTOptions
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

Hyperparameter Optimization Configuration
-----------------------------------------

.. autoclass:: apax.config.optuna_config.OptunaPrunerConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.optuna_config.OptunaSamplerConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields

.. autoclass:: apax.config.optuna_config.OptunaConfig
    :members:
    :exclude-members: model_config, model_computed_fields, model_fields
