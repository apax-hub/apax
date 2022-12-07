import os
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Extra, PositiveFloat, PositiveInt


class DataConfig(BaseModel):
    model_path: str
    model_name: str
    data_path: Optional[str] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    n_train: PositiveInt = 1000
    n_valid: PositiveInt = 100
    batch_size: PositiveInt = 32
    valid_batch_size: PositiveInt = 100
    shuffle_buffer_size: PositiveInt = 1000


class DescriptorConfig(BaseModel, frozen=False):
    n_radial: Union[PositiveInt, List[int]] = 5
    n_basis: Union[PositiveInt, List[int]] = 7
    n_contr: PositiveInt = 8
    cutoff: PositiveFloat = 6.0
    n_species: PositiveInt = 119
    r_min: float = 0.5
    emb_init: Union[List[str], str] = "uniform"
    use_all_features: bool = True


class NeuralNetConfig(BaseModel, extra=Extra.forbid, frozen=True):
    nn: List[PositiveInt] = [512, 512]
    weight_factor: PositiveFloat = 1.0
    weight_init_gain: PositiveFloat = 1.0
    bias_init_gain: PositiveFloat = 1.0
    bias_factor: PositiveFloat = 0.1
    raw_weight_factor: PositiveFloat = 1.0


class ModelConfig(BaseModel, extra=Extra.forbid):
    descriptor: DescriptorConfig = DescriptorConfig()
    representation: NeuralNetConfig = NeuralNetConfig()


class OptimizerConfig(BaseModel, frozen=True, extra=Extra.allow):
    name: str = "adam"
    emb_lr: PositiveFloat = 0.02
    nn_lr: PositiveFloat = 0.03
    scale_lr: PositiveFloat = 0.001
    shift_lr: PositiveFloat = 0.05
    transition_begin: int = 0
    opt_kwargs: dict = {}


class MetricsConfig(BaseModel, extra=Extra.forbid):
    name: str
    reductions: List[str]


class LossConfig(BaseModel, extra=Extra.forbid):
    name: str
    loss_type: str = "molecules"
    weight: PositiveFloat = 1.0


class CallbackConfig(BaseModel, frozen=True, extra=Extra.allow):
    name: str


class Config(BaseModel, frozen=True, extra=Extra.forbid):
    max_epoch: PositiveInt = 100
    seed: int = 1

    data: DataConfig
    model: ModelConfig
    metrics: List[MetricsConfig] = []
    loss: List[LossConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    callbacks: List[CallbackConfig] = [CallbackConfig(name="csv")]

    def dump_config(self, save_path):
        with open(os.path.join(save_path, "parameters.yaml"), "w") as conf:
            yaml.dump(self.dict(), conf, default_flow_style=False)
