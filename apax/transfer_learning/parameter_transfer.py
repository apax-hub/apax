import logging
from typing import Dict, List, Union

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.config.train_config import TransferLearningConfig
from apax.train.checkpoints import load_params

log = logging.getLogger(__name__)


def black_list_param_transfer(
    source_params: Union[FrozenDict, Dict],
    target_params: Union[FrozenDict, Dict],
    param_black_list: List[str],
) -> FrozenDict:
    """Transfer parameters from one dictionary to another, while keeping
        some key-value pairs unchanged.

    Args:
        source_params (Union[FrozenDict, dict]): source parameters
        target_params (Union[FrozenDict, dict]): target parameters
        param_black_list (list[str]): list of keys to keep unchanged.

    Returns:
        transfered_target (dict): target_params with key-value pairs updated.

    """
    source_params = unfreeze(source_params)
    target_params = unfreeze(target_params)

    flat_source = flatten_dict(source_params)
    flat_target = flatten_dict(target_params)
    for p, v in flat_source.items():
        if p[-2] not in param_black_list:
            flat_target[p] = v
            log.info("Transferring parameter: %s", p[-2])

    transfered_target = unflatten_dict(flat_target)
    transfered_target = freeze(transfered_target)
    return transfered_target


def transfer_parameters(
    state: TrainState, ckpt_config: TransferLearningConfig
) -> TrainState:
    """Transfer the parameters from the checkpoint to the train state.

    Args:
        state (TrainState): train state
        ckpt_config (TransferLearningConfig): transfer learning configuration

    Returns:
        state (TrainState): TrainState with the `params` attribute updated
            according to the transfer learning configuration.
    """
    source_params = load_params(ckpt_config.base_model_checkpoint)
    log.info("Transferring parameters from %s", ckpt_config.base_model_checkpoint)
    params = black_list_param_transfer(
        source_params, state.params, ckpt_config.reset_layers
    )
    state = state.replace(params=params)
    return state
