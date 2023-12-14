import logging

from flax.core.frozen_dict import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from apax.train.checkpoints import load_params

log = logging.getLogger(__name__)


def black_list_param_transfer(source_params, target_params, param_black_list):
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


def transfer_parameters(state, ckpt_config):
    source_params = load_params(ckpt_config.base_model_checkpoint)
    log.info("Transferring parameters from %s", ckpt_config.base_model_checkpoint)
    params = black_list_param_transfer(
        source_params, state.params, ckpt_config.reset_layers
    )
    state = state.replace(params=params)
    return state
