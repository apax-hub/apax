import logging

from flax.core.frozen_dict import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

log = logging.getLogger(__name__)

def param_transfer(source_params, target_params, param_black_list):
    source_params = unfreeze(source_params)
    target_params = unfreeze(target_params)

    flat_source = flatten_dict(source_params)
    flat_target = flatten_dict(target_params)
    for p, v in flat_source.items():
        if p[-2] not in param_black_list:
            flat_target[p] = v
            log.info("Transfering parameter: %s", p)

    transfered_target = unflatten_dict(flat_target)
    transfered_target = freeze(transfered_target)
    return transfered_target
