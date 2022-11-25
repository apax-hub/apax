import haiku as hk
import jax.numpy as jnp


def transfer_parameters(params, tf_params):
    params_mutable = hk.data_structures.to_mutable_dict(params)

    trained_params = {k: jnp.asarray(v) for k, v in tf_params.items()}

    params_mutable["gmnn/~/descriptor/~/radial_fn"][
        "atomic_type_embedding"
    ] = trained_params["atomic_type_embedding/Variable:0"]
    params_mutable["gmnn/~/scale_shift"]["scale_per_element"] = trained_params[
        "gmnn/energy_prediction/scale_shift_output/scale:0"
    ]
    params_mutable["gmnn/~/scale_shift"]["shift_per_element"] = trained_params[
        "gmnn/energy_prediction/scale_shift_output/shift:0"
    ]

    params_mutable["gmnn/~/readout/~/linear_0"]["w"] = trained_params[
        "gmnn/sequential_layer_1/linear_layer_2/linear_weight:0"
    ]
    params_mutable["gmnn/~/readout/~/linear_0"]["b"] = trained_params[
        "gmnn/sequential_layer_1/linear_layer_2/linear_bias:0"
    ]

    params_mutable["gmnn/~/readout/~/linear_1"]["w"] = trained_params[
        "gmnn/energy_prediction/sequential_layer/linear_layer/linear_weight:0"
    ]
    params_mutable["gmnn/~/readout/~/linear_1"]["b"] = trained_params[
        "gmnn/energy_prediction/sequential_layer/linear_layer/linear_bias:0"
    ]

    params_mutable["gmnn/~/readout/~/linear_2"]["w"] = trained_params[
        "gmnn/energy_prediction/sequential_layer/linear_layer_1/linear_weight:0"
    ]
    params_mutable["gmnn/~/readout/~/linear_2"]["b"] = trained_params[
        "gmnn/energy_prediction/sequential_layer/linear_layer_1/linear_bias:0"
    ]

    return hk.data_structures.to_immutable_dict(params_mutable)
