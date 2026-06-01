"""MaceBuilder coerces descriptor.interactions from list to tuple of dicts."""

import pytest


def test_mace_builder_coerces_interactions_list_to_tuple_of_dicts(tmp_path):
    pytest.importorskip("e3nn_jax")
    pytest.importorskip("cuequivariance_jax")

    from apax.config.model_config import MaceModelConfig
    from apax.nn.builder import MaceBuilder

    cfg = MaceModelConfig(
        basis={"name": "bessel", "variant": "standard", "n_basis": 8, "r_max": 6.0},
        radial_embedding={"num_polynomial_cutoff": 5, "distance_transform": None},
        descriptor={
            "max_ell": 3,
            "hidden_irreps": "16x0e + 16x1o",
            "correlation": 3,
            "interactions": [
                {"name": "RealAgnosticDensity"},
                {"name": "RealAgnosticDensityResidual"},
            ],
            "avg_num_neighbors": 1.0,
        },
        readout={"MLP_irreps": "16x0e"},
        descriptor_dtype="fp32",
        readout_dtype="fp32",
        scale_shift_dtype="fp64",
    ).model_dump()

    assert isinstance(cfg["descriptor"]["interactions"], list)

    builder = MaceBuilder(cfg, n_species=119)
    descriptor = builder.build_descriptor(apply_mask=False)

    assert isinstance(descriptor.interactions, tuple), (
        f"interactions must be a tuple, got "
        f"{type(descriptor.interactions).__name__}: {descriptor.interactions!r}"
    )
    assert descriptor.interactions == (
        {"name": "RealAgnosticDensity"},
        {"name": "RealAgnosticDensityResidual"},
    )
