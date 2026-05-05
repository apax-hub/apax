"""MaceBuilder coerces flat interaction_cls config into the descriptor's
``interactions`` field of typed dicts.
"""

import pytest


def test_mace_builder_translates_interaction_cls_to_interactions(tmp_path):
    """A list ``interaction_cls`` config becomes a tuple of dicts on the
    descriptor side.
    """
    pytest.importorskip("e3nn_jax")
    pytest.importorskip("cuequivariance_jax")

    from apax.config.model_config import MaceModelConfig
    from apax.nn.builder import MaceBuilder

    cfg = MaceModelConfig(
        r_max=6.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        hidden_irreps="16x0e + 16x1o",
        num_interactions=2,
        correlation=3,
        interaction_cls=[
            "RealAgnosticDensity",
            "RealAgnosticDensityResidual",
        ],
        use_cueq=False,
        descriptor_dtype="fp32",
        readout_dtype="fp32",
        scale_shift_dtype="fp64",
        avg_num_neighbors=1.0,
        distance_transform=None,
    ).model_dump()

    builder = MaceBuilder(cfg, n_species=119)
    descriptor = builder.build_descriptor(apply_mask=False)

    assert isinstance(descriptor.interactions, tuple), (
        f"interactions on MaceRepresentation must be tuple, "
        f"got {type(descriptor.interactions).__name__}: "
        f"{descriptor.interactions!r}"
    )
    assert descriptor.interactions == (
        {"name": "RealAgnosticDensity"},
        {"name": "RealAgnosticDensityResidual"},
    )
