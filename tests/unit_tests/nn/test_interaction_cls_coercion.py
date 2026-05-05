"""I6 — MaceBuilder coerces config['interaction_cls'] list to tuple before
passing to MaceRepresentation, which is typed as ``str | tuple[str, ...]``.
"""
import pytest


def test_mace_builder_coerces_interaction_cls_list_to_tuple(tmp_path):
    """A list ``interaction_cls`` config (as emitted by YAML) becomes a tuple
    by the time it reaches ``MaceRepresentation``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture (unused; reserved if config-on-disk needs to be exercised).
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

    # interaction_cls survives as a list through Pydantic's union acceptance.
    assert isinstance(cfg["interaction_cls"], list)

    builder = MaceBuilder(cfg, n_species=119)
    descriptor = builder.build_descriptor(apply_mask=False)

    assert isinstance(descriptor.interaction_cls, tuple), (
        f"interaction_cls passed to MaceRepresentation must be tuple, "
        f"got {type(descriptor.interaction_cls).__name__}: "
        f"{descriptor.interaction_cls!r}"
    )
    assert descriptor.interaction_cls == (
        "RealAgnosticDensity",
        "RealAgnosticDensityResidual",
    )
