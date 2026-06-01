"""Property-head readout dispatch — readout architecture follows model type.

GMNN/EquivMP/So3krates property heads always build :class:`AtomisticReadout`;
MACE property heads always build :class:`MaceReadout`.
"""

from apax.config.model_config import GMNNConfig, MaceModelConfig
from apax.nn.builder import GMNNBuilder, MaceBuilder


def _mace_cfg(property_heads):
    cfg = MaceModelConfig(
        basis={"name": "bessel", "variant": "standard", "n_basis": 4, "r_max": 5.0},
        radial_embedding={"num_polynomial_cutoff": 5, "distance_transform": None},
        descriptor={
            "max_ell": 1,
            "hidden_irreps": "8x0e",
            "correlation": 2,
            "interactions": [
                {"name": "RealAgnosticResidual"},
                {"name": "RealAgnosticResidual"},
            ],
            "avg_num_neighbors": 1.0,
        },
        readout={"MLP_irreps": "16x0e"},
        property_heads=property_heads,
    )
    return cfg.model_dump()


def _gmnn_cfg(property_heads):
    cfg = GMNNConfig(property_heads=property_heads)
    return cfg.model_dump()


def test_mace_property_head_builds_mace_readout():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([{"name": "charges"}])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
    assert readout.MLP_irreps == "16x0e"


def test_gmnn_property_head_builds_atomistic_readout():
    from apax.layers.readout import AtomisticReadout

    cfg = _gmnn_cfg([{"name": "charges", "nn": [64, 64]}])
    builder = GMNNBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, AtomisticReadout)
    assert tuple(readout.units) == (64, 64)


def test_mace_property_head_mace_readout_n_shallow_members_propagates():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg(
        [
            {"name": "charges", "n_shallow_members": 4},
        ]
    )
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.n_shallow_ensemble == 4


def test_mace_energy_head_uses_mace_readout():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([])
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
