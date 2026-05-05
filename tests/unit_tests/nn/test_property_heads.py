"""Property-head readout dispatch — kind discriminator + cross-model guards.

Covers the (b)+(c) follow-up to PR #558. The contract:

- ``MaceBuilder.build_readout(head_config)`` where ``head_config`` is a
  property-head config (i.e. not ``self.config``) dispatches on
  ``head_config["kind"]``: ``"mace"`` returns a ``MaceReadout``,
  ``"standard"`` raises with an actionable error.
- ``ModelBuilder.build_readout`` (parent — used by GMNN/EquivMP/So3krates)
  raises when a property head sets ``kind="mace"`` because their descriptors
  do not produce the per-layer-concatenated feature shape ``MaceReadout``
  requires.
- The energy-readout path (``head_config is self.config``) is unaffected.
"""
import pytest

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
            "use_cueq": False,
        },
        readout={"kind": "mace", "MLP_irreps": "16x0e"},
        property_heads=property_heads,
    )
    return cfg.model_dump()


def _gmnn_cfg(property_heads):
    cfg = GMNNConfig(property_heads=property_heads)
    return cfg.model_dump()


def test_mace_property_head_kind_mace_builds_mace_readout():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([{"name": "charges", "kind": "mace"}])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
    assert readout.hidden_dim == 8
    assert readout.MLP_irreps == "16x0e"


def test_mace_property_head_kind_standard_raises():
    cfg = _mace_cfg([{"name": "charges", "kind": "standard"}])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    msg = str(excinfo.value)
    assert "charges" in msg
    assert "kind='mace'" in msg or 'kind="mace"' in msg


def test_mace_property_head_default_kind_is_standard_so_it_raises_on_mace():
    cfg = _mace_cfg([{"name": "charges"}])  # default kind="standard"
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    assert "charges" in str(excinfo.value)


def test_gmnn_property_head_default_kind_builds_atomistic_readout():
    from apax.layers.readout import AtomisticReadout

    cfg = _gmnn_cfg([{"name": "charges", "nn": [64, 64]}])
    builder = GMNNBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, AtomisticReadout)
    assert tuple(readout.units) == (64, 64)


def test_gmnn_property_head_kind_mace_raises():
    cfg = _gmnn_cfg([{"name": "charges", "kind": "mace"}])
    builder = GMNNBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]

    with pytest.raises(ValueError) as excinfo:
        builder.build_readout(head_cfg)

    msg = str(excinfo.value)
    assert "charges" in msg
    assert "MaceReadout" in msg


def test_mace_property_head_mace_readout_n_shallow_members_propagates():
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([
        {"name": "charges", "kind": "mace", "n_shallow_members": 4},
    ])
    builder = MaceBuilder(cfg, n_species=5)
    head_cfg = cfg["property_heads"][0]
    readout = builder.build_readout(head_cfg)

    assert isinstance(readout, MaceReadout)
    assert readout.n_shallow_ensemble == 4


def test_mace_energy_head_unchanged_by_property_head_guards():
    """Identity check ``head_config is self.config`` keeps the energy path intact."""
    from apax.layers.readout import MaceReadout

    cfg = _mace_cfg([])  # no property heads
    builder = MaceBuilder(cfg, n_species=5)
    readout = builder.build_readout(builder.config)

    assert isinstance(readout, MaceReadout)
    assert readout.num_interactions == 2
