"""Shape & contract tests for MaceRepresentation (injected radial_embedding)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apax.layers.descriptor.basis_functions import (
    MaceBesselBasis,
    MaceRadialEmbedding,
)
from apax.layers.descriptor.mace import MaceRepresentation


def _build_radial(n_basis: int = 8, r_max: float = 5.0):
    basis = MaceBesselBasis(n_basis=n_basis, r_max=r_max, dtype=jnp.float32)
    return MaceRadialEmbedding(
        basis_fn=basis,
        num_polynomial_cutoff=5,
        r_max=r_max,
    )


@pytest.fixture
def tiny_system():
    n_atoms = 4
    n_neighbors = 8
    rng = np.random.default_rng(0)
    dr_vec = jnp.asarray(rng.normal(size=(n_neighbors, 3))).astype(jnp.float32)
    Z = jnp.asarray([1, 8, 1, 6], dtype=jnp.int32)
    idx = jnp.asarray(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]],
        dtype=jnp.int32,
    )
    return dr_vec, Z, idx, n_atoms


def test_mace_representation_contract(tiny_system):
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(
        radial_embedding=_build_radial(n_basis=8, r_max=5.0),
        max_ell=2,
        hidden_irreps="16x0e + 16x1o",
        correlation=3,
        interactions=(
            {"name": "RealAgnosticResidual"},
            {"name": "RealAgnosticResidual"},
        ),
        avg_num_neighbors=1.0,
        num_elements=119,
        apply_mask=True,
        dtype=jnp.float32,
    )
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = model.apply(params, dr_vec, Z, idx)
    assert out.shape == (n_atoms, 16 * 2)
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out).all()


def test_mace_representation_rejects_no_scalar_irreps(tiny_system):
    dr_vec, Z, idx, _ = tiny_system
    model = MaceRepresentation(
        radial_embedding=_build_radial(),
        max_ell=2,
        hidden_irreps="16x1o",
        correlation=3,
        interactions=({"name": "RealAgnosticResidual"},),
        avg_num_neighbors=1.0,
        num_elements=119,
        apply_mask=True,
        dtype=jnp.float32,
    )
    with pytest.raises(ValueError, match="0e component"):
        model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)


def test_mace_representation_per_layer_variants(tiny_system):
    """Different interaction-block variants per layer are honoured."""
    dr_vec, Z, idx, n_atoms = tiny_system
    model = MaceRepresentation(
        radial_embedding=_build_radial(n_basis=4, r_max=5.0),
        max_ell=2,
        hidden_irreps="8x0e",
        correlation=2,
        interactions=(
            {"name": "RealAgnosticDensity"},
            {"name": "RealAgnosticDensityResidual"},
        ),
        avg_num_neighbors=1.0,
        num_elements=5,
        apply_mask=True,
        dtype=jnp.float32,
    )
    params = model.init(jax.random.PRNGKey(0), dr_vec, Z, idx)
    out = model.apply(params, dr_vec, Z, idx)
    # Two layers, hidden 8x0e -> 2*8 features per atom.
    assert out.shape == (n_atoms, 2 * 8)


def test_mace_representation_has_no_removed_fields():
    """Regression: legacy flat fields must be gone."""
    fields = MaceRepresentation.__dataclass_fields__
    for removed in (
        "r_max",
        "num_bessel",
        "num_polynomial_cutoff",
        "interaction_cls",
        "num_interactions",
    ):
        assert removed not in fields, f"{removed!r} should be gone"
    for kept in (
        "radial_embedding",
        "interactions",
        "max_ell",
        "hidden_irreps",
        "correlation",
    ):
        assert kept in fields, f"{kept!r} should be present"
