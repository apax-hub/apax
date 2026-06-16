import jax
import jax.numpy as jnp
import pytest

from apax.layers.properties import PropertyHead
from apax.layers.readout import AtomisticReadout


@pytest.fixture
def setup_data():
    """Fixture to provide dummy data for testing."""
    n_atoms = 5
    n_features = 3
    n_species = 119

    g = jnp.ones((n_atoms, n_features))
    R = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )  # atom positions
    dr_vec = None
    Z = jnp.array([1, 6, 8, 1, 6])  # atomic numbers
    idx = None
    box = None

    return g, R, dr_vec, Z, idx, box


def test_property_head(setup_data):
    """Test PropertyHead class functionality."""
    g, R, dr_vec, Z, idx, box = setup_data

    # Instantiate PropertyHead
    property_head = PropertyHead(pname="property")

    # Test setup
    params = property_head.init(jax.random.PRNGKey(0), g, R, dr_vec, Z, idx, box)
    assert "scale_per_element" in params["params"]
    assert "shift_per_element" in params["params"]

    # Test forward pass for mode `l0`
    property_head = PropertyHead(
        pname="property", mode="l0", apply_mask=False, aggregation="none"
    )
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    assert output["property"].shape == (5, 1)  # Shape should match input atoms x features

    # Test aggregation: sum
    property_head = PropertyHead(
        pname="property", mode="l0", apply_mask=False, aggregation="sum"
    )
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    assert output["property"].shape == (1,)

    # Test mode `l1`
    property_head = PropertyHead(pname="property", mode="l1", apply_mask=True)
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    assert output["property"].shape == (5, 3)

    # Test mode `symmetric_l2`
    property_head = PropertyHead(pname="property", mode="symmetric_l2", apply_mask=True)
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    prop = output["property"]
    assert prop.shape == (5, 3, 3)
    assert jnp.allclose(prop, jnp.swapaxes(prop, -1, -2))

    # Test mode `symmetric_traceless_l2`
    property_head = PropertyHead(
        pname="property", mode="symmetric_traceless_l2", apply_mask=True
    )
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    prop = output["property"]
    assert prop.shape == (5, 3, 3)
    assert jnp.allclose(prop, jnp.swapaxes(prop, -1, -2))

    diag = jnp.diagonal(prop, axis1=1, axis2=2)
    assert jnp.all(jnp.sum(diag, axis=1) < 1e-4)  # traceless

    # Test ensemble detection
    property_head = PropertyHead(
        pname="property",
        readout=AtomisticReadout(n_shallow_ensemble=10),
        mode="l0",
        apply_mask=False,
        aggregation="mean",
    )
    params = property_head.init(jax.random.PRNGKey(0), g, R, dr_vec, Z, idx, box)
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    assert "property_uncertainty" in output


def test_property_head_uncertainty_is_std(setup_data):
    """The `<name>_uncertainty` output must be a standard deviation.

    The rest of apax (energy/forces in ``apax/nn/models.py`` and the
    ``nll_loss``/``crps_loss`` consumers in ``apax/train/loss.py``) treats the
    ``<name>_uncertainty`` key as a standard deviation (sigma), not a variance.
    This regression test pins ``PropertyHead`` to that convention using the
    Bessel-corrected (``1/(n_ens - 1)``) estimator.
    """
    g, R, dr_vec, Z, idx, box = setup_data

    n_ens = 10
    property_head = PropertyHead(
        pname="property",
        readout=AtomisticReadout(n_shallow_ensemble=n_ens),
        mode="l0",
        apply_mask=False,
        aggregation="mean",
    )
    params = property_head.init(jax.random.PRNGKey(0), g, R, dr_vec, Z, idx, box)
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)

    # Reproduce the per-ensemble-member predictions (aggregation="mean") so we
    # can compute the expected std independently of the head's reduction.
    readout_params = {"params": params["params"]["readout"]}
    h = jax.vmap(lambda x: property_head.readout.apply(readout_params, x))(
        g
    )  # (n_atoms, n_ens)
    scale = params["params"]["scale_per_element"]
    shift = params["params"]["shift_per_element"]
    p_i = h * scale[Z] + shift[Z]  # (n_atoms, n_ens)
    members = jnp.mean(p_i, axis=0)  # (n_ens,), aggregation="mean"

    mean = jnp.mean(members)
    variance = (1 / (n_ens - 1)) * jnp.sum((mean - members) ** 2)
    expected_std = jnp.sqrt(variance)

    uncertainty = output["property_uncertainty"]
    # Must equal the std, NOT the variance.
    assert jnp.allclose(uncertainty, expected_std)
    assert not jnp.allclose(uncertainty, variance)
