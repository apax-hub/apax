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

    # Test mode `symmetric_traceless_l2`
    property_head = PropertyHead(
        pname="property", mode="symmetric_traceless_l2", apply_mask=True
    )
    output = property_head.apply(params, g, R, dr_vec, Z, idx, box)
    assert "property" in output.keys()
    assert output["property"].shape == (5, 3, 3)

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
