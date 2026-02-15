from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from apax.md.vesin_neighborlist import VesinNeighborListWrapper


@pytest.fixture
def sample_atoms():
    atoms = Atoms('H2O', positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]], cell=[10, 10, 10], pbc=False)
    return atoms

@pytest.fixture
def vesin_wrapper():
    # Adding a skin value that is larger than the changes in the test cases
    # to test the 'no recomputation' scenario, and smaller for recomputation.
    return VesinNeighborListWrapper(cutoff=2.0, skin=0.1, padding_factor=1.5)

def test_initial_update(vesin_wrapper, sample_atoms):
    with patch('apax.md.vesin_neighborlist.NeighborList') as mock_nl_class:
        mock_nl_instance = MagicMock()
        mock_nl_class.return_value = mock_nl_instance

        # Simulate compute returning some data
        mock_nl_instance.compute.return_value = (
            np.array([0, 1]),  # idx_i
            np.array([1, 0]),  # idx_j
            np.array([[0,0,0], [0,0,0]]) # offsets
        )

        idxs_i, idxs_j, offsets = vesin_wrapper.update(sample_atoms)

        mock_nl_class.assert_called_once_with(cutoff=vesin_wrapper.cutoff, full_list=True)
        mock_nl_instance.compute.assert_called_once()

        # Check if data is padded and has correct type
        assert idxs_i.dtype == np.int32
        assert idxs_j.dtype == np.int32
        assert offsets.dtype == np.float64
        assert len(idxs_i) >= 2 # Should be padded
        assert len(idxs_j) >= 2
        assert len(offsets) >= 2

        assert np.all(vesin_wrapper._last_positions == sample_atoms.positions)
        assert np.all(vesin_wrapper._last_cell == sample_atoms.cell.array)

def test_no_recomputation_on_no_change(vesin_wrapper, sample_atoms):
    with patch('apax.md.vesin_neighborlist.NeighborList') as mock_nl_class:
        mock_nl_instance = MagicMock()
        mock_nl_class.return_value = mock_nl_instance

        # Initial call
        mock_nl_instance.compute.return_value = (
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([[0,0,0], [0,0,0]])
        )
        vesin_wrapper.update(sample_atoms)
        mock_nl_class.assert_called_once()
        mock_nl_instance.compute.assert_called_once()

        # Second call with no changes
        vesin_wrapper.update(sample_atoms)
        mock_nl_class.assert_called_once() # Should not be called again
        mock_nl_instance.compute.assert_called_once() # Should not be called again

def test_recomputation_on_position_change(vesin_wrapper, sample_atoms):
    with patch('apax.md.vesin_neighborlist.NeighborList') as mock_nl_class:
        mock_nl_instance = MagicMock()
        mock_nl_class.return_value = mock_nl_instance

        mock_nl_instance.compute.return_value = (
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([[0,0,0], [0,0,0]])
        )
        vesin_wrapper.update(sample_atoms)

        sample_atoms.positions[0, 0] += 0.4

        # Ensure compute is called again
        mock_nl_instance.compute.reset_mock()
        vesin_wrapper.update(sample_atoms)
        mock_nl_instance.compute.assert_called_once()

def test_recomputation_on_cell_change(vesin_wrapper, sample_atoms):
    with patch('apax.md.vesin_neighborlist.NeighborList') as mock_nl_class:
        mock_nl_instance = MagicMock()
        mock_nl_class.return_value = mock_nl_instance

        mock_nl_instance.compute.return_value = (
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([[0,0,0], [0,0,0]])
        )
        vesin_wrapper.update(sample_atoms)

        # Change cell (any change should trigger recomputation)
        sample_atoms.cell[0, 0] += 0.3

        # Ensure compute is called again
        mock_nl_instance.compute.reset_mock()
        vesin_wrapper.update(sample_atoms)
        mock_nl_instance.compute.assert_called_once()

def test_padding_and_overflow(vesin_wrapper, sample_atoms):
    with patch('apax.md.vesin_neighborlist.NeighborList') as mock_nl_class:
        mock_nl_instance = MagicMock()
        mock_nl_class.return_value = mock_nl_instance

        # Initial smaller NL
        mock_nl_instance.compute.return_value = (
            np.array([0]),
            np.array([1]),
            np.array([[0,0,0]])
        )
        vesin_wrapper.update(sample_atoms)
        initial_padded_length = vesin_wrapper._padded_length
        # int(1 * 1.5) = 1
        assert initial_padded_length == 1

        # Simulate a larger NL that causes overflow
        mock_nl_instance.compute.return_value = (
            np.array([0, 1, 2, 3]),
            np.array([1, 0, 0, 1]),
            np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
        )
        mock_nl_instance.compute.reset_mock()

        # Trigger recomputation by changing positions
        sample_atoms.positions[0, 0] += 0.5

        vesin_wrapper.update(sample_atoms)
        mock_nl_instance.compute.assert_called_once() # Should recompute

        # New padded length should be based on the new, larger NL size
        # int(4 * 1.5) = 6
        assert vesin_wrapper._padded_length == 6
        assert vesin_wrapper._padded_length > initial_padded_length

        # Check if returned NL is indeed padded to the new length
        idxs_i, _, _ = vesin_wrapper._nl_data
        assert len(idxs_i) == vesin_wrapper._padded_length
