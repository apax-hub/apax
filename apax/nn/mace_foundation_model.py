"""Full-energy MACE module for parity tests and zero-shot inference.

Unlike the standard :class:`apax.nn.models.EnergyModel` +
:class:`MaceRepresentation` + :class:`AtomisticReadout` stack used for
fine-tuning, this module includes MACE's per-layer internal readouts and
per-element atomic-energy reference so it reproduces the full upstream
``ScaleShiftMACE`` forward pass.

Fine-tuning, shallow ensemble, property heads, MD, ASE — none of these go
through this module. It exists solely so the parity test can verify the
converter.

Notes
-----
The ``__call__`` body is deferred until the torch parameter mapping
(``apax.transfer_learning.mace_foundation._map_state_to_pytree``) is
finalized; see plan P3.2 Step 4 and P3.4 Step 3.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class MaceFoundationEnergyModel(nn.Module):
    """Full-energy MACE module used only for parity and zero-shot inference.

    Parameters
    ----------
    r_max : float
        Interaction cutoff in Angstrom.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_polynomial_cutoff : int
        Polynomial order of the envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree used for edge features.
    hidden_irreps : str
        e3nn irreps string for node features, e.g. ``"128x0e + 128x1o"``.
    num_interactions : int
        Number of (interaction, product) layer pairs.
    correlation : int
        Symmetric-contraction correlation order.
    interaction_cls : str
        Which MACE interaction block variant to use.
    num_elements : int
        Size of the chemical-element embedding table.
    atomic_energies : Any
        Per-element reference energies ``E0`` of shape ``(num_elements,)``;
        broadcast over atoms by indexing with ``Z``.
    use_cueq : bool
        If True, use cuequivariance-jax kernels where available.
    """

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    max_ell: int
    hidden_irreps: str
    num_interactions: int
    correlation: int
    interaction_cls: str
    num_elements: int
    atomic_energies: Any  # jnp.ndarray or list of floats
    use_cueq: bool = False

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        """Forward pass — deferred.

        Parameters
        ----------
        dr_vec : jnp.ndarray, shape (n_edges, 3)
            Pair displacement vectors.
        Z : jnp.ndarray, shape (n_atoms,)
            Atomic numbers.
        idx : jnp.ndarray, shape (2, n_edges)
            Edge index array with rows ``(receivers, senders)``.

        Returns
        -------
        jnp.ndarray
            Per-atom energy, shape ``(n_atoms,)``.

        Raises
        ------
        NotImplementedError
            Body pending until ``MaceRepresentation`` exposes per-layer
            node features and the torch-mapping pass is complete; see
            plan P3.4 Step 3.
        """
        raise NotImplementedError(
            "MaceFoundationEnergyModel.__call__ pending; see plan P3.4 Step 3. "
            "Requires per-layer node-feat exposure on MaceRepresentation and a "
            "complete _map_state_to_pytree body (P3.2 Step 4)."
        )
