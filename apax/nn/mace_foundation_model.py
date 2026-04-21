"""Full-energy MACE module for parity tests and zero-shot inference.

Unlike the standard :class:`apax.nn.models.EnergyModel` +
:class:`MaceRepresentation` + :class:`AtomisticReadout` stack used for
fine-tuning, this module includes MACE's per-layer internal readouts and
per-element atomic-energy reference so it reproduces the full upstream
``ScaleShiftMACE`` forward pass.

Fine-tuning, shallow ensemble, property heads, MD, ASE — none of these go
through this module. It exists solely so the parity test can verify the
converter.
"""
from __future__ import annotations

from typing import Any

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn


class MaceFoundationEnergyModel(nn.Module):
    """Full-energy MACE module used only for parity and zero-shot inference.

    Reproduces the forward pass of torch-mace's ``ScaleShiftMACE`` for the
    small MACE-MP-0 foundation model (no pair repulsion, no ZBL, single head,
    scalar-only ``hidden_irreps``).

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
        Final node-feature irreps after each product block (e.g.
        ``"128x0e"``).
    num_interactions : int
        Number of (interaction, product) layer pairs.
    correlation : int
        Symmetric-contraction correlation order.
    interaction_cls : str
        Which MACE interaction block variant to use.
    num_elements : int
        Size of the chemical-element embedding table.
    atomic_energies : list of float or tuple of float
        Per-element reference energies ``E0`` of shape ``(num_elements,)``.
    atomic_numbers : list of int or tuple of int
        Chemical-species table; ``Z`` inputs are mapped through this table to
        element indices before indexing into ``atomic_energies`` and the
        embedding weight matrices.
    avg_num_neighbors : float
        Convolution normalisation from the torch model.
    MLP_irreps : str
        Irreps of the non-linear readout hidden layer (``"16x0e"`` for small).
    interaction_target_irreps : str
        Irreps target for the interaction output (``"128x0e+128x1o+128x2e+128x3o"``
        for small MP-0). When ``None``, defaults to the tensor-product reach
        of ``hidden_irreps × Ylm(max_ell)``.
    use_cueq : bool
        If True, use cuequivariance-jax CUDA kernels where available.
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
    atomic_energies: Any
    atomic_numbers: Any
    avg_num_neighbors: float = 1.0
    MLP_irreps: str = "16x0e"
    interaction_target_irreps: str | None = None
    use_cueq: bool = False

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        """Compute per-atom energy in eV.

        Parameters
        ----------
        dr_vec : jnp.ndarray, shape (n_edges, 3)
            Pair displacement vectors ``R[receiver] - R[sender]``; apax's
            ``compute_distances`` returns ``R[j] - R[i]``, which matches
            torch-mace's ``R[receiver] - R[sender]`` when ``idx[0] = receiver``
            and ``idx[1] = sender``. In apax's own convention ``idx[0] = i =
            receivers`` and ``idx[1] = j = senders`` so ``dr_vec`` already has
            the torch-mace sign.
        Z : jnp.ndarray, shape (n_atoms,)
            Atomic numbers.
        idx : jnp.ndarray, shape (2, n_edges)
            Edge index array with rows ``(receivers, senders)`` in the apax
            convention.

        Returns
        -------
        jnp.ndarray
            Per-atom energy, shape ``(n_atoms,)``.
        """
        from apax.layers.descriptor.mace_blocks import (
            InteractionBlock,
            LinearNodeEmbedding,
            LinearReadoutBlock,
            NonLinearReadoutBlock,
            ProductBlock,
            ScaleShift,
            assemble_edge_features,
        )

        # Map torch's edge_index convention (sender, receiver) into apax's
        # (receivers, senders). dr_vec was computed as R[j] - R[i] =
        # R[sender] - R[receiver] in the apax convention, which is the same
        # vector that torch-mace uses (R[receiver] - R[sender] in its own
        # convention: note that in the MACE forward `edge_index[0]=sender`,
        # `edge_index[1]=receiver`, and vectors = R[receiver] - R[sender]).
        # To match torch we also need to negate the vector: apax computes
        # R[j] - R[i] = R[senders] - R[receivers] = -(R[receivers] - R[senders])
        # = -(torch's vectors). Odd-l spherical harmonics flip sign under
        # r -> -r, so we negate dr_vec before computing edge features.
        receivers = idx[0]
        senders = idx[1]
        dr_vec_torch = -dr_vec  # align sign with torch convention

        radial, sph = assemble_edge_features(
            dr_vec_torch,
            self.r_max,
            self.num_bessel,
            self.num_polynomial_cutoff,
            self.max_ell,
        )

        # Scalar-only subset of hidden irreps (first product output is
        # scalar-only for the small foundation model).
        scalar_init = _scalar_irreps_only(self.hidden_irreps)

        # Determine the interaction target irreps (foundation models use the
        # full 128x0e+128x1o+128x2e+128x3o irreps inside the interaction,
        # before the product block projects back to scalar-only).
        target_irreps = (
            self.interaction_target_irreps
            if self.interaction_target_irreps is not None
            else _default_target_irreps(self.hidden_irreps, self.max_ell)
        )

        # Map atomic numbers Z -> index in the foundation model's atomic_numbers
        # table (not Z directly!). Use a LUT with max Z = 128.
        atomic_numbers_tbl = jnp.asarray(list(self.atomic_numbers), dtype=jnp.int32)
        max_z = 128
        lut = jnp.full((max_z,), -1, dtype=jnp.int32)
        lut = lut.at[atomic_numbers_tbl].set(
            jnp.arange(len(atomic_numbers_tbl), dtype=jnp.int32)
        )
        Z_idx = lut[Z.astype(jnp.int32)]  # (n_atoms,)

        # Initial node features: scalars only, per-element embedding.
        node_feats = LinearNodeEmbedding(
            num_elements=self.num_elements,
            irreps_out=scalar_init,
            name="node_embedding",
        )(Z_idx)

        per_layer_energies = []
        for k in range(self.num_interactions):
            node_feats = InteractionBlock(
                irreps_out=target_irreps,
                target_irreps=target_irreps,
                interaction_cls=self.interaction_cls,
                foundation_mode=True,
                num_elements=self.num_elements,
                hidden_irreps_final=self.hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                name=f"interaction_{k}",
            )(node_feats, sph, radial, receivers, senders, Z=Z_idx)

            node_feats = ProductBlock(
                hidden_irreps=self.hidden_irreps,
                input_irreps=target_irreps,
                correlation=self.correlation,
                num_elements=self.num_elements,
                post_linear=True,
                use_cueq=self.use_cueq,
                name=f"product_{k}",
            )(node_feats, Z_idx)

            if k < self.num_interactions - 1:
                e_k = LinearReadoutBlock(
                    irreps_in=self.hidden_irreps, name=f"readout_{k}"
                )(node_feats)
            else:
                e_k = NonLinearReadoutBlock(
                    irreps_in=self.hidden_irreps,
                    MLP_irreps=self.MLP_irreps,
                    name=f"readout_{k}",
                )(node_feats)
            per_layer_energies.append(e_k)

        node_inter_es = per_layer_energies[0]
        for e_k in per_layer_energies[1:]:
            node_inter_es = node_inter_es + e_k

        # Scale & shift applied to the summed per-atom energy.
        node_inter_es = ScaleShift(name="scale_shift")(node_inter_es)

        # Reference atomic energies (per-element E0).
        atomic_energies = jnp.asarray(list(self.atomic_energies))
        e_ref = atomic_energies[Z_idx]

        return e_ref + node_inter_es


def _scalar_irreps_only(irreps_str: str) -> str:
    """Return the ``0e`` subset of an irreps string.

    Parameters
    ----------
    irreps_str : str
        Full irreps string, e.g. ``"128x0e + 128x1o"``.

    Returns
    -------
    str
        Only the ``0e`` component(s), e.g. ``"128x0e"``.
    """
    parts = [p.strip() for p in irreps_str.split("+")]
    scalar = [p for p in parts if p.endswith("x0e")]
    if not scalar:
        raise ValueError(f"No 0e component in irreps {irreps_str!r}")
    return " + ".join(scalar)


def _default_target_irreps(hidden_irreps: str, max_ell: int) -> str:
    """Return the full ``mul x (0e + 1o + 2e + ...)`` irreps used by interactions.

    The small MP-0 model feeds ``hidden_irreps = "128x0e"`` into the final
    product, but the interaction block internally operates on the full tensor
    product reach of ``hidden × Ylm(max_ell)``. This helper reconstructs that
    target irreps string from ``hidden_irreps`` (scalar-only) and ``max_ell``.

    Parameters
    ----------
    hidden_irreps : str
        Scalar-only hidden irreps (e.g. ``"128x0e"``).
    max_ell : int
        Maximum spherical-harmonic degree.

    Returns
    -------
    str
        Full target irreps, e.g. for ``"128x0e"`` and ``max_ell=3``:
        ``"128x0e + 128x1o + 128x2e + 128x3o"``.
    """
    hidden = e3nn.Irreps(hidden_irreps)
    mul = next(iter({m for m, _ in hidden}))
    parts = []
    for l in range(max_ell + 1):
        p = "e" if l % 2 == 0 else "o"
        parts.append(f"{mul}x{l}{p}")
    return " + ".join(parts)
