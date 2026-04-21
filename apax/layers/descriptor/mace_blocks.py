"""Building blocks for :class:`~apax.layers.descriptor.mace.MaceRepresentation`.

Each block is a Flax linen ``nn.Module`` (or a pure function for stateless
utilities) and is independently testable. Layer order inside
``MaceRepresentation`` is::

    LinearNodeEmbedding → N× (InteractionBlock → ProductBlock) → concat scalar features

Equivariant primitives:
- Irreps / IrrepsArray / tensor products: ``e3nn_jax``
- Symmetric contraction: ``cuequivariance`` / ``cuequivariance_jax``

``cuequivariance-jax`` has a pure-JAX path so blocks work on CPU; CUDA kernels
are an optional acceleration when ``use_cueq=True`` and GPUs are present.
"""

from __future__ import annotations

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import linen as nn


def assemble_edge_features(dr_vec, r_max, num_bessel, num_poly_cutoff, max_ell):
    """Compute per-edge radial × cutoff basis and spherical harmonics.

    Parameters
    ----------
    dr_vec : Array, shape (n_edges, 3)
        Relative position vectors ``r_j - r_i`` for each neighbor pair.
    r_max : float
        Interaction cutoff in the same units as ``dr_vec``.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_poly_cutoff : int
        Polynomial order of the smooth envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree (inclusive). Output irreps are
        ``spherical_harmonics(max_ell)`` which is ``0e + 1o + ... + max_ell(e|o)``.

    Returns
    -------
    radial : Array, shape (n_edges, num_bessel)
        Bessel basis multiplied element-wise by the polynomial cutoff envelope.
        Zero for ``|r| >= r_max``.
    sph : e3nn_jax.IrrepsArray
        Spherical harmonics of ``dr_vec``, normalized on the unit sphere with
        the ``"component"`` normalization that MACE uses.

    Notes
    -----
    The dtype of both outputs matches the dtype of ``dr_vec``.
    """
    from apax.layers.descriptor.basis_functions import BesselBasis, PolynomialCutoff

    dtype = dr_vec.dtype
    r_ij = jnp.linalg.norm(dr_vec, axis=-1)

    bessel_module = BesselBasis(n_basis=num_bessel, r_max=r_max, dtype=dtype)
    bessel = bessel_module.apply({}, r_ij)

    cutoff_module = PolynomialCutoff(p=num_poly_cutoff, r_max=r_max)
    cutoff = cutoff_module.apply({}, r_ij)

    radial = (bessel * cutoff[..., None]).astype(dtype)

    irreps = e3nn.Irreps.spherical_harmonics(max_ell)
    sph = e3nn.spherical_harmonics(
        irreps,
        dr_vec,
        normalize=True,
        normalization="component",
    )
    return radial, sph


class LinearNodeEmbedding(nn.Module):
    """One-hot element embedding followed by an irreps-linear.

    Produces node features in ``irreps_out``. Initial features are pure
    scalars (parity even), so ``irreps_out`` must contain only ``0e`` components.
    """

    num_elements: int
    irreps_out: str  # must be scalar irreps (e.g. "128x0e")

    @nn.compact
    def __call__(self, Z):
        irreps = e3nn.Irreps(self.irreps_out).filter("0e")
        one_hot = jax.nn.one_hot(Z, self.num_elements)  # (n, E)
        # Linear projection E -> irreps.dim, wrapped as IrrepsArray
        w = self.param(
            "weight",
            nn.initializers.normal(stddev=1.0 / jnp.sqrt(self.num_elements)),
            (self.num_elements, irreps.dim),
        )
        feats = one_hot @ w  # (n, irreps.dim)
        return e3nn.IrrepsArray(irreps, feats)


# InteractionBlock (RealAgnosticResidual) — port notes
# Inputs:  node_feats [n_atoms, irreps_in], edge_attrs (sph) [n_edges, Ylm],
#          edge_feats (radial) [n_edges, n_bessel], i (receivers), j (senders), pair_mask
# Layout:
#   1. source_linear:    node_feats[j] -> irreps_in (e3nn Linear)
#   2. conv_tp:          source @ edge_attrs via FullyConnectedTensorProduct
#                        with per-edge MLP weights from radial features
#   3. scatter_sum:      aggregate messages into receiver index i
#   4. target_linear:    aggregated -> irreps_out (e3nn Linear)
#   5. residual:         output + skip connection from original node_feats
# Output: new node_feats [n_atoms, irreps_out]
class InteractionBlock(nn.Module):
    """MACE interaction + residual.

    RealAgnosticResidual variant: one tensor product between node features and
    edge (spherical-harmonic) attributes, weighted by a radial MLP, aggregated
    into receivers via scatter-sum, plus a skip connection.

    Parameters
    ----------
    irreps_out : str
        Target irreps for node features after this block.
    interaction_cls : str
        Which interaction variant; for now only RealAgnosticResidual is
        implemented. Other variants raise NotImplementedError.
    radial_mlp : tuple[int, ...]
        Hidden layer widths for the radial MLP that gates tensor-product
        channels. Defaults to ``(64, 64, 64)``.
    """

    irreps_out: str
    interaction_cls: str = "RealAgnosticResidual"
    radial_mlp: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, node_feats, edge_attrs, edge_feats, receivers, senders):
        if self.interaction_cls != "RealAgnosticResidual":
            raise NotImplementedError(
                f"Interaction variant {self.interaction_cls!r} not yet implemented; "
                "only 'RealAgnosticResidual' is supported at this phase."
            )
        irreps_in = node_feats.irreps
        irreps_out = e3nn.Irreps(self.irreps_out)

        # 1. Linear pre-mix
        x = e3nn.flax.Linear(irreps_in, name="linear_up")(node_feats)

        # 2. Gather source node features at senders
        x_j = x[senders]

        # 3. Tensor product with edge spherical harmonics
        #    Output irreps = full tensor-square subset reachable in irreps_out
        tp = e3nn.tensor_product(x_j, edge_attrs, filter_ir_out=irreps_out)

        # 4. Radial MLP producing a scalar per TP path per edge
        n_paths = tp.irreps.num_irreps
        mlp_widths = (*self.radial_mlp, n_paths)
        weights = e3nn.flax.MultiLayerPerceptron(
            list(mlp_widths), act=jax.nn.silu, name="radial_mlp"
        )(edge_feats)
        weighted = tp * weights                                 # broadcast-safe

        # 5. Scatter-sum into receivers
        out = e3nn.scatter_sum(weighted, dst=receivers, output_size=node_feats.shape[0])

        # 6. Post-mix and residual
        out = e3nn.flax.Linear(irreps_out, name="linear_down")(out)
        skip = e3nn.flax.Linear(irreps_out, name="skip_linear")(node_feats)
        return out + skip
