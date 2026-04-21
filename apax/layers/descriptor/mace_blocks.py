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
import jax.numpy as jnp


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
