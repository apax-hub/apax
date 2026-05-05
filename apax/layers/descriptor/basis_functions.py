from typing import Any, Literal, Optional

import e3nn_jax as e3nn
import einops
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from ase import data

from apax.layers.initializers import uniform_range
from apax.utils.convert import str_to_dtype
from apax.utils.parity_debug import is_parity_debug_enabled


class GaussianBasis(nn.Module):
    n_basis: int = 7
    r_min: float = 0.5
    r_max: float = 6.0
    dtype: Any = jnp.float32
    spacing: Literal["linear", "exponential"] = "linear"

    def setup(self):
        dtype = str_to_dtype(self.dtype)

        if self.spacing == "linear":
            self.betta = self.n_basis**2 / self.r_max**2
            self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
            shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
                self.n_basis
            )
            self.dr_scaling_fn = lambda dr: dr
        elif self.spacing == "exponential":
            self.betta = (
                2.0 / self.n_basis * (jnp.exp(-self.r_min) - jnp.exp(-self.r_max))
            ) ** -2
            self.rad_norm = 1.0
            shifts = np.linspace(
                np.exp(-self.r_max), np.exp(-self.r_min), num=self.n_basis, endpoint=True
            )
            self.dr_scaling_fn = lambda dr: jnp.exp(-dr)
        else:
            raise NotImplementedError(
                f"spacing {self.spacing} has not been implemented. Available options are: ['linear', 'exponential']"
            )

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - self.dr_scaling_fn(dr)

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis

        return basis


class BesselBasis(nn.Module):
    """Non-orthogonalized basis functions of Kocer
    https://doi.org/10.1063/1.5086167
    """

    n_basis: int = 7
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.n = jnp.arange(self.n_basis, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        a = (-1) ** self.n * (jnp.sqrt(2) * np.pi / (self.r_max ** (3 / 2)))
        b = (self.n + 1) * (self.n + 2) / jnp.sqrt((self.n + 1) ** 2 * (self.n + 2) ** 2)
        s1 = jnp.sinc((self.n + 1) * dr / self.r_max)
        s2 = jnp.sinc((self.n + 2) * dr / self.r_max)
        basis = a * b * (s1 + s2)
        return basis


class MaceBesselBasis(nn.Module):
    """Bessel basis used by torch-mace foundation models.

    Implements the radial basis from
    ``mace.modules.radial.BesselBasis`` (the form due to Kocer et al. as
    adapted by MACE):

    .. math::

        b_n(r) = \\sqrt{\\frac{2}{r_\\mathrm{max}}} \\,\\frac{\\sin(n \\pi r / r_\\mathrm{max})}{r}, \\quad n = 1, \\dots, N

    Distinct from :class:`BesselBasis` (Kocer's symmetrised form), which apax
    keeps for legacy users; use this class to reproduce torch-mace exactly.

    Parameters
    ----------
    n_basis : int
        Number of basis functions (``num_basis`` in torch-mace).
    r_max : float
        Cutoff distance.
    dtype : Any
        Floating-point dtype.
    """

    n_basis: int = 8
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        # bessel_weights = pi/r_max * [1, 2, ..., n_basis]
        self.bessel_weights = jnp.asarray(
            np.pi / self.r_max * np.arange(1, self.n_basis + 1, dtype=np.float64),
            dtype=dtype,
        )
        self.prefactor = jnp.asarray(np.sqrt(2.0 / self.r_max), dtype=dtype)

    def __call__(self, dr):
        x = einops.repeat(dr, "neighbors -> neighbors 1")
        numerator = jnp.sin(self.bessel_weights * x)
        return self.prefactor * (numerator / x)


class PolynomialCutoff(nn.Module):
    """MACE-style polynomial envelope cutoff.

    Implements the smooth cutoff function from Klicpera et al. 2020, used by MACE:

        f(r) = 1 - ((p + 1)(p + 2) / 2) x^p
                 + p(p + 2) x^(p+1)
                 - (p(p + 1) / 2) x^(p+2)            for r <= r_max
        f(r) = 0                                       for r >  r_max
    with x = r / r_max.

    Parameters
    ----------
    p : int, default 5
        Polynomial order; controls smoothness at r_max.
    r_max : float, default 6.0
        Distance at which the cutoff becomes 0.
    """

    p: int = 5
    r_max: float = 6.0

    def __call__(self, r):
        x = r / self.r_max
        p = self.p
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
            + p * (p + 2.0) * x ** (p + 1)
            - (p * (p + 1.0) / 2.0) * x ** (p + 2)
        )
        return jnp.where(r <= self.r_max, envelope, 0.0)


def cosine_cutoff(dr, dr_max: float):
    dr_clipped = jnp.clip(dr, max=dr_max)
    cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / dr_max) + 1.0)
    return cos_cutoff


class RadialFunction(nn.Module):
    n_radial: int = 5
    basis_fn: nn.Module = GaussianBasis()
    n_species: int = 119
    emb_init: str = "uniform"
    use_embed_norm: bool = True
    one_sided_dist: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.r_max = self.basis_fn.r_max
        self.embed_norm = jnp.array(1.0 / np.sqrt(self.basis_fn.n_basis), dtype=dtype)
        if self.one_sided_dist:
            lower_bound = 0.0
        else:
            lower_bound = -1.0

        if self.emb_init is not None:
            self._n_radial = self.n_radial
            if self.emb_init == "uniform":
                emb_initializer = uniform_range(lower_bound, 1.0, dtype=dtype)
                self.embeddings = self.param(
                    "atomic_type_embedding",
                    emb_initializer,
                    (
                        self.n_species,
                        self.n_species,
                        self.n_radial,
                        self.basis_fn.n_basis,
                    ),
                    dtype,
                )
            else:
                raise ValueError(
                    "Currently only uniformly initialized embeddings and no embeddings"
                    " are implemented."
                )
        else:
            self._n_radial = self.basis_fn.n_basis

    def __call__(self, dr, Z_i, Z_j):
        dtype = str_to_dtype(self.dtype)
        dr = dr.astype(dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        if self.emb_init is None:
            radial_function = basis
        else:
            # coeffs shape: n_neighbors x n_radialx n_basis
            species_pair_coeffs = self.embeddings[
                Z_j, Z_i, ...
            ]  # reverse convention to match original
            if self.use_embed_norm:
                species_pair_coeffs = self.embed_norm * species_pair_coeffs

            # radial shape: neighbors x n_radial
            radial_function = einops.einsum(
                species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
            )

        # shape: neighbors
        cos_cutoff = cosine_cutoff(dr, self.r_max)
        cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")

        radial_function = radial_function * cutoff

        assert radial_function.dtype == dtype

        return radial_function


class AgnesiTransform(nn.Module):
    """Faithful port of ``mace.modules.radial.AgnesiTransform``.

    Per-pair length transform driven by element-pair covalent radii.

    .. math::

        r_0 &= 0.5 \\, (\\mathrm{cov}[Z_u] + \\mathrm{cov}[Z_v]) \\\\
        T(r) &= \\frac{1}{1 + a \\, (r/r_0)^q / (1 + (r/r_0)^{q-p})}

    All three scalars (``a``, ``q``, ``p``) are stored as buffers when
    ``trainable=False`` (foundation-model regime) and as parameters when
    ``trainable=True`` (so a future fresh-training run can fine-tune them).
    The ``covalent_radii`` table is always a buffer.

    Parameters
    ----------
    a_init : float, default = 1.0805
        Initial value for ``a``.
    q_init : float, default = 0.9183
        Initial value for ``q``.
    p_init : float, default = 4.5791
        Initial value for ``p``.
    trainable : bool, default = False
        If ``True``, ``a``/``q``/``p`` become trainable parameters; otherwise
        they stay as fixed buffers.

    Notes
    -----
    The ``idx`` argument follows apax's convention
    (``idx[0]=receivers``, ``idx[1]=senders``), which is the **opposite** of
    torch-mace's ``edge_index`` ordering (sender first, receiver second).
    The transform is symmetric in ``Z_u``/``Z_v`` (their sum appears in
    ``r_0``), so this convention difference does not change values.

    The default values for ``a``/``q``/``p`` match the buffers shipped with
    the MACE-MPA-0 and MatPES-r2scan-omat-ft foundation models.
    """

    a_init: float = 1.0805
    q_init: float = 0.9183
    p_init: float = 4.5791
    trainable: bool = False

    def setup(self):
        self.covalent_radii = self.variable(
            "buffers",
            "covalent_radii",
            lambda: jnp.asarray(data.covalent_radii, dtype=jnp.float64),
        )
        if self.trainable:
            self.a = self.param(
                "a", lambda rng: jnp.asarray(self.a_init, dtype=jnp.float64)
            )
            self.q = self.param(
                "q", lambda rng: jnp.asarray(self.q_init, dtype=jnp.float64)
            )
            self.p = self.param(
                "p", lambda rng: jnp.asarray(self.p_init, dtype=jnp.float64)
            )
        else:
            self.a = self.variable(
                "buffers", "a", lambda: jnp.asarray(self.a_init, dtype=jnp.float64)
            )
            self.q = self.variable(
                "buffers", "q", lambda: jnp.asarray(self.q_init, dtype=jnp.float64)
            )
            self.p = self.variable(
                "buffers", "p", lambda: jnp.asarray(self.p_init, dtype=jnp.float64)
            )

    def _scalar(self, x):
        """Resolve a scalar parameter (Variable for buffers, Array for params)."""
        return x.value if hasattr(x, "value") else x

    def __call__(self, r, Z, idx):
        """Apply the Agnesi transform to per-edge distances.

        Parameters
        ----------
        r : jnp.ndarray
            Per-edge distances of shape ``(n_edges,)``.
        Z : jnp.ndarray
            Atomic numbers of shape ``(n_atoms,)``.
        idx : jnp.ndarray
            Edge index array of shape ``(2, n_edges)`` with
            ``idx[0]=receivers``, ``idx[1]=senders``.

        Returns
        -------
        jnp.ndarray
            Transformed distances of shape ``(n_edges,)``.
        """
        i, j = idx[0], idx[1]
        Z_u, Z_v = Z[i], Z[j]
        cov = self.covalent_radii.value
        # Clip r0 away from zero so masked / padding edges (which can carry
        # ``Z=0`` and dr=0) don't trigger ``0**(q-p)=0**negative=inf`` and
        # poison gradients via ``inf - inf`` style cancellations. The clip
        # only fires on the masked branch; physical edges always have
        # ``r0 >= 0.5*(min covalent_radii)`` which is well above the floor.
        r0 = jnp.clip(0.5 * (cov[Z_u] + cov[Z_v]), min=0.02)
        # Clip ``r`` similarly so ``x = r/r0`` is bounded away from zero;
        # physical edges have ``r > 0`` but jax.where-style masks may carry
        # ``r = 0`` past the transform.
        r_safe = jnp.clip(r, min=0.02)
        a = self._scalar(self.a)
        q = self._scalar(self.q)
        p = self._scalar(self.p)
        x = r_safe / r0
        denom = 1.0 + a * (x**q) / (1.0 + (x ** (q - p)))
        return 1.0 / denom


class MaceRadialEmbedding(nn.Module):
    """Composable radial embedding: bessel x cutoff with optional transform.

    Mirrors torch-mace's :class:`RadialEmbeddingBlock`. The optional
    ``distance_transform`` is applied between ``cutoff_fn`` and ``bessel_fn``
    and consumes per-edge atomic numbers. The forward returns
    ``(radial, sph)`` where::

        radial = bessel(T(r)) * cutoff(r)        # transform configured
        radial = bessel(r)    * cutoff(r)        # no transform

    Critically the cutoff is computed on the **original** ``r``; only the
    bessel basis sees the transformed value. Reordering breaks parity with
    torch.

    Parameters
    ----------
    r_max : float
        Interaction cutoff in the same units as ``dr_vec``.
    num_bessel : int
        Number of Bessel radial basis functions.
    num_polynomial_cutoff : int
        Polynomial order of the smooth envelope cutoff.
    max_ell : int
        Maximum spherical-harmonic degree (inclusive).
    distance_transform : Any, optional
        Either ``None`` (default) or a Linen ``nn.Module`` instance with
        signature ``__call__(r, Z, idx) -> r_transformed``. The runtime type
        is dynamic; the field is typed as :class:`typing.Any` so any
        future transform module can be attached without widening the type.
    """

    r_max: float
    num_bessel: int
    num_polynomial_cutoff: int
    max_ell: int
    distance_transform: Optional[Any] = None

    @nn.compact
    def __call__(self, dr_vec, Z, idx):
        """Compute the per-edge radial features and spherical harmonics.

        Parameters
        ----------
        dr_vec : jnp.ndarray
            Edge displacement vectors of shape ``(n_edges, 3)``.
        Z : jnp.ndarray
            Atomic numbers of shape ``(n_atoms,)``.
        idx : jnp.ndarray
            Edge index array of shape ``(2, n_edges)`` with
            ``idx[0]=receivers``, ``idx[1]=senders``.

        Returns
        -------
        radial : jnp.ndarray
            Bessel basis multiplied element-wise by the polynomial cutoff
            envelope, shape ``(n_edges, num_bessel)``.
        sph : e3nn_jax.IrrepsArray
            Spherical harmonics of ``dr_vec`` with ``"component"``
            normalization on the unit sphere.
        """
        dtype = dr_vec.dtype
        r_ij = jnp.linalg.norm(dr_vec, axis=-1)
        cutoff = PolynomialCutoff(
            p=self.num_polynomial_cutoff, r_max=self.r_max
        )(r_ij)
        if self.distance_transform is not None:
            r_ij = self.distance_transform(r_ij, Z, idx)
        bessel = MaceBesselBasis(
            n_basis=self.num_bessel, r_max=self.r_max, dtype=dtype,
        )(r_ij)
        radial = (bessel * cutoff[..., None]).astype(dtype)
        # Gated so plain ``model.init`` doesn't sprout a ``debug`` branch.
        if is_parity_debug_enabled():
            self.sow("debug", "radial_embedding", radial)
        sph = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.max_ell),
            dr_vec,
            normalize=True,
            normalization="component",
        )
        return radial, sph
