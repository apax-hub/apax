from dataclasses import field
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ase import data
from jax import vmap
from jax_md import space

from apax.layers.masking import mask_by_atom, mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.math import fp64_sum


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class EmpiricalEnergyTerm(nn.Module):
    dtype: Any = jnp.float32


class ZBLRepulsion(EmpiricalEnergyTerm):
    r_max: float = 2.0
    # Coulomb constant e^2/(4*pi*eps0) = 14.4 eV*Ang; the physical prefactor for
    # the screened Z_i*Z_j/r nuclear repulsion (=1 in Hartree/Bohr units).
    initial_rep_scale: float = 14.4
    apply_mask: bool = True

    def setup(self):
        self.distance = vmap(space.distance, 0, 0)

        coeffs = jnp.array([0.18175, 0.50986, 0.28022, 0.02817])[:, None]
        coeffs_isp = inverse_softplus(coeffs)
        rep_scale_isp = inverse_softplus(self.initial_rep_scale)

        self.a_exp = 0.23
        self.a_num = 0.46850
        self.coefficients = self.param(
            "coefficients",
            nn.initializers.constant(coeffs_isp),
            (4, 1),
        )

        self.exponents = jnp.array([3.19980, 0.94229, 0.4029, 0.20162])[:, None]

        self.rep_scale = self.param(
            "rep_scale", nn.initializers.constant(rep_scale_isp), (1,)
        )

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)
        # Z shape n_atoms

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(dtype)

        dr = jnp.clip(dr, min=0.02, max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        a_exp = self.a_exp
        a_num = self.a_num
        coefficients = jax.nn.softplus(self.coefficients)
        exponents = self.exponents
        rep_scale = jax.nn.softplus(self.rep_scale)[0]

        a_divisor = Z_i**a_exp + Z_j**a_exp
        dist = dr * a_divisor / a_num
        f = coefficients * jnp.exp(-exponents * dist)
        f = jnp.sum(f, axis=0)

        E_ij = Z_i * Z_j / dr * f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx, dr_vec)
        E = 0.5 * rep_scale * fp64_sum(E_ij)
        return E


class ExponentialRepulsion(EmpiricalEnergyTerm):
    r_max: float = 2.0
    apply_mask: bool = True

    def setup(self):
        self.distance = vmap(space.distance, 0, 0)

        radii = data.covalent_radii * 0.8
        self.rscale = self.param("rep_scale", nn.initializers.constant(radii), (119,))

        self.prefactor = self.param(
            "rep_prefactor", nn.initializers.constant(100.0), (119,)
        )

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)

        # Z shape n_atoms
        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(dtype)

        dr = jnp.clip(dr, min=0.02, max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        A_i, A_j = (
            jax.numpy.abs(self.prefactor[Z_i]),
            jax.numpy.abs(self.prefactor[Z_j]),
        )
        R_i, R_j = jax.numpy.abs(self.rscale[Z_i]), jax.numpy.abs(self.rscale[Z_j])

        f = A_i * A_j * jnp.exp(-dr * (R_i + R_j) / (R_i * R_j)) / dr**2

        E_ij = f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx, dr_vec)
        E = fp64_sum(E_ij)
        return E


class LatentEwald(EmpiricalEnergyTerm):
    """Latent Ewald summation by Cheng https://arxiv.org/abs/2408.15165
    Requires a property head which predicts 'charge' per atom.
    """

    kgrid: list[int] = field(default_factory=lambda: [2, 2, 2])
    sigma: float = 1.0
    apply_mask: bool = True
    use_property: str = "charges"

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        # Z shape n_atoms
        if self.use_property not in properties:
            raise KeyError(
                f"property '{self.use_property}' not found. Make sure to predict it in the model section"
            )

        q = properties[self.use_property]

        V = jnp.linalg.det(box)
        Lx, Ly, Lz = jnp.linalg.norm(box, axis=1)

        k_range_x = 2 * np.pi * jnp.arange(1, self.kgrid[0]) / Lx
        k_range_y = 2 * np.pi * jnp.arange(1, self.kgrid[1]) / Ly
        k_range_z = 2 * np.pi * jnp.arange(1, self.kgrid[2]) / Lz

        kx, ky, kz = jnp.meshgrid(k_range_x, k_range_y, k_range_z)
        k = jnp.reshape(jnp.stack((kx, ky, kz), axis=-1), (-1, 3))

        k2 = jnp.sum(k**2, axis=-1)

        sf_k = q * jnp.exp(1j * jnp.einsum("id,jd->ij", R, k))
        if self.apply_mask:
            sf_k = mask_by_atom(sf_k, Z)
        sf = jnp.sum(sf_k, axis=0)
        S2 = jnp.abs(sf) ** 2

        E_lr = -fp64_sum(jnp.exp(-k2 * (self.sigma**2) / 2) / k2 * S2) / V

        return E_lr


class NLHRepulsion(EmpiricalEnergyTerm):
    """Triple-exponential screened nuclear repulsion.

    V_ij(r) = 14.4 * Z_i Z_j / r * phi_ij(r),
    phi_ij(r) = sum_k a[Z_i,Z_j,k] * exp(-b[Z_i,Z_j,k] * r),  k = 1..3.

    Per-pair coefficients `a`, `b` (shape (n_species, n_species, 3)) are
    injected by the builder (dependency injection); non-trainable.

    With poly_degree K > 0 the pair energy is modulated by a learnable, strictly positive factor
    p_ij(r) = exp(sum_{k=1..K} c_ij,k t^k), t = 1 - r/r_max, c_ij = (c_i + c_j)/2 per element (init 0 = plain NLH):
    p = 1 at r_max, smooth and low-order below, and positive, so the wall can be rescaled but never turn attractive.
    """

    a: Any = None  # (n_species, n_species, 3), injected
    b: Any = None
    poly_degree: int = 0
    r_max: float = 6.0
    rep_scale: float = 14.4  # e^2/(4 pi eps0) in eV*Ang
    apply_mask: bool = True

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.distance = vmap(space.distance, 0, 0)
        self._a = jnp.asarray(self.a, dtype=dtype)  # inject at module dtype (fp32)
        self._b = jnp.asarray(self.b, dtype=dtype)
        if self.poly_degree > 0:
            self.poly = self.param("nlh_poly", nn.initializers.zeros, (self._a.shape[0], self.poly_degree), dtype)

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)
        idx_i, idx_j = idx[0], idx[1]
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        dr = self.distance(dr_vec).astype(dtype)
        dr = jnp.clip(dr, min=0.02, max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        a_ij = self._a[Z_i, Z_j]  # (neighbors, 3)
        b_ij = self._b[Z_i, Z_j]
        phi = jnp.sum(a_ij * jnp.exp(-b_ij * dr[:, None]), axis=-1)  # (neighbors,)

        E_ij = self.rep_scale * Z_i * Z_j / dr * phi * cos_cutoff
        if self.poly_degree > 0:
            c_ij = 0.5 * (self.poly[Z_i] + self.poly[Z_j])  # (neighbors, K)
            t = 1.0 - dr / self.r_max
            powers = t[:, None] ** jnp.arange(1, self.poly_degree + 1, dtype=dtype)
            E_ij = E_ij * jnp.exp(jnp.sum(c_ij * powers, axis=-1))
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx, dr_vec)
        return 0.5 * fp64_sum(E_ij)


class PairEnvelope(EmpiricalEnergyTerm):
    """Attractive pair envelope E = -X sum_pairs 0.5 (1 + cos(pi r / r_max)): -X per pair at r = 0, 0 at r_max.

    A smooth cohesive baseline under the network: with isolated-atom shifts it carries the bulk of the
    bonding energy, so the network learns a zero-centred residual. Only the amplitude X is learnable (one scalar);
    it is initialised by least squares on per-atom energies (amp_init, injected by the trainer).
    """

    r_max: float = 5.0
    trainable: bool = True
    amp_init: Any = None
    apply_mask: bool = True

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.distance = vmap(space.distance, 0, 0)
        init = (lambda *_: jnp.full((1,), self.amp_init, dtype)) if self.amp_init is not None else nn.initializers.zeros
        self.amp = self.param("envelope_amp" if self.trainable else "envelope_amp_frozen", init, (1,), dtype)

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)
        dr = jnp.clip(self.distance(dr_vec).astype(dtype), max=self.r_max)
        w = 0.5 * (1.0 + jnp.cos(np.pi * dr / self.r_max))
        if self.apply_mask:
            w = mask_by_neighbor(w, idx, dr_vec)
        return -self.amp[0] * 0.5 * fp64_sum(w)  # each pair appears twice in the neighbour list


def fit_pair_envelope(atoms_list, shift, r_max=5.0):
    """least-squares X for per-atom (E - sum_i shift[Z_i]) ~ -X sum_pairs w(r) / n."""
    from ase.neighborlist import neighbor_list

    shift = np.asarray(shift).reshape(-1)
    x, y = [], []
    for a in atoms_list:
        d = neighbor_list("d", a, r_max)
        x.append(-0.25 * np.sum(1.0 + np.cos(np.pi * d / r_max)) / len(a))
        y.append((a.get_potential_energy() - shift[a.numbers].sum()) / len(a))
    x, y = np.asarray(x), np.asarray(y)
    return float(x @ y / (x @ x))


all_corrections = {
    "zbl": ZBLRepulsion,
    "exponential": ExponentialRepulsion,
    "latent_ewald": LatentEwald,
    "nlh": NLHRepulsion,
    "pair_envelope": PairEnvelope,
}
