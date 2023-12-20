from functools import partial
from pathlib import Path
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from jax_md import partition, quantity, space
from matscipy.neighbours import neighbour_list

from apax.model import ModelBuilder
from apax.model.gmnn import EnergyDerivativeModel, EnergyModel
from apax.train.checkpoints import check_for_ensemble, restore_parameters
from apax.utils import jax_md_reduced


def maybe_vmap(apply, params, Z):
    n_models = check_for_ensemble(params)

    if n_models > 1:
        apply = jax.vmap(apply, in_axes=(0, None, None, None, None, None))

    # Maybe the partial mapping should happen at the very end of initialize
    # That way other functions can mae use of the parameter shape information
    energy_fn = partial(apply, params)
    return energy_fn


def build_energy_neighbor_fns(atoms, config, params, dr_threshold, neigbor_from_jax):
    r_max = config.model.r_max
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
    neigbor_from_jax = neighbor_calculable_with_jax(box, r_max)
    box = box.T
    displacement_fn = None
    neighbor_fn = None

    if neigbor_from_jax:
        if np.all(box < 1e-6):
            displacement_fn, _ = space.free()
        else:
            displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)

        neighbor_fn = jax_md_reduced.partition.neighbor_list(
            displacement_fn,
            box,
            config.model.r_max,
            dr_threshold,
            fractional_coordinates=True,
            disable_cell_list=True,
            format=partition.Sparse,
        )

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    builder = ModelBuilder(config.model.get_dict(), n_species=n_species)

    model = builder.build_energy_derivative_model(
        apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
    )

    energy_fn = maybe_vmap(model.apply, params, Z)

    return energy_fn, neighbor_fn



from functools import partial
import jax
import jax.numpy as jnp
# from jax_md import space
from dispax.disp import rational_damping
from dispax.data import Data
from dispax.reference import Reference
from dispax import model, ncoord

from jax import Array
import flax.linen as nn
# from jax_md import energy
from functools import wraps
f32 = jnp.float32

def multiplicative_isotropic_cutoff(fn: Callable[..., Array],
                                    r_onset: float,
                                    r_cutoff: float) -> Callable[..., Array]:
  """Takes an isotropic function and constructs a truncated function.

  Given a function `f:R -> R`, we construct a new function `f':R -> R` such
  that `f'(r) = f(r)` for `r < r_onset`, `f'(r) = 0` for `r > r_cutoff`, and
  `f(r)` is :math:`C^1` everywhere. To do this, we follow the approach outlined
  in HOOMD Blue  [#hoomd]_ (thanks to Carl Goodrich for the pointer). We
  construct a function `S(r)` such that `S(r) = 1` for `r < r_onset`,
  `S(r) = 0` for `r > r_cutoff`, and `S(r)` is :math:`C^1`. Then
  `f'(r) = S(r)f(r)`.

  Args:
    fn: A function that takes an ndarray of distances of shape `[n, m]` as well
      as varargs.
    r_onset: A float specifying the distance marking the onset of deformation.
    r_cutoff: A float specifying the cutoff distance.

  Returns:
    A new function with the same signature as fn, with the properties outlined
    above.

  .. rubric:: References
  .. [#hoomd] HOOMD Blue documentation. Accessed on 05/31/2019.
      https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
  """

  r_c = r_cutoff ** f32(2)
  r_o = r_onset ** f32(2)

  def smooth_fn(dr):
    r = dr ** f32(2)

    inner = jnp.where(dr < r_cutoff,
                     (r_c - r)**2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o)**3,
                     0)

    return jnp.where(dr < r_onset, 1, inner)

  @wraps(fn)
  def cutoff_fn(dr, *args, **kwargs):
    return smooth_fn(dr) * fn(dr, *args, **kwargs)

  return cutoff_fn

from dispax.data import covalent_rad_d3, sqrt_z_r4_over_r2, vdw_rad_d3
from dispax.reference import _load_c6, _load_cn
from ase import units 

def get_sparse_rc(idxs: Array) -> Array:
    """
    Get covalent radii for given atomic numbers.

    Parameters
    ----------
    numbers : Array
        Atomic numbers.

    Returns
    -------
    Array
        Covalent radii.
    """

    rcovi = covalent_rad_d3[idxs[0]]
    rcovj = covalent_rad_d3[idxs[1]]
    return rcovi + rcovj #jnp.sum(rcov, axis=0)#[:, jnp.newaxis]


def get_sparse_qq(idx: Array) -> Array:
    """
    Get scaling factor for C8 / C6 fraction.

    Parameters
    ----------
    numbers : Array
        Atomic numbers.

    Returns
    -------
    Array
        Scaling factor for C8 / C6 fraction.
    """
    # r4r2 = sqrt_z_r4_over_r2[numbers]
    # r4r2 = sqrt_z_r4_over_r2[idx]

    r4r2i = sqrt_z_r4_over_r2[idx[0]]
    r4r2j = sqrt_z_r4_over_r2[idx[1]]
    # print(sqrt_z_r4_over_r2.shape)
    # print(r4r2.shape)
    # quit()
    # qq = 3 * r4r2.reshape(-1, 1) * r4r2.reshape(1, -1)
    qq = 3 * r4r2i * r4r2j
    return qq


class D3Data:

    rc: Array

    qq: Array

    rvdw: Array

    __slots__ = ["rc", "qq", "rvdw"]

    def __init__(self, rc: Array, qq: Array, rvdw: Array):
        self.rc = rc
        self.qq = qq
        self.rvdw = rvdw

    @classmethod
    def from_idxs(cls, idxs: Array, dtype=jnp.float32) -> "Data":
        rc = get_sparse_rc(idxs)
        qq = get_sparse_qq(idxs)
        rvdw = 0#vdw_rad_d3[idxs]
        # print(vdw_rad_d3.shape)
        # quit()
        return cls(rc, qq, rvdw)

    # def __getitem__(self, item: Array) -> "Data":
    #     """
    #     Get a subset of the data.

    #     Parameters
    #     ----------
    #     item : Array
    #         Atomic numbers.

    #     Returns
    #     -------
    #     Data
    #         Subset of the data.
    #     """

    #     return self.__class__(
    #         self.rc[item.reshape(-1, 1), item.reshape(1, -1)],
    #         self.qq[item.reshape(-1, 1), item.reshape(1, -1)],
    #         self.rvdw[item.reshape(-1, 1), item.reshape(1, -1)],
    #     )
class D3Reference:
    """
    Reference dispersion coefficients and coordination numbers.

    Example
    -------
    >>> numbers = jnp.array([8, 1, 1, 8, 1, 6, 1, 1, 1])
    >>> numbers, species = jnp.unique(numbers, return_inverse=True)
    >>> ref = Reference.from_numbers(numbers)
    >>> ref.cn.shape, ref.c6.shape
    ((3, 5), (3, 3, 5, 5))
    >>> ref = ref[species]
    >>> ref.cn.shape, ref.c6.shape
    ((9, 5), (9, 9, 5, 5))
    """

    cn: Array
    """Reference coordination numbers"""

    c6: Array
    """Reference dispersion coefficients"""

    __slots__ = ["cn", "c6"]

    def __init__(self, cn, c6):
        self.cn = cn
        self.c6 = c6

    @classmethod
    def full(cls, dtype=jnp.float32):
        return cls(_load_cn(dtype=dtype), _load_c6(dtype=dtype))


    @classmethod
    def from_numbers(cls, numbers: Array, dtype=jnp.float32) -> "Reference":
        """
        Create a reference from a list of atomic numbers.

        Parameters
        ----------
        numbers : Array
            Atomic numbers.
        dtype : dtype, optional
            Data type of the arrays.

        Returns
        -------
        Reference
            Dispersion coefficients and coordination numbers.
        """
        return cls(_load_cn(dtype=dtype), _load_c6(dtype=dtype))[numbers]

    def __getitem__(self, item: Array) -> "Reference":
        """
        Get a subset of the reference.

        Parameters
        ----------
        item : Array
            Atomic numbers.

        Returns
        -------
        Reference
            Dispersion coefficients and coordination numbers.
        """
        # print(self.c6[item[0], item[1], ...].shape)
        # print(self.cn.shape)
        # print(self.cn[item[0], :].shape)
        # print(self.c6.shape)
        # quit()
        # return D3Reference(
        #     self.cn[item, :],
        #     self.c6[item.reshape(-1, 1), item.reshape(1, -1), :, :],
        # )
        return D3Reference(
            self.cn[item, :],
            self.c6[item[0], item[1], ...], #item, :, :
        )


def weight_references(
    cn: Array,
    ref_cn: Reference,
    weighting_function,
    **kwargs,
) -> Array:
    """
    Calculate the weights of the reference system.

    Parameters
    ----------
    cn : Array
        Coordination numbers for all atoms in the system.
    reference : Reference
        Reference systems for D3 model.
    weighting_function : Callable
        Function to calculate weight of individual reference systems.

    Returns
    -------
    Array
        Weights of all reference systems
    """
    # print(cn.shape, reference.cn.shape)
    # quit()
    weighted_cn = weighting_function(ref_cn - cn[:, None], **kwargs)
    weights = jnp.where(
        ref_cn >= 0,
        weighted_cn, #.reshape((-1, 1))
        0,
    )
    norms = jnp.sum(weights, -1) + jnp.finfo(cn.dtype).eps

    return weights / norms.reshape((-1, 1))


def atomic_c6(weights: Array, idx, ref_c6: Reference) -> Array:
    """
    Compute atomic dispersion coefficients from reference weights.

    Parameters
    ----------
    weights : Array
        Weights for each atom and reference.
    reference : Reference
        Reference dispersion coefficients.

    Returns
    -------
    Array
        Atomic dispersion coefficients.
    """

    # gw = (
    #     weights[jnp.newaxis, :, jnp.newaxis, :]
    #     * weights[:, jnp.newaxis, :, jnp.newaxis]
    # )
    # gw = (
    #     weights[ :, jnp.newaxis, :]
    #     * weights[:, :, jnp.newaxis]
    # )

    gw2 = weights[idx[0], None, :] * weights[idx[1], :, None]
    # print(gw.shape, gw2.shape)
    # print(gw2.shape, weights.shape, ref_c6.shape)
    # quit()
    # c6 = jnp.sum(gw * reference.c6, axis=(-1, -2))
    c6 = jnp.sum(gw2 * ref_c6, axis=(-1, -2))
    return c6



class DispaxD3(nn.Module):
    s6: float
    s8: float
    a1: float
    a2: float

    cn_onset: Array = 20.0
    cn_cutoff: Array = 25.0
    e2_onset: Array = 55.0
    e2_cutoff: Array = 60.0

    def setup(self):
        self.param = dict(s6=self.s6, a1=self.a1, s8=self.s8, a2=self.a2)

        damping_fn = rational_damping
        counting_fn = ncoord.exp_count

        self.cn_fn = multiplicative_isotropic_cutoff(counting_fn, self.cn_onset, self.cn_cutoff)
        self.e2_fn = multiplicative_isotropic_cutoff(damping_fn, self.e2_onset, self.e2_cutoff)
        self.distance = jax.vmap(space.distance, 0, 0)

    def __call__(self, dr_vec, numbers, idx):
        dr = self.distance(dr_vec) / units.Bohr
        # dr nbr x 3
        # print(idx.shape)
        Zi, Zj = numbers[idx[0]], numbers[idx[1]]

        # ref = D3Reference.from_numbers(numbers) # idx
        ref = D3Reference.full()
        # Z x 5, Zi x Zj x 5 x 5
        # ref_cni = ref.cn[Zi]
        # ref_cnj = ref.cn[Zj]
        ref_cn = ref.cn[numbers]
        ref_c6 = ref.c6[Zj, Zi, ...]
        # print(ref.cn.shape, ref.c6.shape)
        # print(ref.cn, ref.c6)
        # print(ref_c6.shape)
        # print(ref_c6)
        # quit()
        data = D3Data.from_idxs([Zi, Zj])
        # print(data.rc.shape, data.qq.shape)
        # print(data.rc, data.qq)
        # quit()

        mask = dr > 0

        masked_cn = jnp.where(mask, self.cn_fn(dr, rc=data.rc), 0)
        # print(mask.shape, masked_cn.shape)
        # print(masked_cn)
        # quit()
        # cn = jnp.sum(masked_cn, -1) # segment sum

        # N atoms
        cn = jax.ops.segment_sum(masked_cn, idx[0], num_segments=numbers.shape[0])
        # print(cn)
        # quit()

        # Natoms x 5
        weights = weight_references(cn, ref_cn, model.gaussian_weight)
        # print(weights.shape)
        # print(weights)
        # quit()
        c6 = atomic_c6(weights, idx, ref_c6)
        # print(c6.shape)
        # print(c6)
        # quit()
        # print(dr.shape, c6.shape, data.qq.shape)
        # quit()
        energy = self.e2_fn(dr, c6=c6, qq=data.qq, rvdw=data.rvdw, **self.param) / 2
        # print(c6)
        # quit()
        # energy = jnp.where(
        #     mask,
        #     e2,
        #     0,
        # )
        return jnp.sum(energy) * units.Ha



def build_d3_energy_neighbor_fns(atoms, cn_onset, cn_cutoff, e2_onset, e2_cutoff, dr_threshold, neigbor_from_jax):
    r_max = e2_cutoff
    atomic_numbers = jnp.asarray(atoms.numbers)
    box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
    neigbor_from_jax = neighbor_calculable_with_jax(box, r_max)
    box = box.T
    displacement_fn = None
    neighbor_fn = None

    if neigbor_from_jax:
        if np.all(box < 1e-6):
            displacement_fn, _ = space.free()
        else:
            displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)
        neighbor_fn = jax_md_reduced.partition.neighbor_list(
            displacement_fn,
            box,
            r_max,
            dr_threshold,
            fractional_coordinates=True,
            disable_cell_list=True,
            format=partition.Sparse,
        )

    Z = jnp.asarray(atomic_numbers)
    n_species = 119  # int(np.max(Z) + 1)
    # builder = ModelBuilder(config.model.get_dict(), n_species=n_species)

    # model = builder.build_energy_derivative_model(
    #     apply_mask=True, init_box=np.array(box), inference_disp_fn=displacement_fn
    # )

    # param = dict(s6=1.0, a1=0.49484001, s8=0.78981345, a2=5.73083694)
    param = dict(s6=1.0, a1=0.37, s8=1.5, a2=4.1)
    d3model = DispaxD3(**param, cn_onset=cn_onset, cn_cutoff=cn_cutoff, e2_onset=e2_onset, e2_cutoff=e2_cutoff)

    emodel = EnergyModel(
        d3model,
        corrections=[],
        init_box=np.array(box),
        inference_disp_fn=displacement_fn,
    )

    model = EnergyDerivativeModel(
        energy_model=emodel,
        calc_stress=True
    )

    params = {}
    # energy_fn = maybe_vmap(model.apply, params, Z)
    energy_fn = partial(model.apply, params)

    return energy_fn, neighbor_fn





def process_stress(results, box):
    V = quantity.volume(3, box)
    results = {
        # We should properly check whether CP2K uses the ASE cell convention
        # for tetragonal strain, it doesn't matter whether we transpose or not
        k: val.T / V if k.startswith("stress") else val
        for k, val in results.items()
    }
    return results


def make_ensemble(model):
    def ensemble(positions, Z, idx, box, offsets):
        results = model(positions, Z, idx, box, offsets)
        uncertainty = {k + "_uncertainty": jnp.std(v, axis=0) for k, v in results.items()}
        results = {k: jnp.mean(v, axis=0) for k, v in results.items()}
        results.update(uncertainty)

        return results

    return ensemble


class ASECalculator(Calculator):
    """
    ASE Calculator for apax models.
    """

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        model_dir: Union[Path, list[Path]],
        dr_threshold: float = 0.5,
        transformations: Callable = [],
        padding_factor: float = 1.5,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.dr_threshold = dr_threshold
        self.transformations = transformations

        self.model_config, self.params = restore_parameters(model_dir)
        self.n_models = check_for_ensemble(self.params)
        self.padding_factor = padding_factor
        self.padded_length = 0

        if self.model_config.model.calc_stress:
            self.implemented_properties.append("stress")

        if self.n_models > 1:
            uncertainty_kws = [
                prop + "_uncertainty" for prop in self.implemented_properties
            ]
            self.implemented_properties += uncertainty_kws

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None

    def initialize(self, atoms):
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.r_max = self.model_config.model.r_max
        self.neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
        model, neighbor_fn = build_energy_neighbor_fns(
            atoms,
            self.model_config,
            self.params,
            self.dr_threshold,
            self.neigbor_from_jax,
        )

        if self.n_models > 1:
            model = make_ensemble(model)

        for transformation in self.transformations:
            model = transformation.apply(model, self.n_models)

        self.step = get_step_fn(model, atoms, self.neigbor_from_jax)
        self.neighbor_fn = neighbor_fn

    def set_neighbours_and_offsets(self, atoms, box):
        idxs_i, idxs_j, offsets = neighbour_list("ijS", atoms, self.r_max)

        if len(idxs_i) > self.padded_length:
            print("neighbor list overflowed, reallocating.")
            self.padded_length = int(len(idxs_i) * self.padding_factor)
            self.initialize(atoms)

        zeros_to_add = self.padded_length - len(idxs_i)

        self.neighbors = np.array([idxs_i, idxs_j], dtype=np.int32)
        self.neighbors = np.pad(self.neighbors, ((0, 0), (0, zeros_to_add)), "constant")

        offsets = np.matmul(offsets, box)
        self.offsets = np.pad(offsets, ((0, zeros_to_add), (0, 0)), "constant")

    def calculate(self, atoms, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)

        # setup model and neighbours
        if self.step is None:
            self.initialize(atoms)

            if self.neigbor_from_jax:
                self.neighbors = self.neighbor_fn.allocate(positions)
            else:
                idxs_i = neighbour_list("i", atoms, self.r_max)
                self.padded_length = int(len(idxs_i) * self.padding_factor)

        elif "numbers" in system_changes:
            self.initialize(atoms)

            if self.neigbor_from_jax:
                self.neighbors = self.neighbor_fn.allocate(positions)

        elif "cell" in system_changes:
            neigbor_from_jax = neighbor_calculable_with_jax(box, self.r_max)
            if self.neigbor_from_jax != neigbor_from_jax:
                self.initialize(atoms)

        # predict
        if self.neigbor_from_jax:
            results, self.neighbors = self.step(positions, self.neighbors, box)

            if self.neighbors.did_buffer_overflow:
                print("neighbor list overflowed, reallocating.")
                self.initialize(atoms)
                self.neighbors = self.neighbor_fn.allocate(positions)

                results, self.neighbors = self.step(positions, self.neighbors, box)

        else:
            self.set_neighbours_and_offsets(atoms, box)
            positions = np.array(space.transform(np.linalg.inv(box), atoms.positions))

            results = self.step(positions, self.neighbors, box, self.offsets)

        self.results = {k: np.array(v, dtype=np.float64) for k, v in results.items()}
        self.results["energy"] = self.results["energy"].item()




class D3Calculator(ASECalculator):
    """
    ASE Calculator for apax models.
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress"
    ]

    def __init__(
        self,
        cn_onset=20.0,
        cn_cutoff=25.0,
        e2_onset=55.0,
        e2_cutoff=60.0,
        dr_threshold: float = 0.5,
        padding_factor: float = 1.5,
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.cn_onset = cn_onset
        self.cn_cutoff = cn_cutoff
        self.e2_onset = e2_onset
        self.e2_cutoff = e2_cutoff
        self.r_max = e2_cutoff
        self.dr_threshold = dr_threshold
        self.padding_factor = padding_factor
        self.padded_length = 0
        # if self.model_config.model.calc_stress:
        #     self.implemented_properties.append("stress")

        self.step = None
        self.neighbor_fn = None
        self.neighbors = None
        self.offsets = None

    def initialize(self, atoms):
        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        self.neigbor_from_jax = neighbor_calculable_with_jax(box, self.e2_cutoff)
        model, neighbor_fn = build_d3_energy_neighbor_fns(
            atoms,
            self.cn_onset,
            self.cn_cutoff,
            self.e2_onset,
            self.e2_cutoff,
            self.dr_threshold,
            self.neigbor_from_jax,
        )
        self.step = get_step_fn(model, atoms, self.neigbor_from_jax)
        self.neighbor_fn = neighbor_fn






def neighbor_calculable_with_jax(box, r_max):
    if np.all(box < 1e-6):
        return True
    else:
        # all lettice vector combinations to calculate all three plane distances
        a_vec_list = [box[0], box[0], box[1]]
        b_vec_list = [box[1], box[2], box[2]]
        c_vec_list = [box[2], box[1], box[0]]

        height = []
        for i in range(3):
            normvec = np.cross(a_vec_list[i], b_vec_list[i])
            projection = (
                c_vec_list[i]
                - np.sum(normvec * c_vec_list[i]) / np.sum(normvec**2) * normvec
            )
            height.append(np.linalg.norm(c_vec_list[i] - projection))

        if np.min(height) / 2 > r_max:
            return True
        else:
            return False


def get_step_fn(model, atoms, neigbor_from_jax):
    Z = jnp.asarray(atoms.numbers)
    if neigbor_from_jax:

        @jax.jit
        def step_fn(positions, neighbor, box):
            if np.any(atoms.get_cell().lengths() > 1e-6):
                box = box.T
                inv_box = jnp.linalg.inv(box)
                positions = space.transform(inv_box, positions)
                neighbor = neighbor.update(positions, box=box) # neighbor ist array, warum?
            else:
                neighbor = neighbor.update(positions)

            offsets = jnp.full([neighbor.idx.shape[1], 3], 0)
            results = model(positions, Z, neighbor.idx, box, offsets)

            if "stress" in results.keys():
                results = process_stress(results, box)

            return results, neighbor

    else:

        @jax.jit
        def step_fn(positions, neighbor, box, offsets):
            results = model(positions, Z, neighbor, box, offsets)

            if "stress" in results.keys():
                results = process_stress(results, box)

            return results

    return step_fn
