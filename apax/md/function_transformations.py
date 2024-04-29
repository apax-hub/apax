import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class FunctionTransformation:
    def apply(self, model, n_models):
        raise NotImplementedError


def make_biased_energy_force_fn(bias_fn):
    def biased_energy_force_fn(positions, Z, idx, box, offsets):
        gamd_fn = jax.value_and_grad(bias_fn, has_aux=True)

        (E_bias, results), F_bias = gamd_fn(positions, Z, idx, box, offsets)

        if "energy_unbiased" not in results.keys():
            results["energy_unbiased"] = results["energy"]
            results["forces_unbiased"] = results["forces"]

        results["energy"] = results["energy"] + E_bias
        results["forces"] = results["forces"] + F_bias

        return results

    return biased_energy_force_fn


class UncertaintyDrivenDynamics(FunctionTransformation):
    """
    UDD requires an uncertainty aware model.
    It drives the dynamics towards higher uncertainty regions
    up to some maximum bias energy.
    https://doi.org/10.1038/s43588-023-00406-5


    Parameters
    ----------
    height : float
        Maximum bias potential that can be applied
    width : float
        Width of the Gaussian bias.

    """

    height: float
    width: float

    def apply(self, model, n_models):
        def udd_energy(positions, Z, idx, box, offsets):
            n_atoms = positions.shape[0]
            results = model(positions, Z, idx, box, offsets)

            sigma2 = results["energy_uncertainty"] ** 2

            gauss = jnp.exp(-sigma2 / (n_models * n_atoms * self.width**2))
            E_udd = self.height * (gauss - 1)

            return E_udd, results

        udd_energy_force = make_biased_energy_force_fn(udd_energy)

        return udd_energy_force


class GaussianAcceleratedMolecularDynamics(FunctionTransformation):
    """
    Applies a boost potential to the system that pulls it towards a target energy.
    https://pubs.acs.org/doi/10.1021/acs.jctc.5b00436

    Parameters
    ----------
    energy_target : float
        Target potential energy below which to apply the boost potential.
    spring_constant : float
        Spring constant of the boost potential.
    """

    energy_target: float
    spring_constant: float

    def apply(self, model, n_models):
        def gamd_energy(positions, Z, idx, box, offsets):
            results = model(positions, Z, idx, box, offsets)

            energy = jnp.clip(results["energy"], a_max=self.energy_target)

            E_gamd = 0.5 * self.spring_constant * (energy - self.energy_target) ** 2

            return E_gamd, results

        gamd_energy_force = make_biased_energy_force_fn(gamd_energy)

        return gamd_energy_force


available_transformations = {
    "udd": UncertaintyDrivenDynamics,
    "gamd": GaussianAcceleratedMolecularDynamics,
}
