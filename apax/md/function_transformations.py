import dataclasses

import jax
import jax.numpy as jnp


def make_biased_energy_force_fn(bias_fn):
    def biased_energy_force_fn(positions, Z, idx, box, offsets):
        bias_and_grad_fn = jax.value_and_grad(bias_fn, has_aux=True)

        (E_bias, results), neg_F_bias = bias_and_grad_fn(positions, Z, idx, box, offsets)

        if "energy_unbiased" not in results.keys():
            results["energy_unbiased"] = results["energy"]
            results["forces_unbiased"] = results["forces"]

        F_bias = -neg_F_bias
        results["energy"] = results["energy"] + E_bias
        results["forces"] = results["forces"] + F_bias

        return results

    return biased_energy_force_fn


@dataclasses.dataclass
class UncertaintyDrivenDynamics:
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

    def apply(self, model):
        def udd_energy(positions, Z, idx, box, offsets):
            n_atoms = positions.shape[0]
            results = model(positions, Z, idx, box, offsets)
            n_models = results["energy_ensemble"].shape[0]

            sigma2 = results["energy_uncertainty"] ** 2

            gauss = jnp.exp(-sigma2 / (n_models * n_atoms * self.width**2))
            E_udd = self.height * (gauss - 1)

            return E_udd, results

        udd_energy_force = make_biased_energy_force_fn(udd_energy)

        return udd_energy_force


@dataclasses.dataclass
class GaussianAcceleratedMolecularDynamics:
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

    def apply(self, model):
        def gamd_energy(positions, Z, idx, box, offsets):
            results = model(positions, Z, idx, box, offsets)

            energy = jnp.clip(results["energy"], a_max=self.energy_target)

            E_gamd = 0.5 * self.spring_constant * (energy - self.energy_target) ** 2

            return E_gamd, results

        gamd_energy_force = make_biased_energy_force_fn(gamd_energy)

        return gamd_energy_force


@dataclasses.dataclass
class GlobalCalibration:
    """
    Applies a global calibration to energy and force uncertainties.
    Energy ensemble predictions are rescaled according to EQ 7 in
    https://doi.org/10.1063/5.0036522

    Parameters
    ----------
    energy_factor : float
        Global calibration factor by which to scale the energy uncertainty.
    forces_factor : float
        Global calibration factor by which to scale the force uncertainties.
    """

    energy_factor: float
    forces_factor: float

    def apply(self, model):
        def calibrated_model(positions, Z, idx, box, offsets):
            results = model(positions, Z, idx, box, offsets)

            results["energy_uncertainty"] = (
                results["energy_uncertainty"] * self.energy_factor
            )

            Emean = results["energy"]
            Ei = results["energy_ensemble"]
            results["energy_ensemble"] = Emean + self.energy_factor * (Ei - Emean)

            if "forces_uncertainty" in results.keys():
                results["forces_uncertainty"] = (
                    results["forces_uncertainty"] * self.forces_factor
                )

            return results

        return calibrated_model


available_transformations = {
    "udd": UncertaintyDrivenDynamics,
    "gamd": GaussianAcceleratedMolecularDynamics,
    "global_cal": GlobalCalibration,
}
