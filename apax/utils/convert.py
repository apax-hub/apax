import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.units import Ang, Bohr, Hartree, eV, kcal, kJ, mol

from apax.utils import jax_md_reduced


def tf_to_jax_dict(data_dict: dict[str, list]) -> dict:
    """Converts a dict of tf.Tensors to a dict of jax.numpy.arrays.
    tf.Tensors must be padded.

    Parameters
    ----------
    data_dict :
        Dict padded of tf.Tensors

    Returns
    -------
    data_dict :
        Dict of jax.numpy.arrays
    """
    data_dict = {k: jnp.asarray(v) for k, v in data_dict.items()}
    return data_dict


def prune_dict(data_dict):
    pruned = {key: val for key, val in data_dict.items() if len(val) != 0}
    return pruned


def atoms_to_arrays(
    atoms_list: list[Atoms],
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, list]]]:
    """Converts an list of ASE atoms to two dicts where all inputs and labels
    are sorted by there shape (ragged/fixed), and property. Units are
    adjusted if ASE compatible and provided in the inputpipeline.


    Parameters
    ----------
    atoms_list :
        List of all structures. Enties are ASE atoms objects.

    Returns
    -------
    inputs :
        Inputs are untrainable system-determining properties.
    labels :
        Labels are trainable system properties.
    """
    inputs = {
        "ragged": {
            "positions": [],
            "numbers": [],
        },
        "fixed": {
            "n_atoms": [],
            "box": [],
        },
    }

    labels = {
        "ragged": {
            "forces": [],
        },
        "fixed": {
            "energy": [],
            "stress": [],
        },
    }
    DTYPE = np.float64

    unit_dict = {
        "Ang": Ang,
        "Bohr": Bohr,
        "eV": eV,
        "kcal/mol": kcal / mol,
        "Hartree": Hartree,
        "kJ/mol": kJ / mol,
    }
    box = atoms_list[0].cell.array
    pbc = np.all(box > 1e-6)

    for atoms in atoms_list:
        box = (atoms.cell.array * unit_dict[pos_unit]).astype(DTYPE)
        box = box.T  # takes row and column convention of ase into account
        inputs["fixed"]["box"].append(box)

        if pbc != np.all(box > 1e-6):
            raise ValueError(
                "Apax does not support dataset periodic and non periodic structures"
            )

        if np.all(box < 1e-6):
            inputs["ragged"]["positions"].append(
                (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
            )
        else:
            inv_box = np.linalg.inv(box)
            inputs["ragged"]["positions"].append(
                np.array(
                    jax_md_reduced.space.transform(
                        inv_box, (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
                    )
                )
            )

        inputs["ragged"]["numbers"].append(atoms.numbers)
        inputs["fixed"]["n_atoms"].append(len(atoms))
        for key, val in atoms.calc.results.items():
            if key == "forces":
                labels["ragged"][key].append(
                    val * unit_dict[energy_unit] / unit_dict[pos_unit]
                )
            elif key == "energy":
                labels["fixed"][key].append(val * unit_dict[energy_unit])
            elif key == "stress":
                stress = (  # TODO check whether we should transpose
                    atoms.get_stress(voigt=False)  # .T
                    * unit_dict[energy_unit]
                    / (unit_dict[pos_unit] ** 3)
                    * atoms.cell.volume
                )
                labels["fixed"][key].append(stress)

    inputs["fixed"] = prune_dict(inputs["fixed"])
    labels["fixed"] = prune_dict(labels["fixed"])
    inputs["ragged"] = prune_dict(inputs["ragged"])
    labels["ragged"] = prune_dict(labels["ragged"])
    return inputs, labels
