import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.units import eV, kcal, mol, Ang, Bohr, Hartree, kJ

def convert_atoms_to_arrays(
    atoms_list: list[Atoms],
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
    ) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, list]]]:
    """Converts an list of ASE atoms to two dicts where all inputs and labels
    are sorted by there shape (ragged/fixed), and proberty.


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
            "cell": [],
        },
    }

    labels = {
        "ragged": {
            "forces": [],
        },
        "fixed": {
            "energy": [],
        },
    }
    # Hier hard gecoded mit absicht? 
    dtype = np.float32  # float32

    unit_dict = {
        "Ang": Ang,
        "Bohr": Bohr,
        "eV": eV,
        "kcal/mol": kcal / mol,
        'Hartree': Hartree,
        "kJ/mol": kJ / mol,
    }

    for atoms in atoms_list:
        inputs["ragged"]["positions"].append(atoms.positions.astype(dtype) * unit_dict[pos_unit])
        inputs["ragged"]["numbers"].append(atoms.numbers)
        inputs["fixed"]["n_atoms"].append(len(atoms))
        if atoms.pbc.any():
            cell = np.array(atoms.cell).diagonal().astype(dtype)
            inputs["fixed"]["cell"].append(list(cell))

        for key, val in atoms.calc.results.items():
            if key is ["forces"]:
                labels["ragged"][key].append(val * unit_dict[energy_unit] / unit_dict[pos_unit])
            elif key is ["energy"]:
                    labels["fixed"][key].append(val * unit_dict[energy_unit])

    inputs["ragged"] = {
        key: val for key, val in inputs["ragged"].items() if len(val) != 0
    }
    inputs["fixed"] = {key: val for key, val in inputs["fixed"].items() if len(val) != 0}
    labels["ragged"] = {
        key: val for key, val in labels["ragged"].items() if len(val) != 0
    }
    labels["fixed"] = {key: val for key, val in labels["fixed"].items() if len(val) != 0}
    return inputs, labels


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
