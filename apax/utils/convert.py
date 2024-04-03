import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.units import Ang, Bohr, Hartree, eV, kcal, kJ, mol

from apax.utils.jax_md_reduced import space

DTYPE = np.float64
unit_dict = {
    "Ang": Ang,
    "Bohr": Bohr,
    "eV": eV,
    "kcal/mol": kcal / mol,
    "Hartree": Hartree,
    "kJ/mol": kJ / mol,
}


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


def atoms_to_inputs(
    atoms_list: list[Atoms],
    pos_unit: str = "Ang",
) -> dict[str, dict[str, list]]:
    """Converts an list of ASE atoms to a dict where all inputs
    are sorted by their shape (ragged/fixed). Units are
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
        "positions": [],
        "numbers": [],
        "n_atoms": [],
        "box": [],
    }

    box = atoms_list[0].cell.array
    pbc = np.any(np.abs(box) > 1e-6)

    for atoms in atoms_list:
        box = (atoms.cell.array * unit_dict[pos_unit]).astype(DTYPE)
        box = box.T  # takes row and column convention of ase into account
        inputs["box"].append(box)

        if pbc != np.any(np.abs(box) > 1e-6):
            raise ValueError(
                "Apax does not support dataset periodic and non periodic structures"
            )

        if np.all(np.abs(box) < 1e-6):
            inputs["positions"].append(
                (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
            )
        else:
            inv_box = np.linalg.inv(box)
            pos = (atoms.positions * unit_dict[pos_unit]).astype(DTYPE)
            frac_pos = space.transform(inv_box, pos)
            inputs["positions"].append(np.array(frac_pos))

        inputs["numbers"].append(atoms.numbers.astype(np.int16))
        inputs["n_atoms"].append(len(atoms))

    inputs = prune_dict(inputs)
    return inputs


def atoms_to_labels(
    atoms_list: list[Atoms],
    pos_unit: str = "Ang",
    energy_unit: str = "eV",
) -> dict[str, dict[str, list]]:
    """Converts an list of ASE atoms to a dict of labels
    Units are adjusted if ASE compatible and provided in the inputpipeline.

    Parameters
    ----------
    atoms_list :
        List of all structures. Enties are ASE atoms objects.

    Returns
    -------
    labels :
        Labels are trainable system properties.
    """

    labels = {
        "forces": [],
        "energy": [],
        "stress": [],
    }
    # for key in atoms_list[0].calc.results.keys():
    #     if key not in labels.keys():
    #         placeholder = {key: []}
    #         labels.update(placeholder)

    for atoms in atoms_list:
        for key, val in atoms.calc.results.items():
            if key == "forces":
                labels[key].append(val * unit_dict[energy_unit] / unit_dict[pos_unit])
            elif key == "energy":
                labels[key].append(val * unit_dict[energy_unit])
            elif key == "stress":
                factor = unit_dict[energy_unit] / (unit_dict[pos_unit] ** 3)
                stress = atoms.get_stress(voigt=False) * factor
                labels[key].append(stress * atoms.cell.volume)
            # else:
            #     labels[key].append(atoms.calc.results[key])

    labels = prune_dict(labels)
    return labels
