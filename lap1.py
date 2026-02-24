from functools import partial

import jax
import numpy as np
from apax.bal.api import compute_features as compute_features_g
from apax.bal.api import create_feature_fn
from apax.bal.feature_maps import LastLayerForceFeatures, LastLayerGradientFeatures
from apax.data.input_pipeline import OTFInMemoryDataset
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    restore_parameters,
)
from ase import Atoms
from flax.core.frozen_dict import freeze, unfreeze
from tqdm import trange

from apax.utils.transform import make_energy_only_model
from laplax.utils import get_model_and_config, sample_params, save_checkpoint


def compute_features(feature_fn, dataset: OTFInMemoryDataset) -> np.ndarray:
    """Compute the features of a dataset.

    Attributes
    ----------
    feature_fn:
        Function to compute the features with.
    dataset:
        Dataset to compute the features for.
    """
    features = []
    n_data = dataset.n_data
    ds = dataset.batch()

    pbar = trange(n_data, desc="Computing features", ncols=100, leave=True)
    for inputs in ds:
        R, Z, idx, box, offsets = (
            inputs["positions"],
            inputs["numbers"],
            inputs["idx"],
            inputs["box"],
            inputs["offsets"],
        )
        g = feature_fn(R, Z, idx, box, offsets, None)
        features.append(np.asarray(g, dtype=np.float64))
        pbar.update(g.shape[0])
    pbar.close()

    features = np.concatenate(features, axis=0)
    return features


def dataset_features(
    atoms_list, config, params, processing_batch_size, quantity="energy",
):

    dataset = OTFInMemoryDataset(
        atoms_list,
        cutoff=config.model.basis.r_max,
        bs=processing_batch_size,
        n_epochs=1,
        ignore_labels=True,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    _, init_box = dataset.init_input()

    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=119)

    model = builder.build_energy_model(apply_mask=True, init_box=init_box)
    model = make_energy_only_model(model.apply)

    if quantity == "energy":
        feature_map = LastLayerGradientFeatures()
    elif quantity == "forces":
        feature_map = LastLayerForceFeatures()
    else:
        raise NotImplementedError("only energy and force features implemented")

    feature_fn = create_feature_fn(model, params, feature_map, [], False)
    g = compute_features_g(feature_fn, dataset)

    if quantity == "energy":
        # remove bias
        g = g[:, :-1]
    return g


def compute_predictions(model, params, dataset, atomic_numbers=None):
    predictions = {
        "energy": [],
        "forces": [],
    }
    n_data = dataset.n_data
    ds = dataset.batch()

    pbar = trange(n_data, desc="Computing predictions", ncols=100, leave=True)
    for inputs, labels in ds:
        R, Z, idx, box, offsets = (
            inputs["positions"],
            inputs["numbers"],
            inputs["idx"],
            inputs["box"],
            inputs["offsets"],
        )
        pred = model(params, R, Z, idx, box, offsets)
        energies = pred["energy"]
        forces = pred["forces"]

        predictions["energy"].append(energies)
        predictions["forces"].append(forces)

        pbar.update(energies.shape[0])
    pbar.close()

    predictions["energy"] = np.concatenate(predictions["energy"], axis=0)
    predictions["forces"] = np.concatenate(predictions["forces"], axis=0)
    return predictions


def dataset_preds(
    atoms_list: list[Atoms], config, params, processing_batch_size, atomic_numbers = None,
):

    force_loss = "forces" in atoms_list[0].calc.results.keys()

    E_true = [x.get_potential_energy() for x in atoms_list]
    E_true = np.array(E_true)

    if force_loss:
        F_true = []
        for atoms in atoms_list:
            forces = atoms.get_forces()
            if atomic_numbers:
                mask = [a.number in atomic_numbers for a in atoms ]
                forces = forces[mask]
            F_true.append(forces)
        F_true = np.array(F_true)

    atom_counts = np.array([len(x) for x in atoms_list])

    dataset = OTFInMemoryDataset(
        atoms_list,
        cutoff=config.model.basis.r_max,
        bs=processing_batch_size,
        n_epochs=1,
        ignore_labels=False,
        pos_unit=config.data.pos_unit,
        energy_unit=config.data.energy_unit,
    )

    _, init_box = dataset.init_input()

    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump(), n_species=119)
    model = builder.build_energy_derivative_model(apply_mask=True, init_box=init_box)

    batched_model = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0))
    batched_model = jax.jit(batched_model)
    preds = compute_predictions(batched_model, params, dataset)

    if atomic_numbers:
        pred_forces = []

        for i in range(len(atoms_list)):
            numbers = atoms_list[i].get_atomic_numbers()
            mask1d = np.isin(numbers, atomic_numbers)
            pred_forces.append(preds["forces"][i, mask1d, :])

        preds["forces"] = np.array(pred_forces)


    loss = {}
    loss["energy"] = (preds["energy"] - E_true) ** 2

    errors = {}

    E_true = E_true / atom_counts * 1000
    E_pred = preds["energy"] / atom_counts * 1000
    e_errors = np.abs(E_true - E_pred)
    errors["energy"] = e_errors

    if force_loss:
        F_true = np.reshape(F_true, (-1,))
        F_pred = np.reshape(preds["forces"], (-1,))
        loss["forces"] = (F_pred - F_true) ** 2
        f_errors = (F_true - F_pred) * 1000
        errors["forces"] = f_errors

    return preds, loss, errors


def compute_inv_kernel(g, eps):
    lambd = (eps**2) * np.eye(g.shape[1])
    kernel = g.T @ g + lambd
    K_inv = np.linalg.inv(kernel)
    return K_inv


def inv_kernel_precomp(gTg, eps):
    lambd = (eps**2) * np.eye(gTg.shape[0])
    kernel = gTg + lambd
    print("inverting")
    K_inv = np.linalg.pinv(kernel)
    print("inverted")
    K_inv = (K_inv + K_inv.T) / 2.0
    return K_inv


def compute_energy_variance(g_cal, K_inv):
    # b batch, f,g features
    variance = np.einsum("bf, fg, bg -> b", g_cal, K_inv, g_cal)
    return variance


def compute_forces_variance(g_cal, K_inv):
    # b batch, f,g features, a atoms, x cartesian coords
    variance = np.einsum("baxf, fg, baxg -> bax", g_cal, K_inv, g_cal)
    return variance


def is_psd(x):
    return np.all(np.linalg.eigvalsh(x) > 0)


def ensure_psd(g):
    gTg = g.T @ g
    print("gtg")

    eps = 5e-4
    K_inv = inv_kernel_precomp(gTg, eps)
    print("K_inv")
    if not is_psd(K_inv):
        count = 0
        max_tries = 10000
        while not is_psd(K_inv) and count <= max_tries:
            eps = eps * 2.0
            K_inv = inv_kernel_precomp(gTg, eps)

    eps = eps * 10
    K_inv = inv_kernel_precomp(gTg, eps)
    print("K_inv")
    return K_inv, eps


def calibrate_energy(g_cal, K_inv, e_loss):
    e_variances = compute_energy_variance(g_cal, K_inv)
    alpha2 = np.mean(e_loss / e_variances)
    return alpha2


def calibrate_forces(g_cal, K_inv, f_loss):
    f_variances = compute_forces_variance(g_cal, K_inv)

    f_variances = np.reshape(f_variances, (-1,))
    alpha2 = np.mean(f_loss / f_variances)
    return alpha2


def calibrate_mse(K_inv, features, loss, weights, include_force_loss):
    alpha2 = weights["energy"] * calibrate_energy(
        features["energy"], K_inv, loss["energy"]
    )

    if include_force_loss:
        f_alpha2 = weights["forces"] * calibrate_forces(
            features["forces"], K_inv, loss["forces"]
        )
        # alpha2 = (alpha2 + f_alpha2) /2
        alpha2 = f_alpha2

    return alpha2



def flatten_forces(stacked_forces, atom_counts):
    forces = []

    for f, n in zip(stacked_forces, atom_counts):
        filtered_force = f[:n, :]
        filtered_force = np.reshape(filtered_force, (-1))
        forces.append(filtered_force)

    forces = np.concatenate(forces, 0)
    return forces



def create_laplace_ensemble(
    model_dir,
    train_atoms,
    val_atoms,
    test_atoms,
    n_shallow,
    batch_size=10,
    layer_name="dense_2",
    force_variance=True,
    force_contribution=True,
    include_force_loss=True,
    atomic_numbers=None,
    chunk_size=None,
    fix_ll_mean: bool = False,
    seed=0,
    overwrite_dir_and_exp=None,
):
    # atomic_numbers = [6]
    np.random.seed(seed)
    config, params = restore_parameters(model_dir)
    # define it here so we don't have to convert back to energy derivative model
    shallow_params = unfreeze(params)

    # params = canonicalize_energy_model_parameters(params)
    model_dict = config.model.model_dump()
    loss_config = [lossconf.model_dump() for lossconf in config.loss]
    energy_loss = [lc["weight"] for lc in loss_config if lc["name"] == "energy"]
    if len(energy_loss) > 0:
        energy_weight = energy_loss[0]
    else:
        energy_weight = 0.0

    forces_loss = [lc["weight"] for lc in loss_config if lc["name"] == "forces"]
    if len(forces_loss) > 0:
        forces_weight = forces_loss[0]
    else:
        forces_weight = 0.0

    loss_weights = {
        "energy": energy_weight,
        "forces": forces_weight,
    }

    all_same_size = len(set([len(a) for a in train_atoms]))
    all_same_size = all_same_size and len(set([len(a) for a in val_atoms]))
    all_same_size = all_same_size and len(set([len(a) for a in test_atoms]))

    preds_cal, loss_cal, errors_cal = dataset_preds(
        val_atoms, config, params, batch_size, atomic_numbers
    )
    preds_test, _, errors_test = dataset_preds(
        test_atoms, config, params, batch_size, atomic_numbers
    )

    params = canonicalize_energy_model_parameters(params)
    g = dataset_features(
        train_atoms, config, params, batch_size, quantity="energy"
    )
    g_cal = dataset_features(
        val_atoms, config, params, batch_size, quantity="energy"
    )
    g_test = dataset_features(
        test_atoms, config, params, batch_size, quantity="energy"
    )
    features_train = {"full_features": energy_weight * g, "energy": g}
    features_cal = {"full_features": energy_weight * g_cal, "energy": g_cal}
    features_test = {"full_features": energy_weight * g_test, "energy": g_test}

    if force_variance:
        g_f = dataset_features(
            train_atoms, config, params, 1, quantity="forces",
        )
        g_f_cal = dataset_features(
            val_atoms, config, params, 1, quantity="forces",
        )
        g_f_test = dataset_features(
            test_atoms, config, params, 1, quantity="forces",
        )
        print("features done")

        if atomic_numbers is not None:
            g_f_filtered = []
            g_f_cal_filtered = []
            g_f_test_filtered = []

            for i in range(len(train_atoms)):

                numbers = train_atoms[i].get_atomic_numbers()
                mask1d = np.isin(numbers, atomic_numbers)

                g_f_filtered.append(g_f[i, mask1d, :, :])

            for i in range(len(val_atoms)):

                numbers = val_atoms[i].get_atomic_numbers()
                mask1d = np.isin(numbers, atomic_numbers)

                g_f_cal_filtered.append(g_f_cal[i, mask1d, :, :])
            
            for i in range(len(test_atoms)):

                numbers = test_atoms[i].get_atomic_numbers()
                mask1d = np.isin(numbers, atomic_numbers)

                g_f_test_filtered.append(g_f_test[i, mask1d, :, :])

            g_f = g_f_filtered
            g_f_cal = g_f_cal_filtered
            g_f_test = g_f_test_filtered
        print("filtered")

        # if "ensemble" in model_dict.keys() and model_dict["ensemble"]:
        #     ensemble_factor = model_dict["ensemble"]["n_members"]
        #     g_f /= ensemble_factor
        #     g_f_cal /= ensemble_factor
        #     g_f_test /= ensemble_factor

        # if all_same_size:
        g_f_aggregated = np.einsum("baxf -> bf", g_f)
        g_f_cal_aggregated = np.einsum("baxf -> bf", g_f_cal)
        g_f_test_aggregated = np.einsum("baxf -> bf", g_f_test)
        print("aggregated")

        features_train["forces"] = g_f
        features_cal["forces"] = g_f_cal
        features_test["forces"] = g_f_test

        if force_contribution:
            features_train["full_features"] += forces_weight * g_f_aggregated
            features_cal["full_features"] += forces_weight * g_f_cal_aggregated
            features_test["full_features"] += forces_weight * g_f_test_aggregated

    K_inv, eps = ensure_psd(features_train["full_features"])
    alpha2 = calibrate_mse(
        K_inv,
        features_cal,
        loss_cal,
        loss_weights,
        include_force_loss,
    )
    covariance = alpha2 * K_inv
    print("calibrated")

    variance_energy_cal = compute_energy_variance(features_cal["energy"], covariance)
    uncertainty_energy_cal = np.sqrt(variance_energy_cal)
    print("e var")
    variance_energy_test = compute_energy_variance(features_test["energy"], covariance)
    uncertainty_energy_test = np.sqrt(variance_energy_test)
    print("e  var")

    if force_variance:
        variance_forces_cal = compute_forces_variance(
            features_cal["forces"], covariance
        )
        print("f var")
        variance_forces_test = compute_forces_variance(
            features_test["forces"], covariance
        )
        print("f var")

        if all_same_size:
            variance_forces_cal = np.reshape(variance_forces_cal, (-1))
            variance_forces_test = np.reshape(variance_forces_test, (-1))
        else:
            atom_counts = np.array([len(x) for x in val_atoms])
            variance_forces_cal = flatten_forces(variance_forces_cal, atom_counts)
            atom_counts = np.array([len(x) for x in test_atoms])
            variance_forces_test = flatten_forces(variance_forces_test, atom_counts)

        uncertainty_forces_cal = np.sqrt(variance_forces_cal)
        uncertainty_forces_test = np.sqrt(variance_forces_test)

    params = unfreeze(params)
    params = {
        "params": {
            "representation": params["params"]["representation"],
            "readout": params["params"]["readout"],
        },
    }
    params = freeze(params)

    shallow_params, w_old, w_new = sample_params(
        params,
        shallow_params,
        covariance,
        n_shallow,
        layer_name,
    )
    shallow_params = freeze(shallow_params)

    shallow_config, shallow_model = get_model_and_config(
        config, n_shallow, force_variance, chunk_size, fix_ll_mean, overwrite_dir_and_exp
    )
    model_path = shallow_config.data.model_version_path
    save_checkpoint(shallow_config, shallow_model, shallow_params, model_path)

    results = {
        "alpha2": alpha2,
        "eps": eps,
        "preds_cal": preds_cal,
        "preds_test": preds_test,
        "uncertainty_cal": uncertainty_energy_cal,
        "uncertainty_test": uncertainty_energy_test,
        "errors_cal": errors_cal["energy"],
        "errors_test": errors_test["energy"],
        "g": features_train["energy"],
        "g_cal": features_cal["energy"],
        "g_test": features_test["energy"],
        "covariance": covariance,
        "w_old": w_old,
        "w_new": w_new,
    }

    if force_variance:
        results["errors_forces_cal"] = errors_cal["forces"]
        results["errors_forces_test"] = errors_test["forces"]
        results["uncertainty_forces_cal"] = uncertainty_forces_cal
        results["uncertainty_forces_test"] = uncertainty_forces_test
        results["gf"] = features_train["forces"]
        results["gf_cal"] = features_cal["forces"]
        results["gf_test"] = features_test["forces"]
    return results
