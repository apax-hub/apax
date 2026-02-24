from functools import partial
import copy

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
    """Compute the features of a dataset."""
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
        # Assuming modern apax signature; might require 'perturbation=None'
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

    # Remove bias dimension if present (assuming last feature dim is bias)
    if quantity == "energy":
        g = g[:, :-1]
    # elif quantity == "forces":
    #     # Force features shape: (Batch, Atoms, 3, Features)
    #     g = g[..., :-1]
        
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
    atoms_list: list[Atoms], config, params, processing_batch_size, atomic_numbers=None,
):
    force_loss = "forces" in atoms_list[0].calc.results.keys()

    E_true = np.array([x.get_potential_energy() for x in atoms_list])
    atom_counts = np.array([len(x) for x in atoms_list])

    loss = {}
    errors = {}
    
    # --- Prediction Step ---
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

    # --- Energy Error Calculation ---
    loss["energy"] = (preds["energy"] - E_true) ** 2
    # Error in meV/atom
    errors["energy"] = np.abs(E_true - preds["energy"]) / atom_counts * 1000

    # --- Force Error Calculation ---
    if force_loss:
        # Extract true forces and flatten them respecting atom counts
        F_true_flat = []
        F_pred_flat = []
        
        for i, atoms in enumerate(atoms_list):
            # True Forces
            f_true = atoms.get_forces()
            
            # Pred Forces (remove padding)
            n_atoms = len(atoms)
            f_pred = preds["forces"][i, :n_atoms, :]

            # Filter by atomic number if requested
            if atomic_numbers:
                mask = np.isin(atoms.get_atomic_numbers(), atomic_numbers)
                f_true = f_true[mask]
                f_pred = f_pred[mask]
            
            F_true_flat.append(f_true.reshape(-1))
            F_pred_flat.append(f_pred.reshape(-1))

        F_true_flat = np.concatenate(F_true_flat)
        F_pred_flat = np.concatenate(F_pred_flat)

        loss["forces"] = (F_pred_flat - F_true_flat) ** 2
        # Error in meV/A
        errors["forces"] = np.abs(F_true_flat - F_pred_flat) * 1000

    return preds, loss, errors


def inv_kernel_precomp(gTg, eps):
    lambd = (eps**2) * np.eye(gTg.shape[0])
    kernel = gTg + lambd
    # Using inv is standard for strictly regularized matrices, 
    # pinv can be used if stability issues persist.
    K_inv = np.linalg.inv(kernel)
    K_inv = (K_inv + K_inv.T) / 2.0
    return K_inv


def compute_energy_variance(g_cal, K_inv):
    # b batch, f features
    # Variance = diag(G @ K_inv @ G.T)
    variance = np.einsum("bf, fg, bg -> b", g_cal, K_inv, g_cal)
    return variance


def compute_forces_variance(g_cal, K_inv):
    # b batch, a atoms, x cartesian, f features
    variance = np.einsum("baxf, fg, baxg -> bax", g_cal, K_inv, g_cal)
    return variance


def is_psd(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False


def ensure_psd(hessian):
    """
    Inverts the Hessian matrix (gTg) with iterative regularization 
    until Positive Semi-Definite.
    """
    eps = 5e-4
    max_tries = 100
    
    for i in range(max_tries):
        lambd = (eps**2) * np.eye(hessian.shape[0])
        regularized_H = hessian + lambd
        
        if is_psd(regularized_H):
            K_inv = np.linalg.inv(regularized_H)
            K_inv = (K_inv + K_inv.T) / 2.0 # Ensure symmetry
            print(f"Inverted successfully with eps={eps}")
            return K_inv, eps
        
        eps *= 2.0
    
    # Fallback to pinv if Cholesky fails repeatedly
    print("Warning: Cholesky failed, falling back to pinv")
    lambd = (eps**2) * np.eye(hessian.shape[0])
    K_inv = np.linalg.pinv(hessian + lambd)
    return K_inv, eps


def calibrate_mse(K_inv, features, loss, weights, include_force_loss):
    # Energy Calibration
    var_E = compute_energy_variance(features["energy"], K_inv)
    # Avoid division by zero
    mask = var_E > 1e-12
    alpha2_E = np.mean(loss["energy"][mask] / var_E[mask])
    
    alpha2 = weights["energy"] * alpha2_E

    # Force Calibration
    if include_force_loss and "forces" in features:
        # We need to compute variance for the flattened valid forces used in 'loss["forces"]'
        # Since 'loss["forces"]' is already flattened and filtered in dataset_preds,
        # we must process features["forces"] to match that structure.
        
        # NOTE: features["forces"] is passed here as a list or array. 
        # For calibration, it's safer to re-calculate var_F on the fly or flatten beforehand.
        # Assuming features["forces"] is the raw (B, A, 3, F) array (or list of arrays).
        
        g_f = features["forces"]
        
        # Handle list vs array
        if isinstance(g_f, list):
            # List of (N_atoms_i, 3, F)
            g_f_flat = np.concatenate([g.reshape(-1, g.shape[-1]) for g in g_f], axis=0)
        else:
            # Array (B, A, 3, F) - likely needs masking if padded
            # For simplicity, if passed as array here, we flatten everything. 
            # In 'create_laplace_ensemble' we ensure strict handling.
            g_f_flat = g_f.reshape(-1, g_f.shape[-1])

        # Compute variance for every force component
        # (N_total, F) @ (F, F) @ (N_total, F).T -> diag
        var_F_flat = np.einsum("nf, fg, ng -> n", g_f_flat, K_inv, g_f_flat)
        
        # We assume loss["forces"] corresponds exactly to g_f_flat
        # (which it should if both are derived from the same validation set atoms)
        mask_f = var_F_flat > 1e-12
        alpha2_F = np.mean(loss["forces"][mask_f] / var_F_flat[mask_f])

        # If purely force driven or weighted
        f_alpha2 = weights["forces"] * alpha2_F
        
        # Heuristic: if force loss is included, often it dominates or is preferred
        alpha2 = f_alpha2

    return alpha2


def flatten_features(g_list_or_array, counts=None):
    """Flattens feature array/list to (N_observations, N_features)."""
    if isinstance(g_list_or_array, list):
        # List of (N_atoms, ..., F)
        return np.concatenate([g.reshape(-1, g.shape[-1]) for g in g_list_or_array], axis=0)
    elif counts is not None:
        # Padded array (B, MaxA, ..., F) with counts
        flat_list = []
        for i, n in enumerate(counts):
            valid = g_list_or_array[i, :n]
            flat_list.append(valid.reshape(-1, valid.shape[-1]))
        return np.concatenate(flat_list, axis=0)
    else:
        # Standard array, just flatten
        return g_list_or_array.reshape(-1, g_list_or_array.shape[-1])


def compute_forces_variance(g_cal, K_inv):
    # g_cal: (Batch, Atoms, 3, Features)
    # K_inv: (Features_Cov, Features_Cov)
    
    # Check for dimension mismatch (e.g. 63 vs 64)
    # This happens because Force derivatives w.r.t Bias are 0, so the feature is missing.
    f_dim = g_cal.shape[-1]
    cov_dim = K_inv.shape[-1]
    
    if f_dim != cov_dim:
        diff = cov_dim - f_dim
        if diff > 0:
            # Pad features with zeros (derivative w.r.t bias is 0)
            # Shape: (B, A, 3, F) -> (B, A, 3, F+diff)
            pad_width = ((0,0), (0,0), (0,0), (0, diff))
            g_cal = np.pad(g_cal, pad_width, mode='constant', constant_values=0)
        else:
            # If features are larger than cov, slice them
            g_cal = g_cal[..., :cov_dim]

    # b batch, a atoms, x cartesian, f features
    variance = np.einsum("baxf, fg, baxg -> bax", g_cal, K_inv, g_cal)
    return variance

def calc_list_variance(g_list, cov):
    vars_list = []
    
    # Pre-check dimensions
    cov_dim = cov.shape[-1]
    
    for g in g_list:
        # g is (N_obs, F)
        f_dim = g.shape[-1]
        
        if f_dim != cov_dim:
            diff = cov_dim - f_dim
            if diff > 0:
                # Pad with zeros: (N, F) -> (N, F+diff)
                g = np.pad(g, ((0,0), (0, diff)), mode='constant', constant_values=0)
            elif diff < 0:
                # Slice if g is too big
                g = g[:, :cov_dim]

        # Use optimal path for speed
        v = np.einsum("nf, fg, ng -> n", g, cov, g, optimize="optimal")
        vars_list.append(v)
        
    return np.concatenate(vars_list)



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
    np.random.seed(seed)
    config, params = restore_parameters(model_dir)
    
    # Store shallow params before canonicalization if needed
    shallow_params = unfreeze(params)

    # config.model.model_dump() etc.
    loss_config = [lossconf.model_dump() for lossconf in config.loss]
    
    # Extract Weights
    energy_weight = next((lc["weight"] for lc in loss_config if lc["name"] == "energy"), 0.0)
    forces_weight = next((lc["weight"] for lc in loss_config if lc["name"] == "forces"), 0.0)

    # Adjust weights based on flags
    w_E_hess = energy_weight**2
    w_F_hess = forces_weight**2 if force_contribution else 0.0
    
    loss_weights = {"energy": energy_weight, "forces": forces_weight}

    # --- Predictions ---
    preds_cal, loss_cal, errors_cal = dataset_preds(
        val_atoms, config, params, batch_size, atomic_numbers
    )
    preds_test, _, errors_test = dataset_preds(
        test_atoms, config, params, batch_size, atomic_numbers
    )

    # --- Feature Extraction ---
    params = canonicalize_energy_model_parameters(params)
    
    # Energy Features (B, F)
    g_train = dataset_features(train_atoms, config, params, batch_size, quantity="energy")
    g_cal = dataset_features(val_atoms, config, params, batch_size, quantity="energy")
    g_test = dataset_features(test_atoms, config, params, batch_size, quantity="energy")
    
    features_cal = {"energy": g_cal}
    features_test = {"energy": g_test}

    # --- Hessian Construction ---
    print("Constructing Energy Hessian...")
    H_E = g_train.T @ g_train
    H_total = w_E_hess * H_E
    print(H_E.shape)
    print(g_train.shape)

    features_cal["forces"] = None # Placeholder

    if force_variance:
        print("Computing Force Features...")
        # Force Features (B, A, 3, F) or padded
        gf_train = dataset_features(train_atoms, config, params, 1, quantity="forces")
        gf_cal = dataset_features(val_atoms, config, params, 1, quantity="forces")
        gf_test = dataset_features(test_atoms, config, params, 1, quantity="forces")

        # --- Filtering & Flattening ---
        # We must filter/flatten the force features to build the correct Hessian
        # H_F = Sum_i (J_i^T J_i) -> Flatten all valid force descriptors to (N_total, F)
        
        def process_forces(g_f, atoms_list):
            """Filters by atomic number and flattens to (N_obs, F)"""
            processed_list = []
            for i, atoms in enumerate(atoms_list):
                # Handle padding: slice valid atoms
                n_atoms = len(atoms)
                valid_gf = g_f[i, :n_atoms, :, :] # (A, 3, F)

                # Filter by atomic number
                if atomic_numbers is not None:
                    numbers = atoms.get_atomic_numbers()
                    mask = np.isin(numbers, atomic_numbers)
                    valid_gf = valid_gf[mask, :, :]
                
                # Flatten spatial dimensions: (A_filtered * 3, F)
                processed_list.append(valid_gf.reshape(-1, valid_gf.shape[-1]))
            return processed_list # List of arrays

        # Process Training Forces for Hessian
        gf_train_list = process_forces(gf_train, train_atoms)
        gf_train_flat = np.concatenate(gf_train_list, axis=0)
        
        if force_contribution:
            print(f"Constructing Force Hessian with {gf_train_flat.shape[0]} observations...")
            H_F = gf_train_flat.T @ gf_train_flat
            H_total += w_F_hess * H_F

            print(H_F.shape)
            print(gf_train_flat.shape)
            # quit()

        # Process Val/Test Forces for Variance/Calibration
        gf_cal_list = process_forces(gf_cal, val_atoms)
        features_cal["forces"] = gf_cal_list # Store as list for calibration

        gf_test_list = process_forces(gf_test, test_atoms)
        features_test["forces"] = gf_test_list

        # Store raw padded arrays for 'gf' output in results if needed
        # (Be careful, these might be huge)
        gf_train_raw = gf_train
        gf_cal_raw = gf_cal
        gf_test_raw = gf_test

    print("\n--- DIAGNOSTICS: HESSIAN MAGNITUDES ---")
    norm_E = np.linalg.norm(H_E)
    print(f"Energy Hessian Norm: {norm_E:.4e}")
    print(f"Energy Weight (sq):  {w_E_hess:.4e} -> Contribution: {norm_E * w_E_hess:.4e}")

    if force_contribution:
        norm_F = np.linalg.norm(H_F)
        print(f"Force Hessian Norm:  {norm_F:.4e}")
        print(f"Force Weight (sq):   {w_F_hess:.4e} -> Contribution: {norm_F * w_F_hess:.4e}")
        
        ratio = (norm_F * w_F_hess) / (norm_E * w_E_hess + 1e-9)
        print(f"Force/Energy Ratio:  {ratio:.4e}")
        
        if ratio < 1e-6:
            print("WARNING: Force contribution is numerically negligible!")
    
    print(f"Total Hessian Norm:  {np.linalg.norm(H_total):.4e}")
    print("---------------------------------------\n")

    # --- Inversion ---
    print("Inverting Total Hessian...")
    K_inv, eps = ensure_psd(H_total)

    # --- Calibration ---
    print("Calibrating...")
    alpha2 = calibrate_mse(
        K_inv,
        features_cal,
        loss_cal,
        loss_weights,
        include_force_loss,
    )
    covariance = alpha2 * K_inv
    print(f"Calibration Alpha^2: {alpha2}")

    # --- Inference ---
    variance_energy_cal = compute_energy_variance(features_cal["energy"], covariance)
    uncertainty_energy_cal = np.sqrt(variance_energy_cal)

    variance_energy_test = compute_energy_variance(features_test["energy"], covariance)
    uncertainty_energy_test = np.sqrt(variance_energy_test)

    uncertainty_forces_cal = None
    uncertainty_forces_test = None

    if force_variance:
        # Compute force variances on the specific valid atoms
        # Helper to compute variance for list of arrays
        # def calc_list_variance(g_list, cov):
        #     vars_list = []
        #     for g in g_list:
        #         # g is (N_obs, F)
        #         v = np.einsum("nf, fg, ng -> n", g, cov, g)
        #         vars_list.append(v)
        #     return np.concatenate(vars_list)

        var_forces_cal = calc_list_variance(features_cal["forces"], covariance)
        uncertainty_forces_cal = np.sqrt(var_forces_cal)

        var_forces_test = calc_list_variance(features_test["forces"], covariance)
        uncertainty_forces_test = np.sqrt(var_forces_test)

    # --- Sampling ---
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
        "covariance": covariance,
        "w_old": w_old,
        "w_new": w_new,
    }

    if force_variance:
        results["errors_forces_cal"] = errors_cal["forces"]
        results["errors_forces_test"] = errors_test["forces"]
        results["uncertainty_forces_cal"] = uncertainty_forces_cal
        results["uncertainty_forces_test"] = uncertainty_forces_test
        # Saving full feature matrices can be memory intensive; save if needed
        # results["gf"] = gf_train_raw 
        
    return results
