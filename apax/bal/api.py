from functools import partial
from typing import List, Union

import jax
import numpy as np
from ase import Atoms
from click import Path
from tqdm import trange

from apax.bal import feature_maps, kernel, selection, transforms
from apax.data.input_pipeline import TFPipeline
from apax.model.builder import ModelBuilder
from apax.model.gmnn import EnergyModel
from apax.train.checkpoints import restore_parameters
from apax.train.run import RawDataset, initialize_dataset


def create_feature_fn(
    model: EnergyModel,
    params,
    base_feature_map,
    feature_transforms=[],
    is_ensemble: bool = False,
):
    """
    Converts a model into a feature map and transforms it as needed and
    sets it up for use in copmuting the features of a dataset.

    All transformations are applied on the feature function, not on computed features.
    Only the final function is jit compiled.
    """
    feature_fn = base_feature_map.apply(model)

    if is_ensemble:
        feature_fn = transforms.ensemble_features(feature_fn)

    for transform in feature_transforms:
        feature_fn = transform.apply(feature_fn)

    feature_fn = transforms.batch_features(feature_fn)
    feature_fn = partial(feature_fn, params)
    feature_fn = jax.jit(feature_fn)
    return feature_fn


def compute_features(feature_fn, dataset: TFPipeline, processing_batch_size: int):
    """Compute the features of a dataset."""
    features = []
    n_data = dataset.n_data
    ds = dataset.batch(processing_batch_size)

    pbar = trange(n_data, desc="Computing features", ncols=100, leave=True)
    for i, (inputs, _) in enumerate(ds):
        g = feature_fn(inputs)
        features.append(np.asarray(g))
        pbar.update(g.shape[0])
    pbar.close()

    features = np.concatenate(features, axis=0)
    return features


def kernel_selection(
    model_dir: Union[Path, List[Path]],
    train_atoms: List[Atoms],
    pool_atoms: List[Atoms],
    base_fm_options: dict,
    selection_method: str,
    feature_transforms: list = [],
    selection_batch_size: int = 10,
    processing_batch_size: int = 64,
):
    n_models = 1 if isinstance(model_dir, (Path, str)) else len(model_dir)
    is_ensemble = n_models > 1

    selection_fn = {
        "max_dist": selection.max_dist_selection,
    }[selection_method]

    base_feature_config = feature_maps.FeatureMapOptions.model_validate(base_fm_options)
    base_feature_map = base_feature_config.base_feature_map

    config, params = restore_parameters(model_dir)

    n_train = len(train_atoms)
    dataset = initialize_dataset(config, RawDataset(atoms_list=train_atoms + pool_atoms))

    init_box = dataset.init_input()["box"][0]

    builder = ModelBuilder(config.model.get_dict(), n_species=119)
    model = builder.build_energy_model(apply_mask=True, init_box=init_box)

    feature_fn = create_feature_fn(
        model, params, base_feature_map, feature_transforms, is_ensemble
    )
    g = compute_features(feature_fn, dataset, processing_batch_size)
    hm = kernel.KernelMatrix(g, n_train)
    new_indices = selection_fn(hm, selection_batch_size)

    return new_indices
