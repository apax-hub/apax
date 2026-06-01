"""Fine-tune the converted MACE-MP-0 small on a synthetic dataset.

Verifies that the standard apax `TransferLearningConfig` path works on a
converted MACE backbone - no MACE-specific trainer code is needed. Gated
by ``mace_parity`` because conversion requires torch + mace-torch.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.mace_parity, pytest.mark.slow]


def _tiny_dataset(path: Path, n_frames: int = 12) -> Path:
    """Write a small extxyz dataset with energies and forces.

    Parameters
    ----------
    path : Path
        Output ``.extxyz`` file path.
    n_frames : int
        Number of frames to generate.

    Returns
    -------
    Path
        ``path`` for chaining.
    """
    from ase import Atoms
    from ase.io import write

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n_frames):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        ) + 0.05 * rng.normal(size=(3, 3))
        atoms = Atoms(symbols=["O", "H", "H"], positions=positions, pbc=False)
        atoms.info["energy"] = float(rng.normal(loc=-14.0, scale=0.2))
        atoms.arrays["forces"] = 0.1 * rng.normal(size=(3, 3))
        frames.append(atoms)
    write(path, frames)
    return path


def test_finetune_converted_small_runs_end_to_end(tmp_path):
    """Convert MACE-MP-0 small, then fine-tune through TransferLearningConfig.

    This exercises the standard apax training entry point with a converted
    foundation model as the base checkpoint. Asserts that:

    1. Training runs to completion for a single epoch on a tiny synthetic
       dataset (no NaNs, no crashes).
    2. The resulting experiment directory has the apax-native layout
       (``config.yaml`` + ``best/``) and ``restore_parameters`` reads it back.
    3. The fine-tuned config still describes a MACE model.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("yaml")
    import yaml

    from apax.train.checkpoints import restore_parameters
    from apax.train.run import run
    from apax.transfer_learning.mace_foundation import run_conversion

    # 1. Convert the foundation backbone.
    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")
    assert (converted / "config.yaml").is_file()
    assert (converted / "best").is_dir()

    # 2. Synthesize a tiny dataset.
    ds_path = _tiny_dataset(tmp_path / "ds.extxyz")

    # 3. Build a minimal fine-tune config dict.
    cfg = {
        "n_epochs": 1,
        "seed": 1,
        "data": {
            "directory": str(tmp_path),
            "experiment": "ft_smoke",
            "data_path": str(ds_path),
            "n_train": 8,
            "n_valid": 4,
            "batch_size": 2,
            "valid_batch_size": 2,
        },
        "model": {
            "name": "mace",
            "basis": {
                "name": "bessel",
                "variant": "standard",
                "n_basis": 10,
                "r_max": 6.0,
            },
            "radial_embedding": {
                "num_polynomial_cutoff": 5,
                "distance_transform": None,
            },
            "descriptor": {
                "max_ell": 3,
                "hidden_irreps": "128x0e",
                "correlation": 3,
                "interactions": [
                    {"name": "RealAgnosticResidual"},
                    {"name": "RealAgnosticResidual"},
                ],
                "avg_num_neighbors": 1.0,
            },
            "readout": {"MLP_irreps": "16x0e"},
            # Float64 throughout to match the converter dump.
            "descriptor_dtype": "fp64",
            "readout_dtype": "fp64",
            "scale_shift_dtype": "fp64",
        },
        "transfer_learning": {
            "base_model_checkpoint": str(converted),
            "reset_layers": [],
        },
        "loss": [
            {"name": "energy"},
            {"name": "forces"},
        ],
        "optimizer": {"nn_lr": 1e-4, "emb_lr": 1e-4},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # 4. Train one epoch.
    run(cfg_path, log_level="warning")

    # 5. Verify experiment dir contains apax training output.
    ft_dir = tmp_path / "ft_smoke"
    assert (ft_dir / "config.yaml").is_file()
    assert (ft_dir / "best").is_dir()

    restored_cfg, restored_params = restore_parameters(ft_dir)
    assert restored_cfg.model.name == "mace"
    assert len(restored_cfg.model.descriptor.interactions) == 2

    import jax

    leaves = jax.tree_util.tree_leaves(restored_params)
    assert len(leaves) > 0
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves), (
        "Fine-tuned params contain NaN or inf"
    )


def _mace_finetune_cfg(
    tmp_path,
    converted,
    ds_path,
    *,
    experiment: str,
    n_members: int,
    reset_layers: list,
):
    """Build the inline fine-tune config dict shared by the ensemble tests.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for experiment output.
    converted : Path
        Path to the converted MACE checkpoint directory.
    ds_path : Path
        Path to the dataset extxyz file.
    experiment : str
        Experiment name used as the output subdirectory.
    n_members : int
        Number of shallow ensemble members.
    reset_layers : list
        List of parameter paths to mark as freshly initialized.

    Returns
    -------
    dict
        Config dict suitable for ``yaml.safe_dump`` and ``apax.train.run.run``.
    """
    return {
        "n_epochs": 1,
        "seed": 1,
        "data": {
            "directory": str(tmp_path),
            "experiment": experiment,
            "data_path": str(ds_path),
            "n_train": 8,
            "n_valid": 4,
            "batch_size": 2,
            "valid_batch_size": 2,
        },
        "model": {
            "name": "mace",
            "basis": {
                "name": "bessel",
                "variant": "standard",
                "n_basis": 10,
                "r_max": 6.0,
            },
            "radial_embedding": {
                "num_polynomial_cutoff": 5,
                "distance_transform": None,
            },
            "descriptor": {
                "max_ell": 3,
                "hidden_irreps": "128x0e",
                "correlation": 3,
                "interactions": [
                    {"name": "RealAgnosticResidual"},
                    {"name": "RealAgnosticResidual"},
                ],
                "avg_num_neighbors": 1.0,
            },
            "readout": {"MLP_irreps": "16x0e"},
            "ensemble": {
                "kind": "shallow",
                "n_members": n_members,
                "force_variance": True,
            },
            "descriptor_dtype": "fp64",
            "readout_dtype": "fp64",
            "scale_shift_dtype": "fp64",
        },
        "transfer_learning": {
            "base_model_checkpoint": str(converted),
            "reset_layers": reset_layers,
        },
        "loss": [
            {"name": "energy", "loss_type": "crps"},
            {"name": "forces", "loss_type": "crps"},
        ],
        "optimizer": {"nn_lr": 1e-4, "emb_lr": 1e-4},
    }


def test_finetune_foundation_to_ensemble_errors_when_reset_layers_empty(tmp_path):
    """Foundation→shallow-ensemble fine-tune raises a guided structural-mismatch error.

    e3nn encodes the output irreps in the parameter dict key, so widening
    ``n_shallow_ensemble`` from 0 (foundation) to 4 (target) produces sibling
    keys ``w[0,0] AxBe,1x0e`` vs ``w[0,0] AxBe,4x0e`` under the same parent
    path. ``black_list_param_transfer`` must detect this as a structural
    mismatch and raise ``TransferLearningShapeMismatchError`` with the parent
    path and both leaf names in the message.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("yaml")
    import yaml

    from apax.train.run import run
    from apax.transfer_learning import TransferLearningShapeMismatchError
    from apax.transfer_learning.mace_foundation import run_conversion

    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")

    ds_path = _tiny_dataset(tmp_path / "ds.extxyz")

    cfg = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_no_reset",
        n_members=4,
        reset_layers=[],
    )
    cfg_path = tmp_path / "cfg_no_reset.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        run(cfg_path, log_level="warning")

    msg = str(excinfo.value)
    # The structural-mismatch section must mention the readout parent and
    # the target leaf with the new (M, 4) shape.
    assert "readout" in msg
    assert "(128, 4)" in msg or "(16, 4)" in msg, (
        f"expected target shape with n_members=4 in error message; got:\n{msg}"
    )
    assert "reset_layers:" in msg


def test_finetune_foundation_to_ensemble_succeeds_with_suggested_reset_layers(
    tmp_path,
):
    """Pasting the suggested target keys into ``reset_layers`` lets training proceed."""
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    pytest.importorskip("yaml")
    import re

    import yaml

    from apax.train.checkpoints import restore_parameters
    from apax.train.run import run
    from apax.transfer_learning import TransferLearningShapeMismatchError
    from apax.transfer_learning.mace_foundation import run_conversion

    converted = tmp_path / "converted" / "mace-mp-0-small.apax"
    run_conversion("small", converted, head="default", family="mace_mp")

    ds_path = _tiny_dataset(tmp_path / "ds.extxyz")

    # First run: collect the suggested reset_layers from the error message.
    cfg = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_collect",
        n_members=4,
        reset_layers=[],
    )
    cfg_path = tmp_path / "cfg_collect.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(TransferLearningShapeMismatchError) as excinfo:
        run(cfg_path, log_level="warning")

    # The reset_layers: bullets contain spaces (e3nn key names like
    # "w[0,0] 128x0e,4x0e"), so the path regex must run to end-of-line.
    suggested = re.findall(
        r"^\s*-\s+(params/.+?)\s*$",
        str(excinfo.value),
        flags=re.MULTILINE,
    )
    assert suggested, (
        f"error message did not include yaml-ready bullet list; got:\n{excinfo.value}"
    )

    # Second run: paste the suggested keys and train one epoch.
    cfg2 = _mace_finetune_cfg(
        tmp_path,
        converted,
        ds_path,
        experiment="ft_ensemble_with_reset",
        n_members=4,
        reset_layers=suggested,
    )
    cfg2_path = tmp_path / "cfg_with_reset.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2))

    run(cfg2_path, log_level="warning")

    ft_dir = tmp_path / "ft_ensemble_with_reset"
    assert (ft_dir / "config.yaml").is_file()
    assert (ft_dir / "best").is_dir()

    restored_cfg, restored_params = restore_parameters(ft_dir)
    assert restored_cfg.model.name == "mace"
    assert restored_cfg.model.ensemble is not None
    assert restored_cfg.model.ensemble.n_members == 4

    import jax

    leaves = jax.tree_util.tree_leaves(restored_params)
    # At least one leaf has the new (M, 4) ensemble width on the trailing axis.
    has_ensemble_axis = any(
        getattr(leaf, "shape", ()) and leaf.shape[-1] == 4 for leaf in leaves
    )
    assert has_ensemble_axis, "no leaf reflects n_members=4 ensemble axis"
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves), (
        "Fine-tuned params contain NaN or inf"
    )
