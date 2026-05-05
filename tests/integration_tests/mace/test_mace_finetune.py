"""Fine-tune the converted MACE-MP-0 small on a synthetic dataset.

Verifies that the standard apax `TransferLearningConfig` path works on a
converted MACE backbone - no MACE-specific trainer code is needed. Gated
by ``mace_parity`` because conversion requires torch + mace-torch.

The fine-tune target keeps ``readout_kind="mace"`` to match the converter
output's pytree structure; ``black_list_param_transfer`` cannot transfer
into a head-swapped target (e.g. ``readout_kind="standard"``) because the
source's ``readout_*`` keys have no slot in the target's
``dense_*`` head and would error out at the optimizer mask step. Plumbing
a head-swap through transfer learning would require either reset_layers
support for source-side keys or a more permissive transfer mode in
``apax.transfer_learning.parameter_transfer`` - flagged as a follow-up.
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
                "use_cueq": False,
            },
            # Match the converter output's readout to keep pytrees
            # structurally identical; head-swapping (readout.kind=standard)
            # is incompatible with the current black_list_param_transfer.
            "readout": {"kind": "mace", "MLP_irreps": "16x0e"},
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
    assert restored_cfg.model.readout.kind == "mace"
    assert len(restored_cfg.model.descriptor.interactions) == 2

    import jax

    leaves = jax.tree_util.tree_leaves(restored_params)
    assert len(leaves) > 0
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves), (
        "Fine-tuned params contain NaN or inf"
    )
