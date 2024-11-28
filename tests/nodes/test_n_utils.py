import os
import subprocess

import ase.io
import zntrack

from apax.nodes.utils import AddData


def test_add_data(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["dvc", "init"], check=True)
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose, stop=50)

    proj.repro()
    data = data.from_rev()
    assert isinstance(data.frames, list)
    assert len(data.frames) == 50
    assert all(isinstance(atoms, ase.Atoms) for atoms in data.frames)
