import os

import ase.io
import zntrack

from apax.nodes.utils import AddData


def test_add_data(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose)

    proj.repro()
    data = data.from_rev()
    assert isinstance(data.atoms, list)
    assert len(data.atoms) == 50
    assert all(isinstance(atoms, ase.Atoms) for atoms in data.atoms)
