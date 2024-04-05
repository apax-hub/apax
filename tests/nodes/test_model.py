from apax.nodes.utils import AddData
from apax.nodes.model import Apax
import zntrack, os
import pathlib
import shutil

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"

def test_add_data(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    shutil.copy(CONFIG_PATH, tmp_path)
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose)
        model = Apax(data=data.atoms, validation_data=data.atoms, config="config.yaml")

    proj.run()

    model = model.from_rev()
    data = data.from_rev()

    atoms = data.atoms[0]
    atoms.calc = model.get_calculator()

    assert atoms.get_potential_energy() < 0
