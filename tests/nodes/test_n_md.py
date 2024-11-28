import os
import pathlib
import shutil
import subprocess

import zntrack

from apax.nodes.md import ApaxJaxMD
from apax.nodes.model import Apax
from apax.nodes.utils import AddData

CONFIG_PATH = pathlib.Path(__file__).parent / "example.yaml"
MD_CONFIG_PATH = pathlib.Path(__file__).parent / "md.yaml"


def test_n_jax_md(tmp_path, get_md22_stachyose):
    os.chdir(tmp_path)
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["dvc", "init"], check=True)
    shutil.copy(CONFIG_PATH, tmp_path / "example.yaml")
    shutil.copy(MD_CONFIG_PATH, tmp_path / "md.yaml")
    proj = zntrack.Project()
    with proj:
        data = AddData(file=get_md22_stachyose, stop=100)
        model = Apax(data=data.frames, validation_data=data.frames, config="example.yaml")
        md = ApaxJaxMD(model=model, config="md.yaml", data=data.frames)

    proj.repro()

    md = md.from_rev()
    assert len(md.frames) == 50
