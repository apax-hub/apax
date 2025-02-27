import pathlib

import jax

from apax.config.common import parse_config
from apax.utils.data import make_minimal_input

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_builder_feature_model():
    R, Z, idx, box, offsets = make_minimal_input()

    config = parse_config(TEST_PATH / "train.yaml")

    Builder = config.model.get_builder()
    builder = Builder(config.model.model_dump())

    model = builder.build_energy_model()
    key = jax.random.PRNGKey(0)
    params = model.init(key, R, Z, idx, box, offsets)

    model = builder.build_feature_model(only_use_n_layers=0, init_box=box)
    out = model.apply(params, R, Z, idx, box, offsets)
    assert out.shape == (360,)

    model = builder.build_feature_model(only_use_n_layers=1, init_box=box)
    out = model.apply(params, R, Z, idx, box, offsets)
    assert out.shape == (128,)

    model = builder.build_feature_model(only_use_n_layers=2, init_box=box)
    out = model.apply(params, R, Z, idx, box, offsets)
    assert out.shape == (64,)

    model = builder.build_feature_model(only_use_n_layers=3, init_box=box)
    out = model.apply(params, R, Z, idx, box, offsets)
    assert out.shape == (32,)
