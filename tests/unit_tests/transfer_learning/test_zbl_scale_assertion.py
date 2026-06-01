"""ZBL ``output_scale`` assumes a single global scalar.

The ZBL fold (``apax/transfer_learning/mace_foundation.py``) reads
``model.scale_shift.scale`` and casts to a Python float. For multi-element
scale tensors this silently uses only the first element. Pin the
assumption: raise ``NotImplementedError`` on per-element scales.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.mace_parity


def test_extract_config_rejects_per_element_scale():
    """Multi-element ``scale_shift.scale`` triggers a clear NotImplementedError.

    Constructs a torch-mace-shaped fake whose ``scale_shift.scale`` has two
    distinct elements, and runs the head-resolution path far enough to hit
    the ZBL fold. The function should raise ``NotImplementedError`` with a
    "per-element" message before it casts ``scale`` to a Python float.
    """
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    import torch as _torch

    # torch >=2.6 sets ``weights_only=True`` by default, which trips on the
    # ``slice`` global baked into e3nn's precomputed Wigner constants. Allow-
    # list it so ``from e3nn import o3`` succeeds; the constants ship with
    # e3nn itself and are safe.
    _torch.serialization.add_safe_globals([slice])
    import e3nn.o3  # noqa: F401, PLC0415  # ensure e3nn.o3 is loaded

    from apax.transfer_learning.mace_foundation import _extract_config_from_torch

    class _PairRep(_torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("p", _torch.tensor(4.0))
            self.a_exp = _torch.nn.Parameter(_torch.tensor(1.0))
            self.a_prefactor = _torch.nn.Parameter(_torch.tensor(1.0))

    # Class name must match a key in ``_SUPPORTED_TORCH_INTERACTION_CLS``;
    # ``type(inter).__name__`` is what the converter checks.
    class RealAgnosticDensityInteractionBlock(_torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.avg_num_neighbors = 10.0

    fake = SimpleNamespace(
        heads=["default"],
        interactions=[RealAgnosticDensityInteractionBlock()],
        products=[
            SimpleNamespace(
                linear=SimpleNamespace(irreps_out="32x0e"),
                symmetric_contractions=SimpleNamespace(contractions=[SimpleNamespace()]),
            )
        ],
        spherical_harmonics=SimpleNamespace(irreps_out="1x0e + 1x1o"),
        radial_embedding=SimpleNamespace(
            bessel_fn=SimpleNamespace(
                bessel_weights=_torch.zeros(8),
            ),
        ),
        r_max=_torch.tensor(6.0),
        scale_shift=SimpleNamespace(
            scale=_torch.tensor([1.0, 2.0]),  # <- per-element, two distinct values
        ),
        pair_repulsion_fn=_PairRep(),
    )

    with pytest.raises(NotImplementedError, match="per-element"):
        _extract_config_from_torch(fake, head=None)
