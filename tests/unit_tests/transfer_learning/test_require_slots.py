"""_require_slots: actionable errors when the MACE module tree drifts.

The foundation converter maps torch weights into auto-generated Flax module
slots (``InteractionBlock_k`` etc.). If those names change, the converter must
fail with a message naming the missing slot, not a bare KeyError deep in a loop.
"""

import pytest

from apax.transfer_learning.mace_foundation import _require_slots


def test_require_slots_returns_ordered_blocks():
    params = {
        "InteractionBlock_0": {"a": 1},
        "InteractionBlock_1": {"a": 2},
        "other": {},
    }
    slots = _require_slots(params, "InteractionBlock_", 2, context="interactions")
    assert slots == [{"a": 1}, {"a": 2}]
    # the returned dicts are the same objects, so in-place mutation by the
    # mapping functions still updates the parameter tree
    assert slots[0] is params["InteractionBlock_0"]


def test_require_slots_raises_actionable_error_on_missing_slot():
    params = {"InteractionBlock_0": {}}  # InteractionBlock_1 missing
    with pytest.raises(KeyError) as exc:
        _require_slots(params, "InteractionBlock_", 2, context="interactions")
    msg = str(exc.value)
    assert "InteractionBlock_1" in msg
    assert "converter" in msg.lower()
