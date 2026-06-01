"""The non-residual first-layer interaction block is intentionally unported."""

from apax.transfer_learning.mace_foundation import _TORCH_TO_APAX_INTERACTION


def test_non_residual_first_block_intentionally_unsupported():
    # Every MACE foundation in scope (small/medium/MPA-0) is Residual-first, so
    # the plain RealAgnosticInteractionBlock math is unused (YAGNI). Converting a
    # foundation that uses it must fail loudly rather than silently mis-map.
    assert "RealAgnosticInteractionBlock" not in _TORCH_TO_APAX_INTERACTION
    # the residual / density variants the shipped foundations use are supported
    assert "RealAgnosticResidualInteractionBlock" in _TORCH_TO_APAX_INTERACTION
    assert "RealAgnosticDensityInteractionBlock" in _TORCH_TO_APAX_INTERACTION
