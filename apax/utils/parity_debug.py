"""Runtime gate for the MACE layer-by-layer parity harness's ``sow`` calls.

Several apax modules emit ``self.sow("debug", ...)`` calls that the parity
harness in ``scripts/mace_layer_parity.py`` consumes via
``apply(..., mutable=["debug"])``. With Flax, ``init`` defaults to making
*all* collections mutable, which causes the ``debug`` collection to leak
into the parameter pytree returned by any ``model.init(...)`` call that
doesn't explicitly pass ``mutable=DenyList("debug")`` — and that pytree
then can't be checkpointed (orbax sees ``LinearizeTracer`` /
``BatchTracer`` values when a downstream caller wraps init in
``jax.grad``/``jax.vmap``).

Rather than require every caller of ``model.init`` / ``model.apply`` to
remember to pass ``DenyList("debug")``, we make the sow calls themselves
opt-in. The harness sets the flag (or uses the :func:`parity_debug`
context manager) before running the model; everywhere else the sows are
no-ops.

Usage in module code::

    from apax.utils.parity_debug import is_parity_debug_enabled

    if is_parity_debug_enabled():
        self.sow("debug", "scale_shift_out", E_i)

Usage in the harness::

    from apax.utils.parity_debug import parity_debug

    with parity_debug():
        (energy, _), sown = energy_model.apply(
            params, ..., mutable=["debug"]
        )
"""

from __future__ import annotations

from contextlib import contextmanager

_PARITY_DEBUG: bool = False


def is_parity_debug_enabled() -> bool:
    """Return whether the parity-harness ``debug`` sow gate is open.

    Returns
    -------
    bool
        ``True`` only inside a :func:`parity_debug` context (or after a
        manual :func:`set_parity_debug` call). ``False`` everywhere else,
        which makes the ``self.sow("debug", ...)`` sites no-ops so the
        ``debug`` collection never appears in the params pytree returned
        by ``model.init``.
    """
    return _PARITY_DEBUG


def set_parity_debug(enabled: bool) -> None:
    """Set the parity-debug gate explicitly.

    Prefer the :func:`parity_debug` context manager unless you need to
    enable the gate from a long-lived setup hook.

    Parameters
    ----------
    enabled : bool
        New value of the module-level gate.
    """
    global _PARITY_DEBUG
    _PARITY_DEBUG = enabled


@contextmanager
def parity_debug():
    """Context manager that opens the parity-debug sow gate.

    Yields
    ------
    None
        Inside the ``with`` block, :func:`is_parity_debug_enabled` returns
        ``True`` and ``self.sow("debug", ...)`` calls fire. The gate is
        restored to its previous value on exit, even if the body raises.

    Examples
    --------
    >>> from apax.utils.parity_debug import parity_debug
    >>> with parity_debug():
    ...     (energy, _), sown = energy_model.apply(
    ...         params, R, Z, idx, box, offsets, mutable=["debug"],
    ...     )
    """
    global _PARITY_DEBUG
    prev = _PARITY_DEBUG
    _PARITY_DEBUG = True
    try:
        yield
    finally:
        _PARITY_DEBUG = prev
