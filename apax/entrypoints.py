from apax.nodes import __all__


def nodes() -> dict[str, list[str]]:
    """Return all available nodes."""
    return {"apax.nodes": list(__all__)}
