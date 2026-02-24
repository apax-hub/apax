# Project: Apax (MLIPs with JAX)

## Coding Standards
- **Framework:** We use JAX and a reduced internal version of JaxMD.
- **Functional Programming:** prefer pure functions. Avoid side effects.
- **Style:** Follow PEP 8 but allow for JAX-specific conventions (e.g., `def fn(x, y)` is fine for vmapping).

## Common Pitfalls to Avoid
- Watch out for 64-bit precision; we default to float32 for performance unless specified. Especially positions, cells and the final predictions need to be in float 64.
- Remember that `jax.jit` cannot handle dynamic shapes.

## Testing
- Run tests using `uv run coverage run -m pytest -k "not slow"`.
