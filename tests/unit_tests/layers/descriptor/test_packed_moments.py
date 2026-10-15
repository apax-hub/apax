import numpy as np

from apax.layers.descriptor.moments import geometric_moments, geometric_moments_packed


def test_packed_moments_match_full():
    rng = np.random.default_rng(0)
    dn = rng.normal(size=(40, 3))
    dn /= np.linalg.norm(dn, axis=1, keepdims=True)
    rf = rng.normal(size=(40, 5))
    idx = np.repeat(np.arange(8), 5)
    full = geometric_moments(rf, dn, idx, 8)
    packed = geometric_moments_packed(rf, dn, idx, 8)
    for a, b in zip(full, packed):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
