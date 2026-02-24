import jax
import jax.numpy as jnp
import jax.lax as lax
import time
from jax.experimental import pallas as pl
import numpy as np

# ----------------------------
# Data setup
# ----------------------------
a, r, s, i, j, k = 80000, 6, 6, 3, 3, 3
key = jax.random.PRNGKey(0)
moments3 = jax.random.normal(key, (a, r, i, j, k))

# ----------------------------
# Vanilla einsum contraction
# ----------------------------
@jax.jit
def vanilla_contraction(moments):
    return jnp.einsum('arijk, asijk -> ars', moments, moments)

# ----------------------------
# Chunked version
# ----------------------------
@jax.jit
def chunked_contraction(moments, chunk_size=8000):
    results = []
    for start in range(0, moments.shape[0], chunk_size):
        end = min(start + chunk_size, moments.shape[0])
        chunk = moments[start:end]
        chunk_flat = chunk.reshape(chunk.shape[0], chunk.shape[1], -1)
        res = jnp.einsum('ari, asi -> ars', chunk_flat, chunk_flat)
        results.append(res)
    return jnp.concat(results, axis=0)

@jax.jit
def chunked_contraction2(moments, chunk_size=8000):
    results = []
    for start in range(0, moments.shape[0], chunk_size):
        end = min(start + chunk_size, moments.shape[0])
        chunk = moments[start:end]
        # chunk_flat = chunk.reshape(chunk.shape[0], chunk.shape[1], -1)
        res = jnp.einsum('arijk, asijk -> ars', chunk, chunk)
        results.append(res)
    return jnp.concat(results, axis=0)

# ----------------------------
# Benchmarking
# ----------------------------
def benchmark():
    print("Warming up JAX JIT compilation...")
    vanilla_contraction(moments3).block_until_ready()
    chunked_contraction(moments3).block_until_ready()
    chunked_contraction2(moments3).block_until_ready()

    print("\nRunning benchmarks:")

    # with jax.profiler.trace("tensorboard"):

    # jax.profiler.start_trace("/tmp/tensorboard", True, True)

    start = time.time()
    res1 = vanilla_contraction(moments3).block_until_ready()
    end = time.time()
    print(f"Vanilla einsum time: {end - start:.6f} s")

    time.sleep(2)

    start = time.time()
    res2 = chunked_contraction(moments3).block_until_ready()
    end = time.time()
    print(f"Chunked contraction time: {end - start:.6f} s")
    chunked_contraction(moments3).block_until_ready()

    time.sleep(2)

    start = time.time()
    res2 = chunked_contraction2(moments3).block_until_ready()
    end = time.time()
    print(f"Chunked 2 contraction time: {end - start:.4f} s")

    # jax.profiler.stop_trace()
    jax.profiler.save_device_memory_profile("memory.prof")


    # chunked_contraction_scan(moments3).block_until_ready()
    # start = time.time()
    # res2 = chunked_contraction_scan(moments3).block_until_ready()
    # end = time.time()
    # print(f"scan contraction time: {end - start:.4f} s")

    # pallas_contraction(moments3).block_until_ready()
    # start = time.time()
    # res3 = pallas_contraction(moments3).block_until_ready()
    # end = time.time()
    # print(f"Pallas contraction time: {end - start:.4f} s")

    # Check if results are close (should be for numerical sanity)
    print("\nSanity check:")
    print("Chunked close to vanilla:", np.allclose(res1, res2, rtol=1e-3, atol=1e-5))
    # print("Pallas close to vanilla:", np.allclose(res1, res3, rtol=1e-3, atol=1e-5))

if __name__ == "__main__":
    benchmark()