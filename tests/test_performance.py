import time

import jax
import jax.numpy as jnp

from radiax import roots

from .testsets import RANDOM_CUBICS


def test_speedup():
    jitted_jnp_roots = jax.jit(jax.vmap(lambda p: jnp.roots(p, strip_zeros=False)))
    jitted_radiax_roots = jax.jit(roots, static_argnames=("strip_zeros", "real"))

    jitted_jnp_roots(RANDOM_CUBICS).block_until_ready()
    jitted_radiax_roots(RANDOM_CUBICS, strip_zeros=False).block_until_ready()

    start = time.perf_counter()
    jitted_jnp_roots(RANDOM_CUBICS).block_until_ready()
    time_jnp = time.perf_counter() - start

    start = time.perf_counter()
    jitted_radiax_roots(RANDOM_CUBICS, strip_zeros=False).block_until_ready()
    time_radiax = time.perf_counter() - start

    assert time_radiax < 0.15 * time_jnp
