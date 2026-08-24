import time

import jax
import jax.numpy as jnp

from radiax import roots

from .testsets import RANDOM_CUBICS


def test_speedup():
    vmapped_jnp_roots = jax.jit(jax.vmap(lambda p: jnp.roots(p, strip_zeros=False)))
    vmapped_radiax_roots = jax.jit(jax.vmap(lambda p: roots(p, strip_zeros=False)))

    vmapped_jnp_roots(RANDOM_CUBICS).block_until_ready()
    vmapped_radiax_roots(RANDOM_CUBICS).block_until_ready()

    start = time.perf_counter()
    vmapped_jnp_roots(RANDOM_CUBICS).block_until_ready()
    time_jnp = time.perf_counter() - start

    start = time.perf_counter()
    vmapped_radiax_roots(RANDOM_CUBICS).block_until_ready()
    time_radiax = time.perf_counter() - start

    assert time_radiax < 0.5 * time_jnp
