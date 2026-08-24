import jax
import jax.numpy as jnp

KNOWN_ROOTS = [
    # Linear
    (
        jnp.array([2, -8]),
        jnp.array([4]),
    ),
    # Quadratic
    (
        jnp.array([1, -3, 2]),
        jnp.array([1, 2]),
    ),
    (
        jnp.array([1, 0, 1]),
        jnp.array([1j, -1j]),
    ),
    (
        jnp.array([1 + 1j, -3 + 1j]),
        jnp.array([1 - 2j]),
    ),
    (
        jnp.array([1, -3 + 2j, 5 - 1j]),
        jnp.array([1 + 1j, 2 - 3j]),
    ),
    # Cubic
    (
        jnp.array([1, -6, 11, -6]),
        jnp.array([1, 2, 3]),
    ),
    (
        jnp.array([1, 0, 0, 1]),
        jnp.array([-1, 0.5 + 0.5j * jnp.sqrt(3), 0.5 - 0.5j * jnp.sqrt(3)]),
    ),
    (
        jnp.array([1, 0, -3, 2]),
        jnp.array([-2, 1, 1]),
    ),
]


RANDOM_CUBICS = jax.random.uniform(jax.random.key(0), (10_000, 4), minval=1, maxval=10)
