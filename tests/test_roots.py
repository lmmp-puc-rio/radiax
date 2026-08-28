import jax
import jax.numpy as jnp
import pytest

from radiax import roots

from .testsets import KNOWN_ROOTS, RANDOM_CUBICS
from .utils import assert_roots_match


@pytest.mark.parametrize(("p", "expected"), KNOWN_ROOTS)
@pytest.mark.parametrize(
    ("dtype", "real", "expected_dtype"),
    [
        (jnp.float64, False, jnp.complex128),
        (jnp.float64, True, jnp.float64),
        (jnp.complex128, False, jnp.complex128),
        (jnp.float32, False, jnp.complex64),
        (jnp.float32, True, jnp.float32),
        (jnp.complex64, False, jnp.complex64),
        (jnp.int64, False, jnp.complex128),
        (jnp.int64, True, jnp.float64),
        (jnp.int32, False, jnp.complex64),
        (jnp.int32, True, jnp.float32),
    ],
)
@pytest.mark.parametrize(
    ("strip_zeros", "jit"), [(True, False), (False, False), (False, True)]
)
def test_roots(
    p: jax.Array,
    expected: jax.Array,
    dtype: jnp.dtype,
    expected_dtype: jnp.dtype,
    strip_zeros: bool,
    real: bool,
    jit: bool,
) -> None:
    if not jnp.issubdtype(dtype, jnp.complexfloating) and jnp.issubdtype(
        p.dtype, jnp.complexfloating
    ):
        pytest.skip(
            f"Skipping test for dtype {dtype} because it cannot represent complex numbers"
        )
    p = p.astype(dtype)

    for nzeros in range((6 if strip_zeros else 5) - p.size):
        p_with_zeros = jnp.append(jnp.zeros(nzeros, dtype=p.dtype), p)

        if jit:
            r = jax.jit(lambda p: roots(p, real=real, strip_zeros=strip_zeros))(
                p_with_zeros
            )
        else:
            r = roots(p_with_zeros, strip_zeros=strip_zeros, real=real)

        assert isinstance(r, jax.Array)
        assert r.dtype == expected_dtype

        if strip_zeros:
            assert_roots_match(r, expected)
        else:
            assert_roots_match(r, jnp.append(expected, jnp.full(nzeros, jnp.nan)))


@pytest.mark.parametrize("real", [False, True])
def test_triple_root(real: bool) -> None:
    r = roots(jnp.array([1, -3, 3, -1]), real=real)

    assert isinstance(r, jax.Array)
    assert r.dtype == (float if real else complex)

    assert_roots_match(r, jnp.array([1, 1, 1]))


def test_complex_double_root() -> None:
    r = roots(jnp.array([1, -3 + 1j, 4 - 2j, -2 + 2j]))

    assert isinstance(r, jax.Array)
    assert r.dtype == complex

    assert_roots_match(r, jnp.array([1 + 1j, 1 - 1j, 1 - 1j]))


@pytest.mark.parametrize("real", [False, True])
def test_random_vs_jnp(real: bool) -> None:
    vmapped_jnp_roots = jax.jit(jax.vmap(lambda p: jnp.roots(p, strip_zeros=False)))(
        RANDOM_CUBICS
    )

    assert isinstance(vmapped_jnp_roots, jax.Array)
    assert vmapped_jnp_roots.dtype == complex

    vmapped_roots = jax.jit(jax.vmap(lambda p: roots(p, strip_zeros=False, real=real)))(
        RANDOM_CUBICS
    )
    assert isinstance(vmapped_roots, jax.Array)
    assert vmapped_roots.dtype == (float if real else complex)

    assert_roots_match(vmapped_roots, vmapped_jnp_roots, abs=1e-6, rel=1e-3)
