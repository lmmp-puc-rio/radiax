import itertools
import time

import jax
import jax.numpy as jnp
import pytest

from radiax import roots

jax.config.update("jax_enable_x64", True)


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


def assert_roots_match(actual: jax.Array, expected: jax.Array, /) -> None:
    assert expected.ndim == 1
    assert actual.shape == expected.shape

    if jnp.issubdtype(expected.dtype, jnp.complexfloating) and not jnp.issubdtype(
        actual.dtype, jnp.complexfloating
    ):
        expected = jnp.where(
            jnp.isreal(expected),
            jnp.real(expected),
            jnp.nan,
        )

    for perm in itertools.permutations(range(expected.size)):
        perm = jnp.array(list(perm))
        if actual == pytest.approx(expected[perm], nan_ok=True):
            return

    pytest.fail(f"roots do not match: {actual!r} != {expected!r}")


@pytest.mark.parametrize(("p", "expected"), KNOWN_ROOTS)
def test_testset(p: jax.Array, expected: jax.Array) -> None:
    assert jnp.polyval(p, expected) == pytest.approx(0)


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

        assert r.dtype == expected_dtype

        if strip_zeros:
            assert_roots_match(r, expected)
        else:
            assert_roots_match(r, jnp.append(expected, jnp.full(nzeros, jnp.nan)))


def test_invalid_inputs() -> None:
    with pytest.raises(TypeError):
        roots([1, 2, 3])  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="1D"):
        roots(jnp.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError):
        roots(jnp.array([1 + 1j, 2 + 2j]), real=True)


def test_unimplemented_degree() -> None:
    with pytest.raises(NotImplementedError, match="degree"):
        roots(jnp.arange(5), strip_zeros=False)

    with pytest.raises(NotImplementedError, match="degree"):
        roots(jnp.arange(5), strip_zeros=False, real=True)

    roots(jnp.arange(5), strip_zeros=True)  # first zero is stripped, so degree is 3


def test_non_jittable() -> None:
    with pytest.raises(
        jax.errors.ConcretizationTypeError, match="radiax.roots.*strip_zeros=False"
    ):
        jax.jit(lambda p: roots(p, strip_zeros=True))(jnp.array([1, 2, 3]))

    with pytest.raises(
        jax.errors.ConcretizationTypeError, match="radiax.roots.*strip_zeros=False"
    ):
        jax.jit(roots)(jnp.array([1, 2, 3]))


@pytest.mark.parametrize("real", [False, True])
def test_triple_root(real: bool) -> None:
    r = roots(jnp.array([1, -3, 3, -1]), real=real)
    assert r.dtype == (float if real else complex)
    assert_roots_match(r, jnp.array([1, 1, 1]))


def test_complex_double_root() -> None:
    r = roots(jnp.array([1, -3 + 1j, 4 - 2j, -2 + 2j]))
    assert r.dtype == complex
    assert_roots_match(r, jnp.array([1 + 1j, 1 - 1j, 1 - 1j]))


def test_speedup():
    p = jax.random.uniform(jax.random.key(0), (100_000, 4), minval=1, maxval=10)

    vmapped_jnp_roots = jax.jit(jax.vmap(lambda p: jnp.roots(p, strip_zeros=False)))
    vmapped_radiax_roots = jax.jit(jax.vmap(lambda p: roots(p, strip_zeros=False)))

    vmapped_jnp_roots(p).block_until_ready()
    vmapped_radiax_roots(p).block_until_ready()

    start = time.perf_counter()
    vmapped_jnp_roots(p).block_until_ready()
    time_jnp = time.perf_counter() - start

    start = time.perf_counter()
    vmapped_radiax_roots(p).block_until_ready()
    time_radiax = time.perf_counter() - start

    assert time_radiax < 0.5 * time_jnp
