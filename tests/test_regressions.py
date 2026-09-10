import jax
import jax.numpy as jnp
import pytest

from radiax import roots


def _normalized_residual(p: jax.Array, r: jax.Array) -> jax.Array:
    p = jnp.asarray(p)
    r = jnp.asarray(r)
    degree = p.size - 1
    powers = jnp.arange(degree, -1, -1)
    numerator = jnp.abs(jnp.polyval(p, r))
    denominator = jnp.sum(
        jnp.abs(p) * jnp.maximum(1, jnp.abs(r))[..., None] ** powers,
        axis=-1,
    )
    return numerator / denominator


@pytest.mark.parametrize("real", [False, True])
def test_quadratic_cancellation(real: bool) -> None:
    p = jnp.array([1.0, 1e8, 1.0], dtype=jnp.float32)
    r = roots(p, real=real)

    if real:
        actual = jnp.sort(r)
    else:
        assert jnp.abs(r.imag) == pytest.approx(0)
        actual = jnp.sort(r.real)

    assert actual == pytest.approx([-1e8, -1e-8])


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("real", [False, True])
def test_quadratic_scale_invariance(scale: float, real: bool) -> None:
    p = scale * jnp.array([1.0, -3.0, 2.0], dtype=jnp.float32)
    r = roots(p, real=real)

    actual = jnp.sort(r if real else r.real)
    assert actual == pytest.approx([1.0, 2.0])


def test_real_integer_coefficients_do_not_overflow() -> None:
    p = jnp.array([100_000, -300_000, 200_000], dtype=jnp.int32)
    r = roots(p, real=True)

    assert r.dtype == jnp.float32
    assert jnp.sort(r) == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("real", [False, True])
def test_cubic_small_real_root(real: bool) -> None:
    p = jnp.array([1.0, 0.0, 1e8, -1.0], dtype=jnp.float32)
    r = roots(p, real=real)
    finite = r[jnp.isfinite(r)]
    if not real:
        real_roots = finite[jnp.abs(finite.imag) < 1e-3]
        assert real_roots.size == 1
        root = real_roots[0].real
    else:
        assert finite.size == 1
        root = finite[0]

    assert root == pytest.approx(1e-8)
    assert _normalized_residual(p, root) == pytest.approx(0)


@pytest.mark.parametrize("real", [False, True])
def test_cubic_widely_separated_real_roots(real: bool) -> None:
    expected = jnp.array([1e6, 1e-3, -1e-3], dtype=jnp.float32)
    p = jnp.poly(expected)
    r = roots(p, real=real)

    if real:
        actual = jnp.sort(r)
    else:
        assert jnp.abs(r.imag) == pytest.approx(0)
        actual = jnp.sort(r.real)

    assert actual == pytest.approx(jnp.sort(expected))
    assert _normalized_residual(p, r) == pytest.approx(0)


def test_complex_float32_double_root() -> None:
    p = jnp.array([1, -3 + 1j, 4 - 2j, -2 + 2j], dtype=jnp.complex64)
    expected = jnp.array([1 + 1j, 1 - 1j, 1 - 1j], dtype=jnp.complex64)

    r = roots(p)

    assert jnp.count_nonzero(jnp.abs(r - expected[0]) < 2e-6) == 1
    assert jnp.count_nonzero(jnp.abs(r - expected[1]) < 2e-6) == 2


def test_constant_polynomial_with_padding_returns_only_nans() -> None:
    p = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    for real in (False, True):
        r = roots(p, strip_zeros=False, real=real)
        assert jnp.all(jnp.isnan(r))


def test_regressions_work_under_vmap_and_jit() -> None:
    p = jnp.array(
        [
            [1.0, 0.0, 1e8, -1.0],
            [1.0, -1e6, -1e-6, 1.0],
        ],
        dtype=jnp.float32,
    )

    solve = jax.jit(
        jax.vmap(lambda coefficients: roots(coefficients, strip_zeros=False))
    )
    r = solve(p)
    assert jnp.all(jnp.isfinite(r))
    residuals = jax.vmap(_normalized_residual)(p, r)
    assert residuals == pytest.approx(0)


def test_cubic_tiny_real_root_with_complex_pair() -> None:
    expected = jnp.array([1e-6, 2 + 3j, 2 - 3j], dtype=jnp.complex64)
    p = jnp.poly(expected).real.astype(jnp.float32)

    r = roots(p)

    assert _normalized_residual(p, r) == pytest.approx(0, abs=1e-7)


def test_genuinely_complex_cubic_multiscale_roots() -> None:
    expected = jnp.array(
        [1000 + 2000j, -1 + 0.5j, 1e-3 - 1e-3j],
        dtype=jnp.complex64,
    )
    p = jnp.poly(expected)

    r = roots(p)

    assert _normalized_residual(p, r) == pytest.approx(0, abs=1e-7)
