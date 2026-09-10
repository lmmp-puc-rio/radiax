import jax
import jax.numpy as jnp
import pytest

from radiax import roots
from tests.utils import assert_roots_match


def _cubic_from_roots(r: jax.Array) -> jax.Array:
    r0, r1, r2 = r
    return jnp.array(
        [
            1,
            -(r0 + r1 + r2),
            r0 * r1 + r0 * r2 + r1 * r2,
            -r0 * r1 * r2,
        ],
        dtype=r.dtype,
    )


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("real", [False, True])
def test_cubic_monic_normalization_does_not_overflow(
    dtype: jnp.dtype, real: bool
) -> None:
    finfo = jnp.finfo(dtype)
    leading = jnp.asarray(1.25, dtype=dtype) * finfo.tiny
    middle = jnp.asarray(8.0, dtype=dtype)
    p = jnp.array([leading, 0, -middle, 0], dtype=dtype)

    log_expected = 0.5 * (jnp.log(middle) - jnp.log(leading))
    expected_magnitude = jnp.exp(log_expected)
    expected = jnp.array([-expected_magnitude, 0, expected_magnitude], dtype=dtype)

    r = roots(p, strip_zeros=False, real=real)
    actual = jnp.sort(r if real else r.real)

    assert jnp.all(jnp.isfinite(r))
    assert actual == pytest.approx(
        expected, rel=5e-6 if dtype == jnp.float32 else 1e-12
    )
    if not real:
        assert r.imag == pytest.approx(0)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("real", [False, True])
def test_cubic_scale_invariance_over_400_orders(scale: float, real: bool) -> None:
    p = jnp.asarray(scale, dtype=jnp.float64) * jnp.array(
        [1.0, -6.0, 11.0, -6.0], dtype=jnp.float64
    )
    r = roots(p, strip_zeros=False, real=real)

    actual = jnp.sort(r if real else r.real)
    assert actual == pytest.approx([1.0, 2.0, 3.0])
    if not real:
        assert r.imag == pytest.approx(0)


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (
            jnp.array([1e-200, 0.0, -1e100, 0.0], dtype=jnp.float64),
            jnp.array([-1e150, 0.0, 1e150], dtype=jnp.float64),
        ),
        (
            jnp.array([1e100, 0.0, -1e-100, 0.0], dtype=jnp.float64),
            jnp.array([-1e-100, 0.0, 1e-100], dtype=jnp.float64),
        ),
        (
            jnp.array([1e-120, -1e80, 0.0, 0.0], dtype=jnp.float64),
            jnp.array([0.0, 0.0, 1e200], dtype=jnp.float64),
        ),
    ],
)
def test_cubic_extreme_coefficient_dynamic_range(
    p: jax.Array, expected: jax.Array
) -> None:
    r = roots(p, strip_zeros=False)

    assert jnp.all(jnp.isfinite(r))
    assert_roots_match(r, expected.astype(r.dtype), abs=1e-12, rel=2e-10)


@pytest.mark.parametrize(
    "expected",
    [
        jnp.array([1.0, 1.0 + 1e-6, 3.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0 + 1e-8, 1.0 - 1e-8], dtype=jnp.float64),
        jnp.array([-1e8, 1e-8, 2.0], dtype=jnp.float64),
        jnp.array([-1e-8, 1e-8, 1e8], dtype=jnp.float64),
    ],
)
def test_adversarial_real_root_configurations(expected: jax.Array) -> None:
    p = _cubic_from_roots(expected)
    r = roots(p, strip_zeros=False)

    assert_roots_match(r, expected.astype(r.dtype), abs=2e-7, rel=2e-7)
    assert _normalized_residual(p, r) == pytest.approx(0, abs=2e-14)


@pytest.mark.parametrize(
    "expected",
    [
        jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
        jnp.array([1e-20, 1.0, 2.0], dtype=jnp.float64),
        jnp.array([1e-100, -3.0, 4.0], dtype=jnp.float64),
    ],
)
def test_zero_and_almost_zero_constant_terms(expected: jax.Array) -> None:
    p = _cubic_from_roots(expected)
    r = roots(p, strip_zeros=False)

    assert_roots_match(r, expected.astype(r.dtype), abs=1e-14, rel=1e-10)
    assert _normalized_residual(p, r) == pytest.approx(0, abs=1e-14)


def test_genuinely_complex_near_double_root() -> None:
    expected = jnp.array(
        [1.0 + 1.0j, 1.0 + 1.0j + 1e-7, -2.0 + 0.5j],
        dtype=jnp.complex128,
    )
    p = _cubic_from_roots(expected)
    r = roots(p, strip_zeros=False)

    assert_roots_match(r, expected, abs=2e-7, rel=2e-7)
    assert _normalized_residual(p, r) == pytest.approx(0, abs=2e-14)


def test_log_uniform_real_root_fuzz() -> None:
    key_magnitude, key_sign = jax.random.split(jax.random.key(4312))
    exponents = jax.random.uniform(
        key_magnitude,
        (2048, 3),
        minval=-8.0,
        maxval=8.0,
        dtype=jnp.float64,
    )
    signs = jnp.where(
        jax.random.bernoulli(key_sign, shape=(2048, 3)),
        1.0,
        -1.0,
    )
    expected = signs * 10**exponents
    polynomials = jax.vmap(_cubic_from_roots)(expected)
    actual = jax.jit(jax.vmap(lambda p: roots(p, strip_zeros=False)))(polynomials)

    residuals = jax.vmap(_normalized_residual)(polynomials, actual)
    assert jnp.all(jnp.isfinite(actual))
    assert residuals == pytest.approx(0, abs=2e-13)

    assert_roots_match(actual, expected, abs=1e-8, rel=2e-8)


def test_log_uniform_complex_root_fuzz() -> None:
    key = jax.random.key(9271)
    keys = jax.random.split(key, 4)
    log_magnitudes = jax.random.uniform(
        keys[0],
        (1024, 3),
        minval=-6.0,
        maxval=6.0,
        dtype=jnp.float64,
    )
    phases = jax.random.uniform(
        keys[1],
        (1024, 3),
        minval=-jnp.pi,
        maxval=jnp.pi,
        dtype=jnp.float64,
    )
    magnitudes = 10**log_magnitudes
    expected = magnitudes * jnp.exp(1j * phases)
    polynomials = jax.vmap(_cubic_from_roots)(expected.astype(jnp.complex128))
    actual = jax.jit(jax.vmap(lambda p: roots(p, strip_zeros=False)))(polynomials)

    residuals = jax.vmap(_normalized_residual)(polynomials, actual)
    assert jnp.all(jnp.isfinite(actual))
    assert residuals == pytest.approx(0, abs=5e-13)

    assert_roots_match(actual, expected, abs=2e-8, rel=5e-8)
