import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from radiax import roots


def _implicit_jacobian(p: jax.Array, r: jax.Array) -> jax.Array:
    degree = p.size - 1
    powers = jnp.arange(degree, -1, -1)
    derivative_coefficients = p[:-1] * jnp.arange(degree, 0, -1, dtype=p.dtype)
    derivative = jnp.polyval(derivative_coefficients, r)
    return -(r[:, None] ** powers[None, :]) / derivative[:, None]


def _match_to_reference(reference: jax.Array, candidate: jax.Array) -> jax.Array:
    reference_np = np.asarray(reference)
    candidate_np = np.asarray(candidate)
    permutations = itertools.permutations(range(reference_np.size))
    permutation = min(
        permutations,
        key=lambda indices: np.linalg.norm(candidate_np[list(indices)] - reference_np),
    )
    return jnp.asarray(candidate_np[list(permutation)])


REAL_POLYNOMIALS = [
    jnp.array([2.0, -8.0], dtype=jnp.float64),
    jnp.array([1.0, -1.0, -6.0], dtype=jnp.float64),
    jnp.array([1.0, -1.5, -5.5, 3.0], dtype=jnp.float64),
]


@pytest.mark.parametrize("p", REAL_POLYNOMIALS)
def test_jvp_matches_implicit_derivative(p: jax.Array) -> None:
    dp = 0.1 * jnp.arange(1, p.size + 1, dtype=p.dtype)
    r, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False, real=True),
        (p,),
        (dp,),
    )
    expected = _implicit_jacobian(p, r) @ dp

    assert dr == pytest.approx(expected)


@pytest.mark.parametrize("p", REAL_POLYNOMIALS)
def test_grad_matches_implicit_derivative(p: jax.Array) -> None:
    def loss(q: jax.Array) -> jax.Array:
        r = roots(q, strip_zeros=False, real=True)
        return jnp.sum(jnp.exp(r))

    r = roots(p, strip_zeros=False, real=True)
    jacobian = _implicit_jacobian(p, r)
    expected = jnp.sum(jnp.exp(r)[:, None] * jacobian, axis=0)

    actual = jax.grad(loss)(p)

    assert actual == pytest.approx(expected)


def test_cubic_complex_output_jvp_matches_implicit_derivative() -> None:
    p = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    dp = jnp.array([0.2, -0.3, 0.4, -0.5], dtype=jnp.float64)
    r, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False),
        (p,),
        (dp,),
    )
    expected = _implicit_jacobian(p, r) @ dp

    assert dr == pytest.approx(expected)


def test_genuinely_complex_cubic_jvp_matches_implicit_derivative() -> None:
    expected_roots = jnp.array(
        [1.0 + 1.0j, -2.0 + 0.5j, 0.25 - 1.5j],
        dtype=jnp.complex128,
    )
    p = jnp.poly(expected_roots)
    dp = jnp.array(
        [0.2 - 0.1j, -0.3 + 0.05j, 0.4 + 0.2j, -0.5 + 0.3j],
        dtype=jnp.complex128,
    )

    r, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False),
        (p,),
        (dp,),
    )
    expected = _implicit_jacobian(p, r) @ dp
    assert dr == pytest.approx(expected)


def test_grad_through_complex_roots_matches_implicit_derivative() -> None:
    p = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)

    def loss(q: jax.Array) -> jax.Array:
        r = roots(q, strip_zeros=False)
        return jnp.sum(jnp.abs(r) ** 2)

    r = roots(p, strip_zeros=False)
    jacobian = _implicit_jacobian(p, r)
    expected = 2 * jnp.real(jnp.sum(jnp.conj(r)[:, None] * jacobian, axis=0))

    actual = jax.grad(loss)(p)
    assert actual == pytest.approx(expected)


def test_jacfwd_and_jacrev_match_implicit_jacobian() -> None:
    p = jnp.array([1.0, -1.5, -5.5, 3.0], dtype=jnp.float64)

    def solve(q: jax.Array) -> jax.Array:
        return roots(q, strip_zeros=False, real=True)

    r = solve(p)
    expected = _implicit_jacobian(p, r)

    forward = jax.jacfwd(solve)(p)
    reverse = jax.jacrev(solve)(p)

    assert forward == pytest.approx(expected)
    assert reverse == pytest.approx(expected)


def test_jvp_is_zero_in_coefficient_scaling_direction() -> None:
    p = jnp.array([1.0, -1.5, -5.5, 3.0], dtype=jnp.float64)

    _, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False, real=True),
        (p,),
        (p,),
    )

    assert dr == pytest.approx(0)


def test_real_mode_grad_ignores_nan_placeholders() -> None:
    def real_root_sum(d: jax.Array) -> jax.Array:
        r = roots(
            jnp.array([1.0, 0.0, 0.0, d]),
            strip_zeros=False,
            real=True,
        )
        return jnp.nansum(r)

    d = jnp.array(1.0, dtype=jnp.float64)
    value, derivative = jax.value_and_grad(real_root_sum)(d)

    assert value == pytest.approx(-1)
    assert derivative == pytest.approx(-1 / 3)


def test_second_derivative_through_simple_cubic_root() -> None:
    def real_root(d: jax.Array) -> jax.Array:
        r = roots(
            jnp.array([1.0, 0.0, 0.0, d]),
            strip_zeros=False,
        )
        return jnp.min(r.real)

    d = jnp.array(1.0, dtype=jnp.float64)

    first = jax.grad(real_root)(d)
    second = jax.grad(jax.grad(real_root))(d)

    assert first == pytest.approx(-1 / 3)
    assert second == pytest.approx(2 / 9)


def test_differentiation_works_under_jit_and_vmap() -> None:
    polynomials = jnp.array(
        [
            [1.0, -1.5, -5.5, 3.0],
            [1.0, -2.0, -11.0, 12.0],
        ],
        dtype=jnp.float64,
    )

    def loss(p: jax.Array) -> jax.Array:
        return jnp.sum(jnp.exp(roots(p, strip_zeros=False, real=True)))

    batched_grad = jax.jit(jax.vmap(jax.grad(loss)))
    actual = batched_grad(polynomials)
    expected = []
    for p in polynomials:
        r = roots(p, strip_zeros=False, real=True)
        jacobian = _implicit_jacobian(p, r)
        expected.append(jnp.sum(jnp.exp(r)[:, None] * jacobian, axis=0))
    expected = jnp.stack(expected)

    assert actual == pytest.approx(expected)


def test_jitted_grad_requires_strip_zeros_false() -> None:
    def loss(p: jax.Array) -> jax.Array:
        return jnp.sum(roots(p).real)

    differentiated = jax.jit(jax.grad(loss))
    p = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)

    with pytest.raises(jax.errors.ConcretizationTypeError, match="strip_zeros=False"):
        differentiated(p)


def test_quadratic_jvp_is_zero_in_coefficient_scaling_direction() -> None:
    p = jnp.array([1.0, 1e8, 1.0], dtype=jnp.float32)

    _, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False, real=True),
        (p,),
        (p,),
    )

    assert dr == pytest.approx(0)


def test_quadratic_real_mode_grad_ignores_nan_placeholders() -> None:
    def real_root_sum(c: jax.Array) -> jax.Array:
        r = roots(
            jnp.array([1.0, 0.0, c]),
            strip_zeros=False,
            real=True,
        )
        return jnp.nansum(r)

    c = jnp.array(1.0, dtype=jnp.float64)
    value, derivative = jax.value_and_grad(real_root_sum)(c)

    assert value == pytest.approx(0)
    assert derivative == pytest.approx(0)


@pytest.mark.parametrize(
    ("p", "dp", "real", "step"),
    [
        (
            jnp.array([1.0, -3.0, 2.0], dtype=jnp.float64),
            jnp.array([0.2, -0.3, 0.4], dtype=jnp.float64),
            True,
            1e-6,
        ),
        (
            jnp.array([1.0, -1.5, -5.5, 3.0], dtype=jnp.float64),
            jnp.array([0.2, -0.3, 0.4, -0.5], dtype=jnp.float64),
            True,
            1e-6,
        ),
        (
            jnp.poly(
                jnp.array(
                    [1.0 + 1.0j, -2.0 + 0.5j, 0.25 - 1.5j],
                    dtype=jnp.complex128,
                )
            ),
            jnp.array(
                [0.2 - 0.1j, -0.3 + 0.05j, 0.4 + 0.2j, -0.5 + 0.3j],
                dtype=jnp.complex128,
            ),
            False,
            1e-6,
        ),
    ],
)
def test_jvp_matches_central_finite_difference(
    p: jax.Array, dp: jax.Array, real: bool, step: float
) -> None:
    solve = lambda q: roots(q, strip_zeros=False, real=real)
    r, dr = jax.jvp(solve, (p,), (dp,))
    r_plus = _match_to_reference(r, solve(p + step * dp))
    r_minus = _match_to_reference(r, solve(p - step * dp))
    finite_difference = (r_plus - r_minus) / (2 * step)

    assert dr == pytest.approx(finite_difference, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize(
    "p",
    [
        jnp.poly(jnp.array([1.0, 1.0 + 1e-5], dtype=jnp.float64)),
        jnp.poly(jnp.array([1.0, 1.0 + 1e-5, 3.0], dtype=jnp.float64)),
        jnp.poly(jnp.array([1.0, 1.0 + 1e-4, 1.0 - 1e-4], dtype=jnp.float64)),
    ],
)
def test_near_repeated_root_jvp_remains_finite_and_implicit(p: jax.Array) -> None:
    dp = 0.1 * jnp.arange(1, p.size + 1, dtype=p.dtype)
    r, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False, real=True),
        (p,),
        (dp,),
    )
    expected = _implicit_jacobian(p, r) @ dp

    assert jnp.all(jnp.isfinite(dr))
    assert dr == pytest.approx(expected, rel=2e-5, abs=2e-7)


@pytest.mark.parametrize(
    "p",
    [
        jnp.array([1.0, -2.0, 1.0], dtype=jnp.float64),
        jnp.array([1.0, -5.0, 7.0, -3.0], dtype=jnp.float64),
        jnp.array([1.0, -3.0, 3.0, -1.0], dtype=jnp.float64),
    ],
)
def test_exact_repeated_roots_have_singular_root_derivatives(p: jax.Array) -> None:
    dp = jnp.arange(1, p.size + 1, dtype=p.dtype)
    _, dr = jax.jvp(
        lambda q: roots(q, strip_zeros=False, real=True),
        (p,),
        (dp,),
    )

    assert jnp.any(~jnp.isfinite(dr))
