import jax
import jax.core
import jax.numpy as jnp
import numpy as np


def _as_complex(p: jax.Array, /) -> jax.Array:
    dtype = jnp.complex128 if p.dtype.itemsize > 4 else jnp.complex64
    return p.astype(dtype)


def _nan(dtype: jnp.dtype, /) -> jax.Array:
    value = jnp.asarray(jnp.nan, dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        value = value + 1j * value
    return value


def _roots_linear_no_zeros(p: jax.Array, /) -> jax.Array:
    return jnp.array([-p[1] / p[0]])


def _roots_linear_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_linear_no_zeros,
        lambda p: jnp.array([_nan(p.dtype)]),
        p,
    )


def _roots_quadratic_stable(
    p: jax.Array, /, *, snap_near_double: bool = False
) -> jax.Array:
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(p)))
    p = p / scale
    a, b, c = p
    discriminant = b * b - 4 * a * c
    if snap_near_double:
        discriminant_scale = jnp.abs(b * b) + jnp.abs(4 * a * c)
        eps = jnp.finfo(p.real.dtype).eps
        near_double = jnp.abs(discriminant) <= 8 * eps * discriminant_scale
        discriminant = jnp.where(
            near_double, jnp.zeros_like(discriminant), discriminant
        )
    sqrt_discriminant = jnp.sqrt(discriminant)
    q1 = -0.5 * (b + sqrt_discriminant)
    q2 = -0.5 * (b - sqrt_discriminant)
    q = jnp.where(jnp.abs(q1) >= jnp.abs(q2), q1, q2)

    def general() -> jax.Array:
        return jnp.array([q / a, c / q])

    def zero_q() -> jax.Array:
        root = -b / (2 * a)
        return jnp.array([root, root])

    return jax.lax.cond(q != 0, general, zero_q)


def _roots_quadratic_no_zeros_impl(p: jax.Array, /) -> jax.Array:
    return _roots_quadratic_stable(p)


def _projective_tangent(p: jax.Array, dp: jax.Array, /) -> jax.Array:
    return dp - (dp[0] / p[0]) * p


def _quadratic_jvp(p: jax.Array, dp: jax.Array, r: jax.Array, /) -> jax.Array:
    dp = _projective_tangent(p, dp)
    polynomial_tangent = (dp[0] * r + dp[1]) * r + dp[2]
    derivative = 2 * p[0] * r + p[1]
    return -polynomial_tangent / derivative


@jax.custom_jvp
def _roots_quadratic_no_zeros(p: jax.Array, /) -> jax.Array:
    return _roots_quadratic_no_zeros_impl(p)


@_roots_quadratic_no_zeros.defjvp
def _roots_quadratic_no_zeros_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (p,), (dp,) = primals, tangents
    r = _roots_quadratic_no_zeros_impl(p)
    finite = jnp.isfinite(r)
    safe_r = jnp.where(finite, r, jnp.ones_like(r))
    dr = _quadratic_jvp(p, dp, safe_r)
    return r, jnp.where(finite, dr, jnp.zeros_like(dr))


def _roots_quadratic_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_quadratic_no_zeros,
        lambda p: jnp.append(_roots_linear_with_zeros(p[1:]), _nan(p.dtype)),
        p,
    )


def _frexp_abs(value: jax.Array, /) -> tuple[jax.Array, jax.Array]:
    return jnp.frexp(jnp.abs(value))


def _unit_phase(value: jax.Array, /) -> jax.Array:
    magnitude = jnp.abs(value)
    safe_magnitude = jnp.where(magnitude != 0, magnitude, jnp.ones_like(magnitude))
    return value / safe_magnitude


def _scaled_monic_coefficient(
    coefficient: jax.Array,
    leading: jax.Array,
    scale_exponent: jax.Array,
    power: int,
    /,
) -> jax.Array:
    coefficient_mantissa, coefficient_exponent = _frexp_abs(coefficient)
    leading_mantissa, leading_exponent = _frexp_abs(leading)
    exponent = coefficient_exponent - leading_exponent - power * scale_exponent
    magnitude = jnp.ldexp(coefficient_mantissa / leading_mantissa, exponent)
    phase = _unit_phase(coefficient) * jnp.conj(_unit_phase(leading))
    return magnitude * phase


def _scaled_monic_cubic(p: jax.Array, /) -> tuple[jax.Array, ...]:
    leading_mantissa, leading_exponent = _frexp_abs(p[0])
    candidates = []
    for power in range(1, 4):
        mantissa, exponent = _frexp_abs(p[power])
        mantissa_ratio = jnp.where(
            mantissa != 0, mantissa / leading_mantissa, jnp.ones_like(mantissa)
        )
        log2_ratio = jnp.log2(mantissa_ratio) + (exponent - leading_exponent)
        candidate = jnp.where(mantissa != 0, log2_ratio / power, -jnp.inf)
        candidates.append(candidate)

    log2_scale = jnp.maximum(candidates[0], jnp.maximum(candidates[1], candidates[2]))
    scale_exponent = jnp.where(
        jnp.isneginf(log2_scale),
        jnp.asarray(0, dtype=leading_exponent.dtype),
        jnp.floor(log2_scale).astype(leading_exponent.dtype),
    )
    scale_exponent = jax.lax.stop_gradient(scale_exponent)
    scale = jnp.ldexp(jnp.ones_like(leading_mantissa), scale_exponent)

    a = _scaled_monic_coefficient(p[1], p[0], scale_exponent, 1)
    b = _scaled_monic_coefficient(p[2], p[0], scale_exponent, 2)
    c = _scaled_monic_coefficient(p[3], p[0], scale_exponent, 3)
    return a, b, c, scale


def _deflation_error(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    root: jax.Array,
    q1: jax.Array,
    q0: jax.Array,
    /,
) -> jax.Array:
    reconstructed_a = q1 - root
    reconstructed_b = q0 - root * q1
    reconstructed_c = -root * q0
    return (
        jnp.abs(reconstructed_a - a) / (1 + jnp.abs(a))
        + jnp.abs(reconstructed_b - b) / (1 + jnp.abs(b))
        + jnp.abs(reconstructed_c - c) / (1 + jnp.abs(c))
    )


def _deflated_quadratic(
    a: jax.Array, b: jax.Array, c: jax.Array, root: jax.Array, /
) -> jax.Array:
    q1_synthetic = a + root
    q0_synthetic = b + root * q1_synthetic

    def nonzero_root() -> jax.Array:
        q0_vieta = -c / root
        q1_vieta = (q0_vieta - b) / root
        synthetic_error = _deflation_error(a, b, c, root, q1_synthetic, q0_synthetic)
        vieta_error = _deflation_error(a, b, c, root, q1_vieta, q0_vieta)
        use_vieta = vieta_error < synthetic_error
        q1 = jnp.where(use_vieta, q1_vieta, q1_synthetic)
        q0 = jnp.where(use_vieta, q0_vieta, q0_synthetic)
        return jnp.array([jnp.ones_like(root), q1, q0])

    def zero_root() -> jax.Array:
        return jnp.array([jnp.ones_like(root), q1_synthetic, q0_synthetic])

    return jax.lax.cond(root != 0, nonzero_root, zero_root)


def _one_real_depressed_cubic_root(P: jax.Array, Q: jax.Array, /) -> jax.Array:
    def positive_P() -> jax.Array:
        sqrt_P = jnp.sqrt(P)
        argument = Q / (P * sqrt_P)
        return -2 * sqrt_P * jnp.sinh(jnp.arcsinh(argument) / 3)

    def nonpositive_P() -> jax.Array:
        def negative_P() -> jax.Array:
            sqrt_minus_P = jnp.sqrt(-P)
            denominator = (-P) * sqrt_minus_P
            ratio = jnp.abs(Q) / denominator

            def three_real() -> jax.Array:
                argument = jnp.clip(-Q / denominator, -1, 1)
                return 2 * sqrt_minus_P * jnp.cos(jnp.arccos(argument) / 3)

            def one_real() -> jax.Array:
                argument = jnp.maximum(ratio, jnp.asarray(1, dtype=ratio.dtype))
                return (
                    -2
                    * jnp.sign(Q)
                    * sqrt_minus_P
                    * jnp.cosh(jnp.arccosh(argument) / 3)
                )

            return jax.lax.cond(ratio <= 1, three_real, one_real)

        def zero_P() -> jax.Array:
            return jnp.cbrt(-2 * Q)

        return jax.lax.cond(P < 0, negative_P, zero_P)

    return jax.lax.cond(P > 0, positive_P, nonpositive_P)


def _real_cubic_root_and_quadratic(
    p: jax.Array, /
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    a, b, c, scale = _scaled_monic_cubic(p)
    shift = a / 3
    depressed_p = b - a * shift
    depressed_q = c + 2 * shift**3 - b * shift
    P = depressed_p / 3
    Q = depressed_q / 2
    tolerance = 8 * jnp.finfo(p.dtype).eps
    near_triple = (jnp.abs(P) <= tolerance) & (jnp.abs(Q) <= tolerance)
    P = jnp.where(near_triple, jnp.zeros_like(P), P)
    Q = jnp.where(near_triple, jnp.zeros_like(Q), Q)

    def three_real_root() -> jax.Array:
        sqrt_minus_P = jnp.sqrt(-P)
        denominator = (-P) * sqrt_minus_P
        argument = jnp.clip(-Q / denominator, -1, 1)
        theta = jnp.arccos(argument) / 3
        offsets = jnp.asarray([0, -2 * jnp.pi / 3, -4 * jnp.pi / 3], dtype=p.dtype)
        candidates = 2 * sqrt_minus_P * jnp.cos(theta + offsets) - shift
        derivative = (3 * candidates + 2 * a) * candidates + b
        return candidates[jnp.argmax(jnp.abs(derivative))]

    def one_real_root() -> jax.Array:
        return _one_real_depressed_cubic_root(P, Q) - shift

    three_real = (P < 0) & (jnp.abs(Q) <= (-P) * jnp.sqrt(-P))
    root = jax.lax.cond(three_real, three_real_root, one_real_root)
    quadratic = _deflated_quadratic(a, b, c, root)
    return root, quadratic, c, scale


def _roots_cubic_real_no_zeros_impl(p: jax.Array, /) -> jax.Array:
    root, quadratic, c, scale = _real_cubic_root_and_quadratic(p)
    remaining = _roots_quadratic_stable(quadratic, snap_near_double=True)
    roots = jnp.concatenate((jnp.array([root]), remaining))
    roots = jax.lax.cond(
        jnp.all(jnp.isfinite(roots)),
        lambda roots: _polish_smallest_root_from_product(c, roots),
        lambda roots: roots,
        roots,
    )
    return roots * scale


def _roots_cubic_real_as_complex_no_zeros_impl(p: jax.Array, /) -> jax.Array:
    root, quadratic, c, scale = _real_cubic_root_and_quadratic(p)
    remaining = _roots_quadratic_stable(_as_complex(quadratic), snap_near_double=True)
    roots = jnp.concatenate((jnp.array([root]).astype(remaining.dtype), remaining))
    roots = _polish_smallest_root_from_product(c, roots)
    return roots * scale


def _polish_smallest_root_from_product(c: jax.Array, roots: jax.Array, /) -> jax.Array:
    smallest = jnp.argmin(jnp.abs(roots))
    denominators = jnp.array(
        [roots[1] * roots[2], roots[0] * roots[2], roots[0] * roots[1]]
    )
    denominator = denominators[smallest]
    corrected = jax.lax.cond(
        denominator != 0,
        lambda: -c / denominator,
        lambda: roots[smallest],
    )
    return roots.at[smallest].set(corrected)


def _roots_cubic_genuinely_complex_no_zeros(p: jax.Array, /) -> jax.Array:
    a, b, c, scale = _scaled_monic_cubic(p)

    shift = a / 3
    P = (b - a * shift) / 3
    Q = (c + 2 * shift**3 - b * shift) / 2
    sqrt_delta = jnp.sqrt(Q * Q + P * P * P)

    z1 = -Q + sqrt_delta
    z2 = -Q - sqrt_delta
    z = jnp.where(jnp.abs(z1) >= jnp.abs(z2), z1, z2)

    def nonzero_z() -> jax.Array:
        u = z ** (1 / 3)
        v = -P / u
        omega = jnp.asarray(
            -0.5 + 0.5j * jnp.sqrt(3),
            dtype=p.dtype,
        )
        omega_conj = jnp.conj(omega)
        candidates = jnp.array(
            [
                u + v - shift,
                omega * u + omega_conj * v - shift,
                omega_conj * u + omega * v - shift,
            ]
        )
        derivative = (3 * candidates + 2 * a) * candidates + b
        return candidates[jnp.argmax(jnp.abs(derivative))]

    def zero_z() -> jax.Array:
        return -shift

    root = jax.lax.cond(z != 0, nonzero_z, zero_z)
    derivative = (3 * root + 2 * a) * root + b
    value = ((root + a) * root + b) * root + c
    root = jax.lax.cond(
        derivative != 0, lambda: root - value / derivative, lambda: root
    )
    quadratic = _deflated_quadratic(a, b, c, root)
    remaining = _roots_quadratic_stable(quadratic, snap_near_double=True)
    roots = jnp.concatenate((jnp.array([root]), remaining))
    roots = _polish_smallest_root_from_product(c, roots)
    return roots * scale


def _roots_cubic_complex_no_zeros_impl(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        jnp.all(p.imag == 0),
        lambda p: _roots_cubic_real_as_complex_no_zeros_impl(p.real),
        _roots_cubic_genuinely_complex_no_zeros,
        p,
    )


def _cubic_jvp(p: jax.Array, dp: jax.Array, r: jax.Array, /) -> jax.Array:
    dp = _projective_tangent(p, dp)
    polynomial_tangent = ((dp[0] * r + dp[1]) * r + dp[2]) * r + dp[3]
    derivative = (3 * p[0] * r + 2 * p[1]) * r + p[2]
    return -polynomial_tangent / derivative


@jax.custom_jvp
def _roots_cubic_real_no_zeros(p: jax.Array, /) -> jax.Array:
    return _roots_cubic_real_no_zeros_impl(p)


@_roots_cubic_real_no_zeros.defjvp
def _roots_cubic_real_no_zeros_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (p,), (dp,) = primals, tangents
    r = _roots_cubic_real_no_zeros_impl(p)
    finite = jnp.isfinite(r)
    safe_r = jnp.where(finite, r, jnp.ones_like(r))
    dr = _cubic_jvp(p, dp, safe_r)
    return r, jnp.where(finite, dr, jnp.zeros_like(dr))


@jax.custom_jvp
def _roots_cubic_real_as_complex_no_zeros(p: jax.Array, /) -> jax.Array:
    return _roots_cubic_real_as_complex_no_zeros_impl(p)


@_roots_cubic_real_as_complex_no_zeros.defjvp
def _roots_cubic_real_as_complex_no_zeros_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (p,), (dp,) = primals, tangents
    r = _roots_cubic_real_as_complex_no_zeros_impl(p)
    return r, _cubic_jvp(p, dp, r)


@jax.custom_jvp
def _roots_cubic_complex_no_zeros(p: jax.Array, /) -> jax.Array:
    return _roots_cubic_complex_no_zeros_impl(p)


@_roots_cubic_complex_no_zeros.defjvp
def _roots_cubic_complex_no_zeros_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (p,), (dp,) = primals, tangents
    r = _roots_cubic_complex_no_zeros_impl(p)
    return r, _cubic_jvp(p, dp, r)


def _roots_cubic_real_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_cubic_real_no_zeros,
        lambda p: jnp.append(_roots_quadratic_with_zeros(p[1:]), _nan(p.dtype)),
        p,
    )


def _roots_cubic_real_as_complex_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_cubic_real_as_complex_no_zeros,
        lambda p: jnp.append(
            _roots_quadratic_with_zeros(_as_complex(p[1:])),
            _nan(_as_complex(p).dtype),
        ),
        p,
    )


def _roots_cubic_complex_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_cubic_complex_no_zeros,
        lambda p: jnp.append(_roots_quadratic_with_zeros(p[1:]), _nan(p.dtype)),
        p,
    )


@jax.jit
def _roots_no_zeros_real(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_no_zeros(p)
        case 3:
            return _roots_quadratic_no_zeros(p)
        case 4:
            return _roots_cubic_real_no_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_no_zeros_real_as_complex(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_no_zeros(_as_complex(p))
        case 3:
            return _roots_quadratic_no_zeros(_as_complex(p))
        case 4:
            return _roots_cubic_real_as_complex_no_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=_as_complex(p).dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_no_zeros_complex(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_no_zeros(p)
        case 3:
            return _roots_quadratic_no_zeros(p)
        case 4:
            return _roots_cubic_complex_no_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_with_zeros_real(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_with_zeros(p)
        case 3:
            return _roots_quadratic_with_zeros(p)
        case 4:
            return _roots_cubic_real_with_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_with_zeros_real_as_complex(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_with_zeros(_as_complex(p))
        case 3:
            return _roots_quadratic_with_zeros(_as_complex(p))
        case 4:
            return _roots_cubic_real_as_complex_with_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=_as_complex(p).dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_with_zeros_complex(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear_with_zeros(p)
        case 3:
            return _roots_quadratic_with_zeros(p)
        case 4:
            return _roots_cubic_complex_with_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


def roots(
    p: jax.Array, /, *, strip_zeros: bool = True, real: bool = False
) -> jax.Array:
    """Return the roots of a polynomial of degree at most three.

    Args:
        p: One-dimensional coefficient array in descending order of powers.
        strip_zeros: Remove leading zero coefficients before solving. Set this to
            False under transformations such as ``jit`` or ``vmap`` that require
            a static output shape; missing roots are then represented by NaNs.
        real: Use real arithmetic for real coefficients. Non-real roots are
            represented by NaN placeholders rather than converted to complex
            values. Complex coefficient arrays are rejected when ``real=True``.

    Returns:
        A JAX array containing the roots. Root ordering is not specified.

    Notes:
        Derivatives of individually labelled roots are singular at repeated
        roots. Consequently, automatic differentiation may return non-finite
        tangents or cotangents at exact root collisions.
    """
    if not isinstance(p, (jax.Array, np.ndarray)):
        raise TypeError(f"p must be a jax.Array or np.ndarray; got type {type(p)}")
    if p.ndim != 1:
        raise ValueError(f"p must be a 1D array; got shape {p.shape}")

    is_complex = jnp.issubdtype(p.dtype, jnp.complexfloating)
    if real and is_complex:
        raise TypeError(f"p cannot be complex if real=True; got dtype {p.dtype}")
    if not jnp.issubdtype(p.dtype, jnp.inexact):
        dtype = jnp.float64 if p.dtype.itemsize > 4 else jnp.float32
        p = p.astype(dtype)
    if strip_zeros:
        try:
            p = jnp.trim_zeros(p, trim="f")
        except jax.errors.ConcretizationTypeError as e:
            if isinstance(p, jax.core.Tracer):
                msg = (
                    "radiax.roots: pass strip_zeros=False when using JAX "
                    "transformations that require a static output shape "
                    "(e.g. jit or vmap)"
                )
                raise jax.errors.ConcretizationTypeError(p, msg) from e
            raise
        if real:
            return _roots_no_zeros_real(p)
        if is_complex:
            return _roots_no_zeros_complex(p)
        return _roots_no_zeros_real_as_complex(p)

    if real:
        return _roots_with_zeros_real(p)
    if is_complex:
        return _roots_with_zeros_complex(p)
    return _roots_with_zeros_real_as_complex(p)
