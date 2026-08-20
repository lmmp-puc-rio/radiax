import jax
import jax.core
import jax.numpy as jnp
import numpy as np


def _roots_linear(p: jax.Array, /) -> jax.Array:
    return jnp.array([-p[1] / p[0]])


def _roots_quadratic_no_zeros(p: jax.Array, /) -> jax.Array:
    discriminant = p[1] ** 2 - 4 * p[0] * p[2]
    sqrt_discriminant = jnp.sqrt(discriminant)

    return jnp.array(
        [
            (-p[1] + sqrt_discriminant) / (2 * p[0]),
            (-p[1] - sqrt_discriminant) / (2 * p[0]),
        ]
    )


def _roots_quadratic_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_quadratic_no_zeros,
        lambda p: jnp.append(_roots_linear(p[1:]), jnp.nan),
        p,
    )


def _roots_cubic_no_zeros(p: jax.Array, /) -> jax.Array:
    p /= p[0]

    shift = p[1] / 3
    p_ = p[2] - p[1] * shift
    q = p[3] + 2 * shift**3 - p[2] * shift

    P = p_ / 3
    Q = q / 2

    P, Q = jax.lax.optimization_barrier((P, Q))

    delta = Q**2 + P**3

    if not jnp.issubdtype(delta.dtype, jnp.complexfloating):

        def three_real(_: jax.Array, P: jax.Array, Q: jax.Array) -> jax.Array:
            def triple_root() -> jax.Array:
                return jnp.zeros(3, dtype=P.dtype)

            def general() -> jax.Array:
                cos3theta = -Q / jnp.sqrt(-(P**3))
                cos3theta = jnp.clip(cos3theta, -1, 1)

                theta = jnp.arccos(cos3theta) / 3
                r = 2 * jnp.sqrt(-P)

                r1 = r * jnp.cos(theta)
                r2 = r * jnp.cos(theta + 2 * jnp.pi / 3)
                r3 = r * jnp.cos(theta + 4 * jnp.pi / 3)

                return jnp.array([r1, r2, r3])

            return jax.lax.cond(P == 0, triple_root, general)

        def one_real(delta: jax.Array, _: jax.Array, Q: jax.Array) -> jax.Array:
            delta_sqrt = jnp.sqrt(delta)
            u = jnp.cbrt(-Q + delta_sqrt)
            v = jnp.cbrt(-Q - delta_sqrt)
            root = u + v
            return jnp.array([root, jnp.nan, jnp.nan])

        shifted_roots = jax.lax.cond(delta <= 0, three_real, one_real, delta, P, Q)
        return shifted_roots - shift

    delta_sqrt = jnp.sqrt(delta)

    u = (-Q + delta_sqrt) ** (1 / 3)
    v = jnp.where(
        u != 0,
        -P / u,
        (-Q - delta_sqrt) ** (1 / 3),
    )

    omega = jnp.exp(2j * jnp.pi / 3)

    r1 = u + v
    r2 = omega * u + omega**2 * v
    r3 = omega**2 * u + omega * v

    return jnp.array([r1, r2, r3]) - shift


def _roots_cubic_with_zeros(p: jax.Array, /) -> jax.Array:
    return jax.lax.cond(
        p[0] != 0,
        _roots_cubic_no_zeros,
        lambda p: jnp.append(_roots_quadratic_with_zeros(p[1:]), jnp.nan),
        p,
    )


@jax.jit
def _roots_no_zeros(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear(p)
        case 3:
            return _roots_quadratic_no_zeros(p)
        case 4:
            return _roots_cubic_no_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


@jax.jit
def _roots_with_zeros(p: jax.Array, /) -> jax.Array:
    match p.size:
        case 2:
            return _roots_linear(p)
        case 3:
            return _roots_quadratic_with_zeros(p)
        case 4:
            return _roots_cubic_with_zeros(p)
        case 0 | 1:
            return jnp.array([], dtype=p.dtype)
        case _:
            raise NotImplementedError(
                "radiax.roots currently supports polynomials of degree <= 3"
            )


def roots(
    p: jax.Array, /, *, strip_zeros: bool = True, real: bool = False
) -> jax.Array:
    if not isinstance(p, (jax.Array, np.ndarray)):
        raise TypeError(f"p must be a jax.Array or np.ndarray; got type {type(p)}")
    if p.ndim != 1:
        raise ValueError(f"p must be a 1D array; got shape {p.shape}")

    if real:
        if jnp.issubdtype(p.dtype, jnp.complexfloating):
            raise TypeError(f"p cannot be complex if real=True; got dtype {p.dtype}")
    elif not jnp.issubdtype(p.dtype, jnp.complexfloating):
        p = p.astype(jnp.complex128 if p.dtype.itemsize > 4 else jnp.complex64)

    if strip_zeros:
        try:
            p = jnp.trim_zeros(p, trim="f")
        except jax.errors.ConcretizationTypeError as e:
            if isinstance(p, jax.core.Tracer):
                raise jax.errors.ConcretizationTypeError(
                    p,
                    "radiax.roots: pass strip_zeros=False for compatibility with JIT compilation",
                ) from e
            raise
        return _roots_no_zeros(p)
    else:
        return _roots_with_zeros(p)
