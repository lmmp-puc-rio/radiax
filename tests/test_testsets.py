import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .testsets import KNOWN_ROOTS, RANDOM_CUBICS
from .utils import assert_roots_match


@pytest.mark.parametrize(("p", "expected"), KNOWN_ROOTS)
def test_known_roots_polyval(p: jax.Array, expected: jax.Array) -> None:
    y = jnp.polyval(p, expected)

    assert isinstance(y, jax.Array)

    assert y == pytest.approx(0)


@pytest.mark.parametrize(("p", "expected"), KNOWN_ROOTS)
def test_known_vs_np(p: jax.Array, expected: jax.Array) -> None:
    r = np.roots(p)

    assert isinstance(r, np.ndarray)
    assert r.dtype == (
        complex if np.issubdtype(expected.dtype, np.complexfloating) else float
    )

    assert_roots_match(r, expected)


@pytest.mark.parametrize(("p", "expected"), KNOWN_ROOTS)
@pytest.mark.parametrize(
    ("strip_zeros", "jit"), [(True, False), (False, False), (False, True)]
)
def test_known_vs_jnp(
    p: jax.Array, expected: jax.Array, strip_zeros: bool, jit: bool
) -> None:
    if jit:
        r = jax.jit(lambda p: jnp.roots(p, strip_zeros=strip_zeros))(p)
    else:
        r = jnp.roots(p, strip_zeros=strip_zeros)

    assert isinstance(r, jax.Array)
    assert r.dtype == complex

    assert_roots_match(r, expected)


def test_random_jnp_polyval() -> None:
    vmapped_roots = jax.jit(jax.vmap(lambda p: jnp.roots(p, strip_zeros=False)))(
        RANDOM_CUBICS
    )
    assert isinstance(vmapped_roots, jax.Array)
    assert vmapped_roots.dtype == complex

    vmapped_polyval = jax.jit(jax.vmap(jnp.polyval))(RANDOM_CUBICS, vmapped_roots)
    assert isinstance(vmapped_polyval, jax.Array)
    assert vmapped_polyval.dtype == complex

    assert vmapped_polyval == pytest.approx(0, abs=1e-11)
