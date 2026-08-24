import jax
import jax.numpy as jnp
import pytest

from radiax import roots


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
