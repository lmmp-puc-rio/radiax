import itertools

import jax
import numpy as np
import pytest


def assert_roots_match(
    actual: jax.Array | np.ndarray,
    expected: jax.Array | np.ndarray,
    /,
    *,
    abs: float = 1e-12,
    rel: float = 1e-6,
) -> None:
    assert actual.ndim >= 1
    assert expected.ndim >= 1
    assert actual.shape[-1] == expected.shape[-1]

    if np.issubdtype(expected.dtype, np.complexfloating) and not np.issubdtype(
        actual.dtype, np.complexfloating
    ):
        expected = np.where(np.isreal(expected), expected.real, np.nan)

    permutations = np.array(list(itertools.permutations(range(expected.shape[-1]))))

    matches = (
        np.isclose(
            actual[..., None, :],
            expected[..., permutations],
            atol=abs,
            rtol=rel,
            equal_nan=True,
        )
        .all(axis=-1)
        .any(axis=-1)
    )

    if not matches.all():
        pytest.fail(
            f"some roots do not match: {actual[~matches]} vs {expected[~matches]}"
        )
