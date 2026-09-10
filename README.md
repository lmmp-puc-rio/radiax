<div align="center">
  <a href="https://github.com/lmmp-puc-rio/radiax"><img src="https://github.com/lmmp-puc-rio/radiax/raw/main/logo.png" alt="radiax" width="400"/></a>

  **Analytic polynomial roots within JAX**

  JIT-compatible fully analytic variant of [`jax.numpy.roots`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roots.html) for small-degree polynomials

  [![CI](https://github.com/lmmp-puc-rio/radiax/actions/workflows/ci.yml/badge.svg)](https://github.com/lmmp-puc-rio/radiax/actions/workflows/ci.yml)
  [![Codecov](https://codecov.io/gh/lmmp-puc-rio/radiax/branch/main/graph/badge.svg)](https://codecov.io/gh/lmmp-puc-rio/radiax)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
  [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
  [![Publish](https://github.com/lmmp-puc-rio/radiax/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/lmmp-puc-rio/radiax/actions/workflows/pypi-publish.yml)
  [![PyPI](https://img.shields.io/pypi/v/radiax)](https://pypi.org/project/radiax/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/radiax)](https://pypi.org/project/radiax/)
</div>


## Overview

`radiax.roots` is a drop-in replacement for [`jax.numpy.roots`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roots.html) for polynomials of up to the third degree<sup>1</sup>. It is fully analytic, with support for complex roots and coefficients. It is [JIT](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)-compatible<sup>2</sup> and has been benchmarked to be around 20x faster than `jax.numpy.roots` for cubic polynomials<sup>3</sup>.

## Installation

```bash
pip install radiax
```

## Usage

```python
import jax.numpy as jnp
from radiax import roots

# Coefficients of the polynomial: x**3 - 6*x**2 + 11*x - 6
coeffs = jnp.array([1, -6, 11, -6])

# Find the roots of the polynomial
r = roots(coeffs)
print(r)
```

## Footnotes

<sup>1</sup> Support for degree 4 (quartic) polynomials has not been implemented yet.

<sup>2</sup> Just like with [`jax.numpy.roots`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roots.html), the optional keyword argument `strip_zeros` must be set to `False` for JIT-compatibility.

<sup>3</sup> **radiax** 0.1.3 running on an Intel Core i7-14700 CPU (JAX 0.11.1, Python 3.14.7, Ubuntu 24.04.5).

---

<div align="center">
  <a href="https://lmmp.mec.puc-rio.br/"><img src="http://lmmp.mec.puc-rio.br/wp-content/uploads/2022/02/cropped-cropped-lmmp_200x65.png" alt="LMMP" width="200"/></a>
</div>
