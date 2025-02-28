import jax.numpy as jnp
import numpy as np


def hermgauss(n, dtype=jnp.float32):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = jnp.asarray(x, dtype=dtype), jnp.asarray(w, dtype=dtype)
    return x, w
