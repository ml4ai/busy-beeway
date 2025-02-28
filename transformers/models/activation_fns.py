__all__ = ['rbf', 'linear']

import jax.numpy as jnp


# RBF function
rbf = lambda x: jnp.exp(-x**2)

# Linear function
linear = lambda x: x
