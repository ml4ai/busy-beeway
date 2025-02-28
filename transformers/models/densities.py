import jax.numpy as jnp

def gaussian(x, mu, var):
    return float(-0.5 * (jnp.log(2 * jnp.pi) + jnp.log(var) + ((x-mu)**2)/var))

def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.
    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    if d.ndim == 1:
        d = jnp.expand_dims(d,1)
    alpha, _ = jnp.linalg.solve(d, L)
    alpha = alpha.squeeze(1)
    num_col = 1 if x.ndim == 1 else x.shape[1]
    num_dims = x.shape[0]
    ret = - 0.5 * num_dims * num_col * float(jnp.log(2 * jnp.pi))
    ret += - num_col * jnp.log(jnp.diag(L)).sum()
    ret += - 0.5 * (alpha**2).sum()
    # ret = - 0.5 * (alpha**2).mean()
    return ret