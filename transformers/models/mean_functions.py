from flax import nnx
import jax.numpy as jnp

class MeanFunction(nnx.Module):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """

    def __call__(self, X):
        raise NotImplementedError("Implement the forward method for this mean function")

    def __add__(self, other):
        return MeanAdditive(self, other)

    def __mul__(self, other):
        return MeanProduct(self, other)


class Zero(MeanFunction):
    def __call__(self, X):
        return jnp.zeros((X.shape[0], 1), dtype=X.dtype)
