import jax.numpy as jnp
from flax import nnx
import optax

class LikelihoodModule(nnx.Module):
    def __call__(self, fx, y):
        return -self.loglik(fx, y)

    def loglik(self, fx, y):
        raise NotImplementedError


class LikGaussian(LikelihoodModule):
    def __init__(self, var):
        self.loss = optax.losses.squared_error
        self.var = var

    def loglik(self, fx, y):
        return -0.5 / self.var * self.loss(fx, y).sum()
