import abc
import jax.numpy as jnp
import jax
from flax import nnx


class ParamWithPrior(nnx.Param):
    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def log_jacobian(self):
        pass

    @abc.abstractmethod
    def untransform(t):
        pass

    def __init__(self, val, prior=None, dtype=jnp.float32):
        if jnp.isscalar(val):
            val = jnp.asarray([val], dtype=dtype)
        raw = self.untransform(val)
        super(ParamWithPrior, self).__init__(raw)
        self.prior = prior
        self.dtype = dtype

    def set(self, t):
        if jnp.isscalar(t):
            t = jnp.asarray(t, dtype=self.dtype)
        self.value = self.untransform(t)

    def get_prior(self):
        if self.prior is None:
            return 0.0

        log_jacobian = self.log_jacobian()  # (unconstrained_tensor)
        logp_var = self.prior.logp(self.get())
        return log_jacobian + logp_var


class PositiveParam(ParamWithPrior):  # log(1+exp(r))
    def untransform(self, t):
        return jax.lax.stop_gradient(jnp.log(jnp.exp(t) - 1))

    def get(self):
        return jnp.log(1 + jnp.exp(self.value))

    def log_jacobian(self):
        return -(nnx.softplus(-self.value))
