from flax import nnx
import jax
import jax.numpy as jnp

def named_parameters(params_state):
    for i in params_state:
        if isinstance(params_state[i], nnx.State):
            yield from named_parameters(params_state[i])
        else:
            yield i, params_state[i]


class PriorModule(nnx.Module):
    """Generic class of Prior module"""

    def __init__(self):
        self.hyperprior = False

    def __call__(self, net):
        """Compute the negative log likelihood.

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        return -self.logp(net)

    def initialize(self, net, rngs):
        """Initialize neural network's parameters according to the prior.

        Args:
            net: nn.Module, the input network needs to be initialzied.
        """
        params_state = nnx.state(net, nnx.Param)
        for name, param in named_parameters(params_state):
            value = self.sample(name, param, rngs)
            if value is not None:
                param.value = value
        nnx.update(net,params_state)

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        raise NotImplementedError

    def sample(self, name, param, rngs):
        """Sample parameters from prior.

        Args:
            name: str, the name of the parameter.
            param: torch.Parameter, the parameter need to be sampled.
        """
        raise NotImplementedError


class FixedGaussianPrior(PriorModule):
    """Class of Standard Gaussian Prior."""

    def __init__(self, mu=0.0, std=1.0, rngs=nnx.Rngs(0, params=1, dropout=2)):
        """Initialization."""
        super(FixedGaussianPrior, self).__init__()

        self.mu = mu
        self.std = std
        self.rngs = rngs

    def sample(self, name, param):
        """Sample parameters from prior.

        Args:
            name: str, the name of the parameter.
            param: torch.Parameter, the parameter need to be sampled.
        """
        mu, std = self._get_params_by_name(name)

        if (mu is None) and (std is None):
            return None
        rng = self.rngs()
        return mu + std * jax.random.normal(rng, param.shape)

    def _get_params_by_name(self, name):
        """Get the paramters of prior by name."""

        if not (("W" in name) or ("b" in name)):
            return None, None
        else:
            return self.mu, self.std

    def logp(self, net):
        """Compute the log likelihood

        Args:
            net: nn.Module, the input network needs to be evaluated.
        """
        res = 0.0
        params_state = nnx.state(net, nnx.Param)
        for name, param in named_parameters(params_state):
            mu, std = self._get_params_by_name(name)
            if (mu is None) and (std is None):
                continue
            var = std**2
            p_val = param.value
            res -= jnp.sum(((p_val - mu) ** 2) / (2 * var))
        return res
