import functools
from typing import Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from transformers.models.value_net import Identity, default_init, MLP


class NormalTanhPolicy(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        mlp_output_dim: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        state_dependent_std: bool = True,
        dropout_rate: Optional[float] = None,
        log_std_scale: float = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2,
        tanh_squash_distribution: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.mlp = MLP(
            state_dim,
            mlp_output_dim,
            hidden_dims,
            activation_final=True,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )
        self.mean_linear = nnx.Linear(
            mlp_output_dim,
            action_dim,
            kernel_init=default_init(),
            rngs=rngs,
        )

        if state_dependent_std:
            self.get_log_stds = nnx.Linear(
                mlp_output_dim,
                action_dim,
                kernel_init=default_init(log_std_scale),
                rngs=rngs,
            )
        else:
            self.log_stds = nnx.Param(jnp.zeros(action_dim))
            self.get_log_stds = lambda x: self.log_stds.value

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if tanh_squash_distribution:
            self.trans_mean = Identity()
            self.get_dist = lambda dist: tfd.TransformedDistribution(
                distribution=dist, bijector=tfb.Tanh()
            )
        else:
            self.trans_mean = nnx.tanh
            self.get_dist = lambda dist: dist

    def __call__(
        self,
        states: jax.Array,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        outputs = self.mlp(states, training=training)

        means = self.mean_linear(outputs)

        log_stds = self.get_log_stds(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        means = self.trans_mean(means)

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        return self.get_dist(base_dist)


@nnx.jit
def sample_actions(
    actor: nnx.Module,
    states: jax.Array,
    temperature: float = 1.0,
    rngs: nnx.Rngs = nnx.Rngs(sample=4),
) -> jax.Array:
    dist = actor(states, temperature)
    key = rngs.sample()
    return dist.sample(seed=key)
