import functools
from typing import Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from tensorflow_probability.substrates import jax as tfp
from transformers.training.utils import prng_to_raw, raw_to_prng

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


def load_IQLPolicy(model_dir, chkptr, on_cpu=False):
    model_args = chkptr.restore(
        model_dir,
        args=ocp.args.Composite(
            model_args=ocp.args.ArrayRestore(),
        ),
    )
    model_args = model_args["model_args"]
    rng_key = jax.random.key(int(model_args[9]))
    rng_key, _ = jax.random.split(rng_key, 2)
    rng_subkey1, rng_subkey2, rng_subkey3, rng_subkey4 = jax.random.split(rng_key, 4)
    rngs = nnx.Rngs(
        rng_subkey1, params=rng_subkey2, dropout=rng_subkey3, sample=rng_subkey4
    )

    model = NormalTanhPolicy(
        state_dim=int(model_args[0]),
        mlp_output_dim=int(model_args[1]),
        hidden_dims=[int(x) for x in model_args[10:]],
        action_dim=int(model_args[2]),
        state_dependent_std=True if model_args[3] else False,
        dropout_rate=None if model_args[4] == -1 else model_args[4],
        log_std_scale=model_args[5],
        log_std_min=model_args[6],
        log_std_max=model_args[7],
        tanh_squash_distribution=True if model_args[8] else False,
        rngs=rngs,
    )
    prng_to_raw(model)
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)
    # Loads onto first cpu found
    if on_cpu:

        def set_sharding(var):
            var.sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
            return var

        abstract_state = jax.tree.map(set_sharding, abstract_state)
    model_state = chkptr.restore(
        model_dir,
        args=ocp.args.Composite(
            model_state=ocp.args.StandardRestore(abstract_state),
        ),
    )
    model = nnx.merge(graphdef, model_state["model_state"])
    raw_to_prng(model)
    return model
