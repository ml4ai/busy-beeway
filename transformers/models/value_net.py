from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, X: jax.Array, *args, **kwargs):
        return X


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nnx.initializers.orthogonal(scale)


def init_dropout_layer(dropout_rate=None, rngs=nnx.Rngs(0, params=1, dropout=2)):
    if dropout_rate is not None:
        return nnx.Dropout(dropout_rate, rngs=rngs)
    else:
        return Identity()


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activations: Callable[[jax.Array], jax.Array] = nnx.relu,
        activation_final: bool = False,
        dropout_rate: Optional[float] = None,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.layers = [
            nnx.Linear(
                input_dim,
                hidden_dims[0],
                kernel_init=default_init(),
                rngs=rngs,
            )
        ]
        self.dropout_layers = [init_dropout_layer(self.dropout_rate, rngs=rngs)]
        for i in range(1, self.hidden_dims):
            self.layers.append(
                nnx.Linear(
                    hidden_dims[i - 1],
                    hidden_dims[i],
                    kernel_init=default_init(),
                    rngs=rngs,
                )
            )
            self.dropout_layers.append(init_dropout_layer(self.dropout_rate, rngs=rngs))

        self.output_layer = nnx.Linear(
            hidden_dims[-1],
            output_dim,
            kernel_init=default_init(),
            rngs=rngs,
        )
        self.activations = activations
        if activation_final:
            self.final_activation = activations
            self.final_dropout = init_dropout_layer(self.dropout_rate, rngs=rngs)
        else:
            self.final_activation = Identity()
            self.final_dropout = Identity()

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        for linear_layer, drop_layer in zip(
            list(self.layers), list(self.dropout_layers)
        ):
            x = drop_layer(
                self.activations(linear_layer(x)), deterministic=not training
            )

        x = self.final_dropout(
            self.final_activation(self.output_layer(x)), deterministic=not training
        )

        return x


class ValueCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int],
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.mlp = MLP(state_dim, hidden_dims, 1, rngs=rngs)

    def __call__(self, states: jax.Array) -> jax.Array:
        critic = self.mlp(states)
        return jnp.squeeze(critic, -1)


class Critic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activations: Callable[[jax.Array], jax.Array] = nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.mlp = MLP(
            state_dim + action_dim,
            hidden_dims,
            1,
            activations=activations,
            rngs=rngs,
        )

    def __call__(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        inputs = jnp.concatenate([states, actions], -1)
        critic = self.mlp(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        activations: Callable[[jax.Array], jax.Array] = nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        self.critic1 = Critic(
            state_dim, action_dim, hidden_dims, activations=activations, rngs=rngs
        )

        self.critic2 = Critic(
            state_dim, action_dim, hidden_dims, activations=activations, rngs=rngs
        )

    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        critic1 = self.critic1(
            states, actions
        )
        critic2 = self.critic2(
            states, actions
        )
        return critic1, critic2
