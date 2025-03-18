import jax.numpy as jnp
from flax import nnx


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, X: jax.Array, *args, **kwargs):
        return X

class Q_MLP(nnx.Module):
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 3,
        hidden_dims: list[int] = [256, 256],
        orthogonal_init: bool = False,
        activations: str = "relu",
        activation_final: str = "none",
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        # Setup activation function
        options = {
            "cos": jnp.cos,
            "tanh": nnx.tanh,
            "relu": nnx.relu,
            "softplus": nnx.softplus,
            "rbf": rbf,
            "linear": linear,
            "sin": jnp.sin,
            "leaky_relu": nnx.leaky_relu,
            "swish": nnx.swish,
            "none": Identity,
        }
        self.activations = options[activations]

        self.activation_final = options[activation_final]

        # Initialize layers
        if orthogonal_init:
            self.layers = [
                nnx.Linear(
                    state_dim + action_dim,
                    hidden_dims[0],
                    kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
            ]

            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    nnx.Linear(
                        hidden_dims[i - 1],
                        hidden_dims[i],
                        kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2.0)),
                        bias_init=nnx.initializers.zeros_init(),
                        rngs=rngs,
                    )
                )

            self.output_layer = nnx.Linear(
                hidden_dims[-1],
                1,
                kernel_init=nnx.initializers.orthogonal(1e-2),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        else:
            self.layers = [
                nnx.Linear(
                    state_dim + action_dim,
                    hidden_dims[0],
                    rngs=rngs,
                )
            ]

            for i in range(1, len(hidden_dims)):
                self.layers.append(
                    nnx.Linear(
                        hidden_dims[i - 1],
                        hidden_dims[i],
                        rngs=rngs,
                    )
                )

            self.output_layer = nnx.Linear(
                hidden_dims[-1],
                1,
                kernel_init=nnx.initializers.variance_scaling(
                    1e-2, "fan_in", "uniform"
                ),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        for linear_layer in self.layers:
            x = self.activations(linear_layer(x))

        return jnp.squeeze(self.activation_final(self.output_layer(x)),-1)

        