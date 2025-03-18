from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from transformers.models.activation_fns import *


class Identity(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, X: jax.Array, *args, **kwargs):
        return X


class Linear(nnx.Module):
    def __init__(
        self, n_in, n_out, scaled_variance=True, rngs=nnx.Rngs(0, params=1, dropout=2)
    ):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.rngs = rngs

        # Initialize the parameters
        self.W = nnx.Param(jnp.zeros((self.n_in, self.n_out)))
        self.b = nnx.Param(jnp.zeros(self.n_out))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0
        if not self.scaled_variance:
            std = std / jnp.sqrt(self.n_in)
        W_init = nnx.initializers.normal(std)
        W_rng = self.rngs.params()
        self.W.value = W_init(W_rng, (self.n_in, self.n_out))

        b_init = nnx.initializers.constant(0)
        b_rng = self.rngs.params()
        self.b.value = b_init(b_rng, (self.n_out,))

    def __call__(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W.value
        if self.scaled_variance:
            W = W / jnp.sqrt(self.n_in)
        b = self.b.value
        return X @ W + b


def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nnx.BatchNorm(
            input_dim,
            epsilon=0,
            momentum=0.0,
        )
    elif norm_layer is None:
        return Identity()


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation_fn,
        scaled_variance=True,
        norm_layer=None,
        task="regression",
        rngs=nnx.Rngs(0, params=1, dropout=2),
    ):
        """Initialization.

        Args:
            input_dim: int, the size of the input data.
            output_dim: int, the size of the output data.
            hidden_dims: list of int, the list containing the size of
                hidden layers.
            activation_fn: str, the name of activation function to be used
                in the network.
            W_std: float, the initial value of the logarithm of
                the standard deviation of the weights.
            b_std: float, the initial value of the logarithm of
                the standard deviation of the biases.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer
        self.task = task
        self.rngs = rngs

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
        }
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        # Initialize layers
        self.layers = [
            Linear(
                input_dim,
                hidden_dims[0],
                scaled_variance=scaled_variance,
                rngs=rngs,
            )
        ]

        self.norm_layers = [init_norm_layer(hidden_dims[0], self.norm_layer)]

        for i in range(1, len(hidden_dims)):
            self.layers.append(
                Linear(
                    hidden_dims[i - 1],
                    hidden_dims[i],
                    scaled_variance=scaled_variance,
                    rngs=rngs,
                )
            )
            self.norm_layers.append(init_norm_layer(hidden_dims[i], self.norm_layer))

        self.output_layer = Linear(
            hidden_dims[-1],
            output_dim,
            scaled_variance=scaled_variance,
            rngs=rngs,
        )

    def reset_parameters(self):
        for m in self.iter_modules():
            if isinstance(m, Linear):
                m.reset_parameters()

    def __call__(self, X, log_softmax=False):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            sample: boolean, whether or not perform forward pass using
                sampled weights.

        Returns:
            torch.tensor, [batch_size, output_dim], the output data.
        """
        X = X.reshape(-1, self.input_dim)

        for linear_layer, norm_layer in zip(list(self.layers), list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)
        if (self.task == "classification") and log_softmax:
            X = nnx.log_softmax(X, axis=1)
        return X

    def predict(self, X):
        """Performs predictions using `n_samples` set of weights.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [batch_size, n_samples, output_dim], the output
            data.
        """
        self.eval()
        if self.task == "classification":
            return jnp.exp(self(X, log_softmax=True))
        else:
            return self(X, log_softmax=False)
