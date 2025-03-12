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


class GaussianLinearReparameterization(nnx.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        W_std: Optional[float] = None,
        b_std: Optional[float] = None,
        scaled_variance: bool = True,
        prior_per: str = "layer",
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
    ):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
            W_std: float, the initial value of
                the standard deviation of the weights.
            b_std: float, the initial value of
                the standard deviation of the biases.
            prior_per: str, indicates whether using different prior for
                each parameter, option `parameter`, or use the share the
                prior for all parameters in the same layer, option `layer`.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.rngs = rngs

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.0
            else:
                W_std = 1.0 / jnp.sqrt(self.n_in)
        if b_std is None:
            b_std = 1.0

        if prior_per == "layer":
            W_shape, b_shape = 1, 1
        elif prior_per == "parameter":
            W_shape, b_shape = (self.n_in, self.n_out), self.n_out
        else:
            raise ValueError("Accepted values: `parameter` or `layer`")

        self.W_mu = 0.0
        self.b_mu = 0.0

        self.W_std = nnx.Param(jnp.ones(W_shape) * W_std)
        self.b_std = nnx.Param(jnp.ones(b_shape) * b_std)

    def __call__(
        self,
        X: jax.Array,
    ):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        rng = self.rngs()
        W_std = self.W_std.value
        W = self.W_mu + nnx.softplus(W_std) * jax.random.normal(
            rng, (self.n_in, self.n_out)
        )
        if self.scaled_variance:
            W = W / jnp.sqrt(self.n_in)

        rng = self.rngs()
        b_std = self.b_std.value
        b = self.b_mu + nnx.softplus(b_std) * jax.random.normal(rng, (self.n_out,))

        return X @ W + b

    def sample_predict(self, X, n_samples):
        """Makes predictions using a set of sampled weights.

        Args:
            X: torch.tensor, [n_samples, batch_size, input_dim], the input
                data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [n_samples, batch_size, output_dim], the output data.
        """
        rng = self.rngs()
        W_std = self.W_std.value
        Ws = self.W_mu + nnx.softplus(W_std) * jax.random.normal(
            rng,
            (n_samples, self.n_in, self.n_out),
        )

        if self.scaled_variance:
            Ws = Ws / jnp.sqrt(self.n_in)

        rng = self.rngs()
        b_std = self.b_std.value
        bs = self.b_mu + nnx.softplus(b_std) * jax.random.normal(
            rng,
            (n_samples, 1, self.n_out),
        )

        return X @ Ws + bs


def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nnx.BatchNorm(
            input_dim,
            epsilon=0,
            momentum=0.0,
        )
    elif norm_layer is None:
        return Identity()


class GaussianMLPReparameterization(nnx.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation_fn,
        W_std=None,
        b_std=None,
        scaled_variance=True,
        norm_layer=None,
        rngs: nnx.Rngs = nnx.Rngs(0, params=1, dropout=2),
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

        if b_std is None:
            b_std = W_std

        # Initialize layers
        self.layers = [
            GaussianLinearReparameterization(
                input_dim,
                hidden_dims[0],
                W_std,
                b_std,
                scaled_variance=scaled_variance,
                rngs=rngs,
            )
        ]

        self.norm_layers = [init_norm_layer(hidden_dims[0], self.norm_layer)]

        for i in range(1, len(hidden_dims)):
            self.layers.append(
                GaussianLinearReparameterization(
                    hidden_dims[i - 1],
                    hidden_dims[i],
                    W_std,
                    b_std,
                    scaled_variance=scaled_variance,
                    rngs=rngs,
                )
            )
            self.norm_layers.append(init_norm_layer(hidden_dims[i], self.norm_layer))

        self.output_layer = GaussianLinearReparameterization(
            hidden_dims[-1],
            output_dim,
            W_std,
            b_std,
            scaled_variance=scaled_variance,
            rngs=rngs,
        )

    def __call__(self, X):
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

        return X

    def sample_functions(self, X, n_samples):
        """Performs predictions using `n_samples` set of weights.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [batch_size, n_samples, output_dim], the output
            data.
        """
        X = X.reshape(-1, self.input_dim)
        X = jnp.tile(jnp.expand_dims(X, 0), [n_samples, 1, 1])
        for linear_layer, norm_layer in zip(list(self.layers), list(self.norm_layers)):
            if self.norm_layer is None:
                X = self.activation_fn(linear_layer.sample_predict(X, n_samples))
            else:
                X = linear_layer.sample_predict(X, n_samples)
                out = jnp.zeros_like(X, dtype=X.dtype)
                for i in range(n_samples):
                    out[i, :, :] = norm_layer(X[i, :, :])
                X = self.activation_fn(out)

        X = self.output_layer.sample_predict(X, n_samples)
        X = X.transpose(1, 0, 2)

        return X
