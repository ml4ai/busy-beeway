from flax import linen as nn
import jax.numpy as jnp
import ops

class MLP(nn.Module):
    observation_dim: int = 308
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(features=128)(x)
        x = ops.apply_activation(x, activation="relu")
        x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        x = nn.Dense(features=1)(x)
        return x