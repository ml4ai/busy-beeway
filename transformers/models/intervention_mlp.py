import jax.numpy as jnp
from flax import linen as nn

import transformers.models.ops


class MLP(nn.Module):
    observation_dim: int = 327
    embd_dim: int = 64
    activation: str = "relu"
    embd_dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(features=self.embd_dim)(x)
        x = ops.apply_activation(x, activation=self.activation)
        x = nn.Dropout(rate=embd_dropout)(x, deterministic=not training)
        x = nn.Dense(features=1)(x)
        return x
