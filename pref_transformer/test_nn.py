from flax import linen as nn
import jax.numpy as jnp

class T_NN(nn.Module):
    @nn.compact
    def __call__(self, s, a, t):
        batch_size = s.shape[0]
        seq_length = s.shape[1]
        s = nn.Dense(features=256)(s)
        a = nn.Dense(features=256)(a)
        t = nn.Embed(num_embeddings=1001, features=256)(t)

        s = s + t
        a = a + t
        i = jnp.stack([a,s],axis=1).transpose(0,2,1,3).reshape(batch_size,2*seq_length,256)
        return i
