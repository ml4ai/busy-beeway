from jax import random, numpy as jnp
import jax
import test_nn
from jax_utils import init_rng, next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss

test = test_nn.T_NN()
init_rng(2020)

test_params = test.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, 29)),
            jnp.zeros((10, 25, 8)),
            jnp.ones((10, 25), dtype=jnp.int32)
        )
s = random.normal(next_rng(),(256,100,29))
a = random.normal(next_rng(),(256,100,8))
t = jnp.ones((256,100),dtype=jnp.int32)
i = test.apply(test_params,s,a,t)

print(test_params['params']['Dense_0']['kernel'].shape)
