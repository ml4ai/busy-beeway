import jax.numpy as jnp

class GridGenerator(object):
    def __init__(self, x_min, x_max, input_dim=1):
        self.x_min = x_min
        self.x_max = x_max
        self.input_dim = input_dim

    def get(self, n_data):
        X = jnp.linspace(self.x_min, self.x_max, n_data)
        return X.reshape((-1, self.input_dim))