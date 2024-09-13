import random
import time
import cloudpickle as pickle
import os
import jax.numpy as jnp
import jax


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# Memory Mapped Arrays get loaded into physical memory here. The indices must be sorted
# to load the batch into memory. The batch can then be shuffled using the rng.
def index_batch(batch, indices, rng_key=None):
    if rng_key is None:
        indexed = {}
        for key in batch.keys():
            indexed[key] = batch[key][jnp.sort(indices), ...]
        return indexed
    else:
        indexed = {}
        for key in batch.keys():
            shuffled_idx = jax.random.permutation(rng_key, indices.shape[0])
            indexed[key] = batch[key][jnp.sort(indices), ...][shuffled_idx, ...]
        return indexed


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def save_pickle(obj, filename, output_dir):
    with open(os.path.join(output_dir, filename), "wb") as fout:
        pickle.dump(obj, fout)


def load_pickle(filename):
    with open(os.path.expanduser(filename), "rb") as fin:
        return pickle.load(fin)


class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time
