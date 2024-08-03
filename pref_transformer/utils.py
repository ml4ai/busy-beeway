import random
import time
import cloudpickle as pickle
import os
from jax_utils import init_rng
import numpy as np

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed

def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

def save_pickle(obj, filename, output_dir):
    with open(os.path.join(output_dir, filename), 'wb') as fout:
        pickle.dump(obj, fout)

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
