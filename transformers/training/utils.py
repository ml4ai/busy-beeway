import os
import random
import time
from pathlib import Path
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import numpy
from flax import nnx


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


def save_model(model, model_args, file_tag, save_dir, chkptr):
    _, state = nnx.split(model)
    chkptr.save(
        save_dir + "/" + file_tag + ".ckpt",
        args=ocp.args.Composite(
            model_state=ocp.args.StandardSave(state),
            model_args=ocp.args.ArraySave(model_args),
        ),
        force=True,
    )


def ensure_dir(dirname):
    """Check whether a given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


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
