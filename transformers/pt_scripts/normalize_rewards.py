import argparse
import os
import sys

import h5py
import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Adds a set of rewards to dataset given a reward function (overwrites if needed). \nAlso removes observations that cause NAN rewards",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "participants",
        metavar="P",
        type=str,
        help="Participant ID or a txt file listing Participant IDs",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="Directory with data",
    )
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
    p_id = os.path.expanduser(args.participants)
    if p_id.endswith(".txt"):
        P = load_list(p_id)
        max_ep_length = -jnp.inf
        max_ep_rtn = -jnp.inf
        min_ep_rtn = jnp.inf
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                max_ep_length = max(max_ep_length, f["max_ep_length"][()])
                max_ep_rtn = max(max_ep_rtn, jnp.max(f["ep_returns"][:]))
                min_ep_rtn = min(min_ep_rtn, jnp.min(f["ep_returns"][:]))

        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                rewards = f["rewards"][:]
                n_rewards = max_ep_length * (
                    (rewards - max_ep_rtn) / (max_ep_rtn - min_ep_rtn)
                )
                if "n_rewards" in f:
                    del f["n_rewards"]
                f.create_dataset("n_rewards", data=n_rewards, chunks=True)
    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
            max_ep_length = f["max_ep_length"][()]
            max_ep_rtn = jnp.max(f["ep_returns"][:])
            min_ep_rtn = jnp.min(f["ep_returns"][:])

            rewards = f["rewards"][:]
            n_rewards = max_ep_length * (
                (rewards - max_ep_rtn) / (max_ep_rtn - min_ep_rtn)
            )
            if "n_rewards" in f:
                del f["n_rewards"]
            f.create_dataset("n_rewards", data=n_rewards, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
