import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import h5py

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter
from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Normalizes rewards (and state)",
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
    parser.add_argument(
        "-n",
        "--normalize_states",
        action="store_true",
        help="Normalize states",
    )
    parser.add_argument(
        "-e",
        "--eps",
        type=float,
        default=1e-3,
        help="Ensures no zero division for normalizing states",
    )
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
    p_id = os.path.expanduser(args.participants)
    if p_id.endswith(".txt"):
        P = load_list(p_id)
        max_ep_length = -np.inf
        max_ep_rtn = -np.inf
        min_ep_rtn = np.inf
        if args.normalize_states:
            numeric_states = []
            numeric_next_states = []
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                max_ep_length = max(max_ep_length, f["max_ep_length"][()])
                max_ep_rtn = max(max_ep_rtn, np.max(f["ep_returns"][:]))
                min_ep_rtn = min(min_ep_rtn, np.min(f["ep_returns"][:]))
                if args.normalize_states:
                    numeric_states.append(f["states"][:, -4:])
                    numeric_next_states.append(f["next_states"][:, -4:])
        if args.normalize_states:
            numeric_states = np.concatenate(numeric_states)
            numeric_next_states = np.concatenate(numeric_next_states)
            mean_s = numeric_states.mean(0)
            std_s = numeric_states.std(0) + args.eps
            mean_n_s = numeric_next_states.mean(0)
            std_n_s = numeric_next_states.std(0) + args.eps
            del numeric_states
            del numeric_next_states
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                rewards = f["rewards"][:]
                n_rewards = max_ep_length * (
                    (rewards - max_ep_rtn) / (max_ep_rtn - min_ep_rtn)
                )
                if "n_rewards" in f:
                    del f["n_rewards"]
                f.create_dataset("n_rewards", data=n_rewards, chunks=True)

                if args.normalize_states:
                    states = f["states"][:]
                    states[:, -4:] = (states[:, -4:] - mean_s) / std_s
                    next_states = f["next_states"][:]
                    next_states[:, -4:] = (next_states[:, -4:] - mean_n_s) / std_n_s

                    if "states" in f:
                        del f["states"]
                        f.create_dataset("states", data=states, chunks=True)

                    if "next_states" in f:
                        del f["next_states"]
                        f.create_dataset("next_states", data=next_states, chunks=True)

    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
            max_ep_length = f["max_ep_length"][()]
            max_ep_rtn = np.max(f["ep_returns"][:])
            min_ep_rtn = np.min(f["ep_returns"][:])

            rewards = f["rewards"][:]
            n_rewards = max_ep_length * (
                (rewards - max_ep_rtn) / (max_ep_rtn - min_ep_rtn)
            )
            if "n_rewards" in f:
                del f["n_rewards"]
            f.create_dataset("n_rewards", data=n_rewards, chunks=True)

            if args.normalize_states:
                states = f["states"][:]
                states[:, -4:] = (states[:, -4:] - states[:, -4:].mean(0)) / (
                    states[:, -4:].std(0) + args.eps
                )
                next_states = f["next_states"][:]
                next_states[:, -4:] = (
                    next_states[:, -4:] - next_states[:, -4:].mean(0)
                ) / (next_states[:, -4:].std(0) + args.eps)

                if "states" in f:
                    del f["states"]
                    f.create_dataset("states", data=states, chunks=True)

                if "next_states" in f:
                    del f["next_states"]
                    f.create_dataset("next_states", data=next_states, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
