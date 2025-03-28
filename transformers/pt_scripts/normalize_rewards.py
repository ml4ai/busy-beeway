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
        "reward",
        metavar="R",
        type=str,
        help="A .ckpt directory of PT model (must be absolute)",
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
        "-e",
        "--ep_returns",
        action="store_true",
        help="Compute Normalized Episode Returns.",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    p_id = os.path.expanduser(args.participants)
    if p_id.endswith(".txt"):
        P = load_list(p_id)
        max_ep_length = -jnp.inf
        max_ep_rtn = -jnp.inf
        min_ep_rtn = jnp.inf
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                max_ep_length = max(max_ep_length, jnp.max(f["max_ep_length"][:]))
                max_ep_rtn = max(max_ep_rtn, jnp.max(f["ep_returns"][:]))
                min_ep_rtn = min(min_ep_rtn, jnp.min(f["ep_returns"][:]))

        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                rewards = f["rewards"][:]
                n_rewards = max_ep_length*((rewards - max_ep_rtn)/(max_ep_rtn - min_ep_rtn))
                if "n_rewards" in f:
                    del f["n_rewards"]
                f.create_dataset("n_rewards", data=n_rewards, chunks=True)
                
                if args.ep_returns:
                    n_rewards = n_rewards.ravel()
                    am = f["attn_mask"][:]
                    ts = f["timesteps"][:]
                    R = 0.0
                    n_ep_rtns = []
                    for i in tqdm(
                        reversed(range(n_rewards.shape[0])),
                        total=n_rewards.shape[0],
                        desc="Normalized Episode Returns",
                    ):
                        if am[i] != 0:
                            R = R + n_rewards[i]
                        if ts[i] == 0:
                            n_ep_rtns.append(R)
                            R = 0.0
                    n_ep_rtns = jnp.array(n_ep_rtns)

                    if "n_ep_returns" in f:
                        del f["n_ep_returns"]
                    f.create_dataset("n_ep_returns", data=n_ep_rtns, chunks=True)
    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
            max_ep_length = jnp.max(f["timesteps"][:])
            max_ep_rtn = jnp.max(f["ep_returns"][:])
            min_ep_rtn = jnp.min(f["ep_returns"][:])
            
            rewards = f["rewards"][:]
            n_rewards = max_ep_length*((rewards - max_ep_rtn)/(max_ep_rtn - min_ep_rtn))
            if "n_rewards" in f:
                del f["n_rewards"]
            f.create_dataset("n_rewards", data=n_rewards, chunks=True)
            
            if args.ep_returns:
                n_rewards = n_rewards.ravel()
                am = f["attn_mask"][:]
                ts = f["timesteps"][:]

                R = 0.0
                n_ep_rtns = []
                for i in tqdm(
                    reversed(range(n_rewards.shape[0])),
                    total=n_rewards.shape[0],
                    desc="Normalized Episode Returns",
                ):
                    if r_am[i] != 0:
                        R = R + n_rewards[i]
                    if r_ts[i] == 0:
                        n_ep_rtns.append(R)
                        R = 0.0
                n_ep_rtns = jnp.array(n_ep_rtns)
    
                if "n_ep_returns" in f:
                    del f["n_ep_returns"]
                f.create_dataset("n_ep_returns", data=n_ep_rtns, chunks=True)

    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
