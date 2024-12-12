import argparse
import os
import sys

import h5py
import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_platforms", "cpu")

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.training.utils import load_pickle


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates Monte Carlo Targets for Value Function Approximations \nusing precollected trajectory samples. \nThis physically resaves data to a new data file.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "reward",
        metavar="R",
        type=str,
        help="File with Reward function (as pickled dictionary)",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="File with sample Trajectories",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers",
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--set_2",
        action="store_true",
        help="This uses the real data samples \nversus the random policy samples.",
    )
    parser.add_argument(
        "-t",
        "--data_tag",
        type=str,
        default=None,
        help="adds identifier to return_to_go dataset",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    output_dir = os.path.expanduser(args.output_dir)
    if args.data_tag is not None:
        data_tag = args.data_tag
    else:
        data_tag = "data"
    r_model = load_pickle(reward)["model"]
    with h5py.File(data, "r") as f:
        if args.set_2:
            sts = f["states_2"][:]
            acts = f["actions_2"][:]
            ts = f["timesteps_2"][:]
            am = f["attn_mask_2"][:]
        else:
            sts = f["states"][:]
            acts = f["actions"][:]
            ts = f["timesteps"][:]
            am = f["attn_mask"][:]
        seq_length = sts.shape[1]
        rewards = []
        for i in tqdm(range(seq_length)):
            preds, _ = r_model._train_state.apply_fn(
                r_model._train_state.params,
                sts[:, : (i + 1), :],
                acts[:, : (i + 1), :],
                ts[:, : (i + 1)],
                training=False,
                attn_mask=am[:, : (i + 1)],
            )
            rewards.append(preds["value"][:, 0, -1])
        rewards = jnp.concatenate(rewards, axis=1)
        if jnp.any(jnp.isnan(rewards)):
            sts = jnp.delete(
                sts, jnp.unique(jnp.argwhere(jnp.isnan(rewards))[:, 0]), axis=0
            )
            acts = jnp.delete(
                acts, jnp.unique(jnp.argwhere(jnp.isnan(rewards))[:, 0]), axis=0
            )
            ts = jnp.delete(
                ts, jnp.unique(jnp.argwhere(jnp.isnan(rewards))[:, 0]), axis=0
            )
            am = jnp.delete(
                am, jnp.unique(jnp.argwhere(jnp.isnan(rewards))[:, 0]), axis=0
            )
            rewards = jnp.delete(
                rewards,
                jnp.unique(jnp.argwhere(jnp.isnan(rewards))[:, 0]),
                axis=0,
            )
        rtns = []
        for c in range(seq_length):
            rtns.append(jnp.sum(rewards[:, c:]*am[:,c:], axis=1))
        rtns = jnp.stack(rtns).transpose()
        with h5py.File(f"{output_dir}/{data_tag}.hdf5", "a") as g:
            g.create_dataset("states", data=sts, chunks=True)
            g.create_dataset("actions", data=acts, chunks=True)
            g.create_dataset("timesteps", data=ts, chunks=True)
            g.create_dataset("attn_mask", data=am, chunks=True)
            g.create_dataset("returns", data=rtns, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
