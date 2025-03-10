import argparse
import os
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from functools import partial
from tqdm import tqdm

import h5py

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter
from transformers.models.pref_transformer import load_PT


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates Monte Carlo Targets for Value Function Approximations \nusing precollected trajectory samples. \nThis physically resaves data to a new data file.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "reward",
        metavar="R",
        type=str,
        help="A .ckpt directory of PT model (must be absolute)",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="File with sample Trajectories",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="~/busy-beeway/transformers/data.hdf5",
        help="Output directory",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    output = os.path.expanduser(args.output)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    r_model = load_PT(reward, checkpointer, on_cpu=True)
    r_model = nnx.jit(r_model, static_argnums=4)
    checkpointer.close()
    with h5py.File(data, "r") as f:
        sts = f["states"][:]
        acts = f["actions"][:]
        ts = f["timesteps"][:]
        am = f["attn_mask"][:]
        preds, _ = r_model(
            sts,
            acts,
            ts,
            am,
            training=False,
        )
        seq_length = sts.shape[1]
        rewards = []
        for i in range(seq_length):
            rewards.append(preds["value"][:, 0, i])
        rewards = jnp.concatenate(rewards, axis=1)
        del preds
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
        rewards = rewards.ravel()
        r_am = am.ravel()
        r_ts = ts.ravel()
        returns = jnp.zeros_like(rewards, dtype=float)
        R = 0.0
        ep_rtns = []
        for i in tqdm(
            reversed(range(rewards.shape[0])), total=rewards.shape[0], desc="Returns"
        ):
            if r_am[i] != 0:
                R = R + rewards[i]
                returns = returns.at[i].set(R)
            if r_ts[i] == 0:
                ep_rtns.append(R)
                R = 0.0
        returns = returns.reshape(am.shape[0], am.shape[1])
        ep_rtns = jnp.array(ep_rtns)

        sts_2 = f["states_2"][:]
        acts_2 = f["actions_2"][:]
        ts_2 = f["timesteps_2"][:]
        am_2 = f["attn_mask_2"][:]
        preds_2, _ = r_model(
            sts_2,
            acts_2,
            ts_2,
            am_2,
            training=False,
        )
        seq_length_2 = sts_2.shape[1]
        rewards_2 = []
        for i in range(seq_length_2):
            rewards_2.append(preds_2["value"][:, 0, i])
        rewards_2 = jnp.concatenate(rewards_2, axis=1)
        del preds_2
        if jnp.any(jnp.isnan(rewards_2)):
            sts_2 = jnp.delete(
                sts_2, jnp.unique(jnp.argwhere(jnp.isnan(rewards_2))[:, 0]), axis=0
            )
            acts_2 = jnp.delete(
                acts_2, jnp.unique(jnp.argwhere(jnp.isnan(rewards_2))[:, 0]), axis=0
            )
            ts_2 = jnp.delete(
                ts_2, jnp.unique(jnp.argwhere(jnp.isnan(rewards_2))[:, 0]), axis=0
            )
            am_2 = jnp.delete(
                am_2, jnp.unique(jnp.argwhere(jnp.isnan(rewards_2))[:, 0]), axis=0
            )
            rewards_2 = jnp.delete(
                rewards_2,
                jnp.unique(jnp.argwhere(jnp.isnan(rewards_2))[:, 0]),
                axis=0,
            )
        rewards_2 = rewards_2.ravel()
        r_am_2 = am_2.ravel()
        r_ts_2 = ts_2.ravel()
        returns_2 = jnp.zeros_like(rewards_2, dtype=float)
        R_2 = 0.0
        ep_rtns_2 = []
        for i in tqdm(
            reversed(range(rewards_2.shape[0])),
            total=rewards_2.shape[0],
            desc="Returns_2",
        ):
            if r_am_2[i] != 0:
                R_2 = R_2 + rewards_2[i]
                returns_2 = returns_2.at[i].set(R_2)
            if r_ts_2[i] == 0:
                ep_rtns_2.append(R_2)
                R_2 = 0.0
        returns_2 = returns_2.reshape(am_2.shape[0], am_2.shape[1])

        ep_rtns_2 = jnp.array(ep_rtns_2)

        c_sts = jnp.concatenate([sts, sts_2])
        c_acts = jnp.concatenate([acts, acts_2])
        c_ts = jnp.concatenate([ts, ts_2])
        c_am = jnp.concatenate([am, am_2])
        c_returns = jnp.concatenate([returns, returns_2])

        with h5py.File(output, "a") as g:
            g.create_dataset("states", data=c_sts, chunks=True)
            g.create_dataset("actions", data=c_acts, chunks=True)
            g.create_dataset("timesteps", data=c_ts, chunks=True)
            g.create_dataset("attn_mask", data=c_am, chunks=True)
            g.create_dataset("returns", data=c_returns, chunks=True)
            g.create_dataset("ep_returns", data=ep_rtns, chunks=True)
            g.create_dataset("ep_returns_2", data=ep_rtns_2, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
