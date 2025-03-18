import argparse
import os
import sys
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.models.pref_transformer import load_PT
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
        help="Compute Raw Episode Returns (from unnormalized returns).",
    )
    parser.add_argument(
        "-r",
        "--returns",
        action="store_true",
        help="Compute Raw Trajectory returns (from unnormalized returns) and Raw Episode Returns (overwrites --ep_returns flag).",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    p_id = os.path.expanduser(args.participants)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    r_model = load_PT(reward, checkpointer, on_cpu=True)
    r_model = nnx.jit(r_model, static_argnums=4)
    checkpointer.close()
    if p_id.endswith(".txt"):
        P = load_list(p_id)
        for p_id in tqdm(P):
            with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
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
                del f["states"]
                f.create_dataset("states", data=sts, chunks=True)

                del f["actions"]
                f.create_dataset("actions", data=acts, chunks=True)

                del f["timesteps"]
                f.create_dataset("timesteps", data=ts, chunks=True)

                del f["attn_mask"]
                f.create_dataset("attn_mask", data=am, chunks=True)

                if args.ep_returns or args.returns:
                    rewards = rewards.ravel()
                    r_am = am.ravel()
                    r_ts = ts.ravel()
                    if args.returns:
                        returns = jnp.zeros_like(rewards, dtype=float)
                        R = 0.0
                        ep_rtns = []
                        for i in tqdm(
                            reversed(range(rewards.shape[0])),
                            total=rewards.shape[0],
                            desc="Raw Returns",
                        ):
                            if r_am[i] != 0:
                                R = R + rewards[i]
                                returns = returns.at[i].set(R)
                            if r_ts[i] == 0:
                                ep_rtns.append(R)
                                R = 0.0
                        returns = returns.reshape(am.shape[0], am.shape[1])
                        ep_rtns = jnp.array(ep_rtns)

                        if "raw_returns" in f:
                            del f["raw_returns"]
                        f.create_dataset("raw_returns", data=returns, chunks=True)

                    else:
                        R = 0.0
                        ep_rtns = []
                        for i in tqdm(
                            reversed(range(rewards.shape[0])),
                            total=rewards.shape[0],
                            desc="Raw Episode Returns",
                        ):
                            if r_am[i] != 0:
                                R = R + rewards[i]
                            if r_ts[i] == 0:
                                ep_rtns.append(R)
                                R = 0.0
                        ep_rtns = jnp.array(ep_rtns)

                    if "raw_ep_returns" in f:
                        del f["raw_ep_returns"]
                    f.create_dataset("raw_ep_returns", data=ep_rtns, chunks=True)

                if "rewards" in f:
                    del f["rewards"]
                f.create_dataset("rewards", data=rewards, chunks=True)
    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
            sts = f["states"][:]
            acts = f["actions"][:]
            ts = f["timesteps"][:]
            am = f["attn_mask"][:]

            max_episode_length = np.max(ts)
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
            del f["states"]
            f.create_dataset("states", data=sts, chunks=True)

            del f["actions"]
            f.create_dataset("actions", data=acts, chunks=True)

            del f["timesteps"]
            f.create_dataset("timesteps", data=ts, chunks=True)

            del f["attn_mask"]
            f.create_dataset("attn_mask", data=am, chunks=True)

            if args.ep_returns or args.returns:
                rewards = rewards.ravel()
                r_am = am.ravel()
                r_ts = ts.ravel()
                if args.returns:
                    returns = jnp.zeros_like(rewards, dtype=float)
                    R = 0.0
                    ep_rtns = []
                    for i in tqdm(
                        reversed(range(rewards.shape[0])),
                        total=rewards.shape[0],
                        desc="Raw Returns",
                    ):
                        if r_am[i] != 0:
                            R = R + rewards[i]
                            returns = returns.at[i].set(R)
                        if r_ts[i] == 0:
                            ep_rtns.append(R)
                            R = 0.0
                    returns = returns.reshape(am.shape[0], am.shape[1])
                    ep_rtns = jnp.array(ep_rtns)

                    if "raw_returns" in f:
                        del f["raw_returns"]
                    f.create_dataset("raw_returns", data=returns, chunks=True)

                else:
                    R = 0.0
                    ep_rtns = []
                    for i in tqdm(
                        reversed(range(rewards.shape[0])),
                        total=rewards.shape[0],
                        desc="Raw Episode Returns",
                    ):
                        if r_am[i] != 0:
                            R = R + rewards[i]
                        if r_ts[i] == 0:
                            ep_rtns.append(R)
                            R = 0.0
                    ep_rtns = jnp.array(ep_rtns)

                if "raw_ep_returns" in f:
                    del f["raw_ep_returns"]
                f.create_dataset("raw_ep_returns", data=ep_rtns, chunks=True)

            if "rewards" in f:
                del f["rewards"]
            f.create_dataset("rewards", data=rewards, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
