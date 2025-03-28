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

                rewards, _ = r_model(
                    sts,
                    acts,
                    ts,
                    am,
                    training=False,
                )
                seq_length = sts.shape[1]
                rewards = rewards.reshape(-1, seq_length)
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

                sts = sts.reshape(-1, sts.shape[2])
                if "states" in f:
                    del f["states"]
                f.create_dataset("states", data=sts, chunks=True)

                ts = ts.ravel()
                next_sts = []
                for i in tqdm(
                    range(1, ts.shape[0]), total=ts.shape[0], desc="Next States"
                ):
                    if ts[i] == 0:
                        next_sts.append(jnp.zeros(sts.shape[1]))
                    next_sts.append(sts[i])

                next_sts = jnp.stack(next_sts)

                if "next_states" in f:
                    del f["next_states"]
                f.create_dataset("next_states", data=next_sts, chunks=True)

                acts = acts.reshape(-1, acts.shape[2])
                if "actions" in f:
                    del f["actions"]
                f.create_dataset("actions", data=acts, chunks=True)

                am = am.ravel()
                if "attn_mask" in f:
                    del f["attn_mask"]
                f.create_dataset("attn_mask", data=am, chunks=True)

                rewards = rewards.ravel()
                if "rewards" in f:
                    del f["rewards"]
                f.create_dataset("rewards", data=rewards, chunks=True)

                if args.ep_returns:
                    R = 0.0
                    ep_rtns = []
                    for i in tqdm(
                        reversed(range(rewards.shape[0])),
                        total=rewards.shape[0],
                        desc="Episode Returns",
                    ):
                        if am[i] != 0:
                            R = R + rewards[i]
                        if ts[i] == 0:
                            ep_rtns.append(R)
                            R = 0.0
                    ep_rtns = jnp.array(ep_rtns)

                    if "ep_returns" in f:
                        del f["ep_returns"]
                    f.create_dataset("ep_returns", data=ep_rtns, chunks=True)

                    max_ep_length = jnp.max(ts)

                    if "max_ep_length" in f:
                        del f["max_ep_length"]
                    f.create_dataset("max_ep_length", data=max_ep_length, chunks=True)
    else:
        with h5py.File(f"{data}/{p_id}.hdf5", "a") as f:
                sts = f["states"][:]
                acts = f["actions"][:]
                ts = f["timesteps"][:]
                am = f["attn_mask"][:]

                rewards, _ = r_model(
                    sts,
                    acts,
                    ts,
                    am,
                    training=False,
                )
                seq_length = sts.shape[1]
                rewards = rewards.reshape(-1, seq_length)
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

                sts = sts.reshape(-1, sts.shape[2])
                if "states" in f:
                    del f["states"]
                f.create_dataset("states", data=sts, chunks=True)

                ts = ts.ravel()
                next_sts = []
                for i in tqdm(
                    range(1, ts.shape[0]), total=ts.shape[0], desc="Next States"
                ):
                    if ts[i] == 0:
                        next_sts.append(jnp.zeros(sts.shape[1]))
                    next_sts.append(sts[i])

                next_sts = jnp.stack(next_sts)

                if "next_states" in f:
                    del f["next_states"]
                f.create_dataset("next_states", data=next_sts, chunks=True)

                acts = acts.reshape(-1, acts.shape[2])
                if "actions" in f:
                    del f["actions"]
                f.create_dataset("actions", data=acts, chunks=True)

                am = am.ravel()
                if "attn_mask" in f:
                    del f["attn_mask"]
                f.create_dataset("attn_mask", data=am, chunks=True)

                rewards = rewards.ravel()
                if "rewards" in f:
                    del f["rewards"]
                f.create_dataset("rewards", data=rewards, chunks=True)

                if args.ep_returns:
                    R = 0.0
                    ep_rtns = []
                    for i in tqdm(
                        reversed(range(rewards.shape[0])),
                        total=rewards.shape[0],
                        desc="Episode Returns",
                    ):
                        if am[i] != 0:
                            R = R + rewards[i]
                        if ts[i] == 0:
                            ep_rtns.append(R)
                            R = 0.0
                    ep_rtns = jnp.array(ep_rtns)

                    if "ep_returns" in f:
                        del f["ep_returns"]
                    f.create_dataset("ep_returns", data=ep_rtns, chunks=True)

                    max_ep_length = jnp.max(ts)

                    if "max_ep_length" in f:
                        del f["max_ep_length"]
                    f.create_dataset("max_ep_length", data=max_ep_length, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
