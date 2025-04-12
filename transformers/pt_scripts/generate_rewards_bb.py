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
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers/t0012/reward_data_1",
        help="Output directory",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    data = os.path.expanduser(args.data)
    output_dir = os.path.expanduser(args.output_dir)
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
                rewards = rewards["value"].reshape(-1, seq_length)
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

                with h5py.File(f"{output_dir}/{p_id}.hdf5", "a") as g:
                    sts = sts.reshape(-1, sts.shape[2])
                    if "states" in g:
                        del g["states"]
                    g.create_dataset("states", data=sts, chunks=True)

                    ts = ts.ravel()
                    next_sts = []
                    for i in tqdm(
                        range(1, ts.shape[0]), total=ts.shape[0], desc="Next States"
                    ):
                        if ts[i] == 0:
                            next_sts.append(jnp.zeros(sts.shape[1]))
                        else:
                            next_sts.append(sts[i])
                    next_sts.append(jnp.zeros(sts.shape[1]))
                    
                    next_sts = jnp.stack(next_sts)

                    if "next_states" in g:
                        del g["next_states"]
                    g.create_dataset("next_states", data=next_sts, chunks=True)

                    acts = acts.reshape(-1, acts.shape[2])
                    if "actions" in g:
                        del g["actions"]
                    g.create_dataset("actions", data=acts, chunks=True)

                    am = am.ravel()
                    if "attn_mask" in g:
                        del g["attn_mask"]
                    g.create_dataset("attn_mask", data=am, chunks=True)

                    rewards = rewards.ravel()
                    if "rewards" in g:
                        del g["rewards"]
                    g.create_dataset("rewards", data=rewards, chunks=True)

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

                        if "ep_returns" in g:
                            del g["ep_returns"]
                        g.create_dataset("ep_returns", data=ep_rtns, chunks=True)

                        max_ep_length = jnp.max(ts)

                        if "max_ep_length" in g:
                            del g["max_ep_length"]
                        g.create_dataset("max_ep_length", data=max_ep_length)
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
            rewards = rewards["value"].reshape(-1, seq_length)
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
            with h5py.File(f"{output_dir}/{p_id}.hdf5", "a") as g:
                sts = sts.reshape(-1, sts.shape[2])
                if "states" in g:
                    del g["states"]
                g.create_dataset("states", data=sts, chunks=True)

                ts = ts.ravel()
                next_sts = []
                for i in tqdm(
                    range(1, ts.shape[0]), total=ts.shape[0], desc="Next States"
                ):
                    if ts[i] == 0:
                        next_sts.append(jnp.zeros(sts.shape[1]))
                    else:
                        next_sts.append(sts[i])
                next_sts.append(jnp.zeros(sts.shape[1]))

                next_sts = jnp.stack(next_sts)

                if "next_states" in g:
                    del g["next_states"]
                g.create_dataset("next_states", data=next_sts, chunks=True)

                acts = acts.reshape(-1, acts.shape[2])
                if "actions" in g:
                    del g["actions"]
                g.create_dataset("actions", data=acts, chunks=True)

                am = am.ravel()
                if "attn_mask" in g:
                    del g["attn_mask"]
                g.create_dataset("attn_mask", data=am, chunks=True)

                rewards = rewards.ravel()
                if "rewards" in g:
                    del g["rewards"]
                g.create_dataset("rewards", data=rewards, chunks=True)

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

                    if "ep_returns" in g:
                        del g["ep_returns"]
                    g.create_dataset("ep_returns", data=ep_rtns, chunks=True)

                    max_ep_length = jnp.max(ts)

                    if "max_ep_length" in g:
                        del g["max_ep_length"]
                    g.create_dataset("max_ep_length", data=max_ep_length)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
