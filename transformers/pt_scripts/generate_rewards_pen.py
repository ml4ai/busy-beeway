import argparse
import os
import sys
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import minari
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import load_list
from transformers.models.pref_transformer import load_PT


def main(argv):
    parser = argparse.ArgumentParser(
        description="Adds a set of rewards to dataset given a reward function (overwrites if needed). \nAlso removes observations that cause NAN rewards",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="D4RL/pen/human-v2",
        help="Environment Name",
    )
    parser.add_argument(
        "-s",
        "--save_file",
        type=str,
        default="~/busy-beeway/transformers/pen_labels/AdroitHandPen-v1_human_task_rewards.hdf5",
        help="Path for saving output.",
    )
    parser.add_argument(
        "-f",
        "--reward_function",
        type=str,
        default=None,
        help="Generate Rewards from given reward function. \nCan default to none if just wanting to get precomputed task \nrewards",
    )
    parser.add_argument(
        "-q",
        "--query_len",
        type=int,
        default=100,
        help="Query Length",
    )
    parser.add_argument(
        "-p",
        "--ep_returns",
        action="store_true",
        help="Compute Raw Episode Returns (from unnormalized returns). \nDoes nothing if no external reward function is given.",
    )
    parser.add_argument(
        "-t",
        "--task_ep_returns",
        action="store_true",
        help="Compute Task Episode Returns",
    )
    parser.add_argument(
        "-n",
        "--normalize_states",
        action="store_true",
        help="Normalize states",
    )
    parser.add_argument(
        "-i",
        "--eps",
        type=float,
        default=1e-3,
        help="Ensures no zero division for normalizing states",
    )
    args = parser.parse_args(argv)
    save_file = args.save_file
    dataset = minari.load_dataset(args.env_name)
    if args.reward_function:
        reward_function = os.path.expanduser(args.reward_function)
        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        r_model = load_PT(reward_function, checkpointer, on_cpu=True)
        r_model = nnx.jit(r_model, static_argnums=4)
        checkpointer.close()
        episodes = dataset.iterate_episodes()
        states = []
        next_states = []
        actions = []
        timesteps = []
        attn_mask = []
        rwd = []
        t_rwd = []
        with h5py.File(save_file, "a") as f:
            for ep in tqdm(episodes):
                if len(ep) % args.query_len == 0:
                    fill_size = len(ep)
                else:
                    fill_size = len(ep) + (args.query_len - (len(ep) % args.query_len))
                n_splits = int(fill_size / args.query_len)
                if fill_size > len(ep):
                    sts = ep.observations[:-1, ...]
                    sts = np.pad(
                        sts,
                        ((0, fill_size - len(ep)), (0, 0)),
                        constant_values=0,
                    )

                    next_sts = ep.observations[1:, ...]

                    next_sts = np.pad(
                        next_sts,
                        ((0, fill_size - len(ep)), (0, 0)),
                        constant_values=0,
                    )

                    acts = np.pad(
                        ep.actions,
                        ((0, fill_size - len(ep)), (0, 0)),
                        constant_values=0,
                    )

                    ts = np.arange(fill_size)
                    am = np.zeros(fill_size)
                    am[: len(ep)] = 1

                    t_r = np.pad(
                        ep.rewards,
                        (0, fill_size - len(ep)),
                        constant_values=0,
                    )
                else:
                    sts = ep.observations[:fill_size]

                    next_sts = ep.observations[1 : fill_size + 1, ...]

                    acts = ep.actions[:fill_size, ...]

                    ts = np.arange(fill_size)

                    am = np.zeros(fill_size)
                    am[: len(ep)] = 1

                    t_r = ep.rewards[:fill_size]

                r_sts = sts.reshape((n_splits, args.query_len, sts.shape[1]))
                r_acts = acts.reshape((n_splits, args.query_len, acts.shape[1]))
                r_ts = ts.reshape((n_splits, args.query_len))
                r_am = am.reshape((n_splits, args.query_len))

                rewards, _ = r_model(
                    r_sts,
                    r_acts,
                    r_ts,
                    r_am,
                    training=False,
                )
                seq_length = sts.shape[1]
                rewards = rewards["value"].ravel()

                states.append(sts)
                next_states.append(next_sts)
                actions.append(acts)
                timesteps.append(ts)
                attn_mask.append(am)
                t_rwd.append(t_r)
                rwd.append(rewards)

            states = jnp.concatenate(states)
            next_states = jnp.concatenate(next_states)
            actions = jnp.concatenate(actions)
            timesteps = jnp.concatenate(timesteps)
            attn_mask = jnp.concatenate(attn_mask)
            rwd = jnp.concatenate(rwd)
            t_rwd = jnp.concatenate(t_rwd)

            if args.normalize_states:
                states = (states - states.mean(0)) / (states.std(0) + args.eps)
                next_states = (next_states - next_states.mean(0)) / (
                    next_states.std(0) + args.eps
                )

            if "states" in f:
                del f["states"]
            f.create_dataset("states", data=states, chunks=True)

            if "next_states" in f:
                del f["next_states"]
            f.create_dataset("next_states", data=next_states, chunks=True)

            if "actions" in f:
                del f["actions"]
            f.create_dataset("actions", data=actions, chunks=True)

            if "attn_mask" in f:
                del f["attn_mask"]
            f.create_dataset("attn_mask", data=attn_mask, chunks=True)

            if "rewards" in f:
                del f["rewards"]
            f.create_dataset("rewards", data=rwd, chunks=True)

            if "task_rewards" in f:
                del f["task_rewards"]
            f.create_dataset("task_rewards", data=t_rwd, chunks=True)

            if args.ep_returns:
                R = 0.0
                ep_rtns = []
                for i in tqdm(
                    reversed(range(rwd.shape[0])),
                    total=rwd.shape[0],
                    desc="Episode Returns",
                ):
                    if attn_mask[i] != 0:
                        R = R + rwd[i]
                    if timesteps[i] == 0:
                        ep_rtns.append(R)
                        R = 0.0
                ep_rtns = jnp.array(ep_rtns)

            if "ep_returns" in f:
                del f["ep_returns"]
            f.create_dataset("ep_returns", data=ep_rtns, chunks=True)

            if args.task_ep_returns:
                t_R = 0.0
                t_ep_rtns = []
                for i in tqdm(
                    reversed(range(t_rwd.shape[0])),
                    total=t_rwd.shape[0],
                    desc="Task Episode Returns",
                ):
                    if attn_mask[i] != 0:
                        t_R = t_R + t_rwd[i]
                    if timesteps[i] == 0:
                        t_ep_rtns.append(t_R)
                        t_R = 0.0
                t_ep_rtns = jnp.array(t_ep_rtns)

            if "task_ep_returns" in f:
                del f["task_ep_returns"]
            f.create_dataset("task_ep_returns", data=t_ep_rtns, chunks=True)
    else:
        episodes = dataset.iterate_episodes()
        states = []
        next_states = []
        actions = []
        timesteps = []
        attn_mask = []
        t_rwd = []
        with h5py.File(save_file, "a") as f:
            for ep in tqdm(episodes):
                sts = ep.observations[:-1, ...]

                next_sts = ep.observations[1:, ...]

                acts = ep.actions

                ts = np.arange(len(ep))
                am = np.ones(len(ep))

                t_r = ep.rewards

                states.append(sts)
                next_states.append(next_sts)
                actions.append(acts)
                timesteps.append(ts)
                attn_mask.append(am)
                t_rwd.append(t_r)

            states = jnp.concatenate(states)
            next_states = jnp.concatenate(next_states)
            actions = jnp.concatenate(actions)
            timesteps = jnp.concatenate(timesteps)
            attn_mask = jnp.concatenate(attn_mask)
            t_rwd = jnp.concatenate(t_rwd)

            if args.normalize_states:
                states = (states - states.mean(0)) / (states.std(0) + args.eps)
                next_states = (next_states - next_states.mean(0)) / (
                    next_states.std(0) + args.eps
                )

            if "states" in f:
                del f["states"]
            f.create_dataset("states", data=states, chunks=True)

            if "next_states" in f:
                del f["next_states"]
            f.create_dataset("next_states", data=next_states, chunks=True)

            if "actions" in f:
                del f["actions"]
            f.create_dataset("actions", data=actions, chunks=True)

            if "attn_mask" in f:
                del f["attn_mask"]
            f.create_dataset("attn_mask", data=attn_mask, chunks=True)

            if "task_rewards" in f:
                del f["task_rewards"]
            f.create_dataset("task_rewards", data=t_rwd, chunks=True)

            if args.task_ep_returns:
                t_R = 0.0
                t_ep_rtns = []
                for i in tqdm(
                    reversed(range(t_rwd.shape[0])),
                    total=t_rwd.shape[0],
                    desc="Task Episode Returns",
                ):
                    if attn_mask[i] != 0:
                        t_R = t_R + t_rwd[i]
                    if timesteps[i] == 0:
                        t_ep_rtns.append(t_R)
                        t_R = 0.0
                t_ep_rtns = jnp.array(t_ep_rtns)

            if "task_ep_returns" in f:
                del f["task_ep_returns"]
            f.create_dataset("task_ep_returns", data=t_ep_rtns, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
