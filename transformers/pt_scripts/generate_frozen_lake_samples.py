import argparse
import os
import sys
import gymnasium as gym
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

jax.config.update("jax_platforms", "cpu")

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates random policy samples from Frozen Lake.",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "trials",
        metavar="T",
        type=int,
        help="Number of samples",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=25102,
        help="Seed (25102 by default)",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="~/busy-beeway/transformers/random_policy_frozen_lake.hdf5",
        help="Output file",
    )
    parser.add_argument(
        "-c",
        "--context_length",
        type=int,
        default=5,
        help="Episodes are divided into segments of the set context length",
    )
    parser.add_argument(
        "-m",
        "--max_episode_steps",
        type=int,
        default=30,
        help="Episodes are divided into segments of the set context length",
    )
    args = parser.parse_args(argv)
    trials = args.trials
    seed = args.seed
    output_file = os.path.expanduser(args.output_file)
    max_episode_steps = args.max_episode_steps
    context = args.context_length
    env = gym.make(
        "FrozenLake-v1",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        desc=["SFHH", "FFHH", "FFFH", "FFFG"],
        map_name="Custom",
        is_slippery=True,
    )
    t_sts = []
    t_acts = []
    t_rwds = []
    t_ts = []
    e_labels = []
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    for i in range(trials):
        states = []
        actions = []
        rewards = []
        state, info = env.reset()
        states.append(jax.nn.one_hot(state, env.observation_space.n))
        episode_over = False
        while not episode_over:
            action = (
                env.action_space.sample()
            )  # agent policy that uses the observation and info
            actions.append(jax.nn.one_hot(action, env.action_space.n))
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            episode_over = terminated or truncated
            if not episode_over:
                states.append(jax.nn.one_hot(state, env.observation_space.n))
        env.close()
        actions = jnp.stack(actions)
        states = jnp.stack(states)
        rewards = jnp.stack(rewards)
        timesteps = jnp.arange(states.shape[0])
        t_sts.append(states)
        t_acts.append(actions)
        t_rwds.append(rewards)
        t_ts.append(timesteps)
        e_labels.append(int(rewards[-1]))
    am = []
    sts = []
    acts = []
    rwds = []
    ts = []
    labels = []
    for i in range(trials):
        fill_size = t_sts[i].shape[0] + (context - (t_sts[i].shape[0] % context))
        n_splits = int(fill_size / context)
        pad_size = fill_size - t_sts[i].shape[0]
        attn_mask = jnp.ones(t_sts[i].shape[0])
        sts.append(
            jnp.pad(t_sts[i], ((0, pad_size), (0, 0))).reshape(
                n_splits, context, t_sts[i].shape[1]
            )
        )
        acts.append(
            jnp.pad(t_acts[i], ((0, pad_size), (0, 0))).reshape(
                n_splits, context, t_acts[i].shape[1]
            )
        )

        rwds.append(jnp.pad(t_rwds[i], (0, pad_size)).reshape(n_splits, context))
        ts.append(jnp.pad(t_ts[i], (0, pad_size)).reshape(n_splits, context))
        am.append(jnp.pad(attn_mask, (0, pad_size)).reshape(n_splits, context))
        l = np.zeros(n_splits)
        l[-1] = e_labels[i]
        labels.append(l)
    sts = jnp.concatenate(sts)
    acts = jnp.concatenate(acts)
    rwds = jnp.concatenate(rwds)
    ts = jnp.concatenate(ts)
    am = jnp.concatenate(am)
    labels = jnp.concatenate(labels)
    rtns = []
    for c in range(context):
        rtns.append(jnp.sum(rwds[:, c:], axis=1))
    rtns = jnp.stack(rtns).transpose()
    with h5py.File(output_file, "a") as g:
        g.create_dataset("states", data=sts, chunks=True)
        g.create_dataset("actions", data=acts, chunks=True)
        g.create_dataset("timesteps", data=ts, chunks=True)
        g.create_dataset("attn_mask", data=am, chunks=True)
        g.create_dataset("returns", data=rtns, chunks=True)
        g.create_dataset("labels", data=labels, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
