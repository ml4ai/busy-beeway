import argparse
import os
import sys
import gymnasium as gym
import h5py
import jax
import jax.numpy as jnp
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
    args = parser.parse_args(argv)
    trials = args.trials
    seed = args.seed
    output_file = args.output_file
    env = gym.make(
        "FrozenLake-v1",
        render_mode="rgb_array",
        max_episode_steps=100,
        desc=["SFHH", "FFHH", "FFFH", "FFFG"],
        map_name="Custom",
        is_slippery=True,
    )
    t_sts = []
    t_acts = []
    t_rtns = []
    t_ts = []
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    for i in range(trials):
        states = []
        actions = []
        returns = []
        state, info = env.reset()
        states.append(jax.nn.one_hot(state, env.observation_space.n))
        episode_over = False
        rtn = 0
        while not episode_over:
            action = (
                env.action_space.sample()
            )  # agent policy that uses the observation and info
            actions.append(jax.nn.one_hot(action, env.action_space.n))
            state, reward, terminated, truncated, info = env.step(action)
            rtn += reward
            returns.append(rtn)
            episode_over = terminated or truncated
            if not episode_over:
                states.append(jax.nn.one_hot(state, env.observation_space.n))
        env.close()
        actions = jnp.stack(actions)
        states = jnp.stack(states)
        returns = jnp.stack(returns)
        timesteps = jnp.arange(states.shape[0])
        t_sts.append(states)
        t_acts.append(actions)
        t_rtns.append(returns)
        t_ts.append(timesteps)

    max_size = max([i.shape[0] for i in t_sts])
    t_am = []
    for i in range(trials):
        attn_mask = jnp.ones(t_sts[i].shape[0])
        t_sts[i] = jnp.pad(t_sts[i], ((0, max_size - t_sts[i].shape[0]), (0, 0)))
        t_acts[i] = jnp.pad(t_acts[i], ((0, max_size - t_acts[i].shape[0]), (0, 0)))
        t_rtns[i] = jnp.pad(t_rtns[i], (0, max_size - t_rtns[i].shape[0]))
        t_ts[i] = jnp.pad(t_ts[i], (0, max_size - t_ts[i].shape[0]))
        t_am.append(jnp.pad(attn_mask, (0, max_size - attn_mask.shape[0])))
    t_sts = jnp.stack(t_sts)
    t_acts = jnp.stack(t_acts)
    t_rtns = jnp.stack(t_rtns)
    t_ts = jnp.stack(t_ts)
    t_am = jnp.stack(t_am)
    with h5py.File(output_file, "a") as g:
        g.create_dataset("states", data=t_sts, chunks=True)
        g.create_dataset("actions", data=t_acts, chunks=True)
        g.create_dataset("timesteps", data=t_ts, chunks=True)
        g.create_dataset("attn_mask", data=t_am, chunks=True)
        g.create_dataset("returns", data=t_rtns, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
