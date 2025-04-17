import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath("../.."))
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from argformat import StructuredFormatter
from jax import random

from transformers.training.utils import load_pickle
from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Computes Starc metric for two reward functions",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "rewards",
        metavar="R",
        type=str,
        help="File with list of reward functions to compare. \nThe names should omit any extensions",
    )
    parser.add_argument(
        "-n",
        "--n_samps",
        type=int,
        default=1000,
        help="Number of state-actions to sample",
    )
    parser.add_argument(
        "-m",
        "--m_samps",
        type=int,
        default=1000,
        help="Number of next states to sample",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        type=str,
        default="~/busy-beeway/transformers/t0012",
        help="Working directory for pickled models",
    )
    parser.add_argument(
        "-s",
        "--sample_file",
        type=str,
        default="~/busy-beeway/transformers/t0012/preference_data_1/t0012.hdf5",
        help="File with samples",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="~/busy-beeway/transformers/t0012/results/t0012_starc.hdf5",
        help="Output file with distance matrix",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=131387,
        help="Random Seed",
    )
    args = parser.parse_args(argv)
    rewards = os.path.expanduser(args.rewards)
    rewards = load_list(rewards)
    n = args.n_samps
    m = args.m_samps
    w_dir = os.path.expanduser(args.work_dir)
    samps = os.path.expanduser(args.sample_file)
    output = os.path.expanduser(args.output_file)
    seed = args.random_seed
    keys = random.key(seed)
    dists = np.zeros((len(rewards), len(rewards)))
    with h5py.File(samps) as f:
        s_shape = f["states"].shape
        a_shape = f["actions"].shape
        c_state_list = []
        action_list = []
        c_ts_list = []
        am_list = []
        n_samps_list = []
        n_ts_list = []
        n_am_list = []
        for k, j in random.split(keys, (n, 2)):
            o_idx = random.choice(k, s_shape[0])
            am = f["attn_mask"][o_idx]
            s_am = sum(am)
            while s_am == 0:
                k, _ = random.split(k)
                o_idx = random.choice(k, s_shape[0])
                am = f["attn_mask"][o_idx]
                s_am = sum(am)
            s_idx = random.choice(j, min(int(s_am), s_shape[1] - 1))
            am = am[s_idx].reshape(1, 1)
            c_ts = f["timesteps"][o_idx, s_idx].reshape(1, 1)
            c_state = f["states"][o_idx, s_idx, :].reshape(1, 1, s_shape[2])
            action = f["actions"][o_idx, s_idx, :].reshape(1, 1, a_shape[2])
            n_state = f["states"][o_idx, s_idx + 1, :]
            n_samps = []
            for h in random.split(j, m - 1):
                n_samps.append(
                    jnp.clip(
                        jnp.put(
                            n_state,
                            jnp.array([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            random.multivariate_normal(
                                h,
                                n_state[2:-4],
                                jnp.diag(jnp.ones(n_state[2:-4].shape[0])),
                            ),
                            inplace=False,
                        ),
                        jnp.array([0, -1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
                        jnp.array(
                            [
                                500,
                                1,
                                500,
                                3,
                                3,
                                3,
                                500,
                                3,
                                3,
                                500,
                                3,
                                500,
                                500,
                                500,
                                1,
                            ]
                        ),
                    )
                )
            n_samps.append(n_state)
            n_samps = jnp.stack(n_samps).reshape(m, 1, n_state.shape[0])
            n_am = jnp.tile(am, (m, 1))
            n_ts = jnp.tile(c_ts + 1, (m, 1))
            c_state_list.append(c_state)
            action_list.append(action)
            c_ts_list.append(c_ts)
            am_list.append(am)
            n_samps_list.append(n_samps)
            n_ts_list.append(n_ts)
            n_am_list.append(n_am)

    for r_1, r_2 in combinations(range(len(rewards)), 2):
        con_r_1 = []
        con_r_2 = []
        r_1_model = load_pickle(
            f"{w_dir}/results/no_pretrain/{rewards[r_1]}/best_model.pkl"
        )["model"]
        v_1_model = load_pickle(
            f"{w_dir}/results/v_results/{rewards[r_1]}/best_model.pkl"
        )["model"]
        r_2_model = load_pickle(
            f"{w_dir}/results/no_pretrain/{rewards[r_2]}/best_model.pkl"
        )["model"]
        v_2_model = load_pickle(
            f"{w_dir}/results/v_results/{rewards[r_2]}/best_model.pkl"
        )["model"]
        for i in range(n):
            r_1_pred, _ = r_1_model._train_state.apply_fn(
                r_1_model._train_state.params,
                c_state_list[i],
                action_list[i],
                c_ts_list[i],
                training=False,
                attn_mask=am_list[i],
            )
            c_v_1_pred = v_1_model._train_state.apply_fn(
                v_1_model._train_state.params,
                c_state_list[i],
                c_ts_list[i],
                training=False,
                attn_mask=am_list[i],
            )
            n_v_1_pred = v_1_model._train_state.apply_fn(
                v_1_model._train_state.params,
                n_samps_list[i],
                n_ts_list[i],
                training=False,
                attn_mask=n_am_list[i],
            )

            r_2_pred, _ = r_2_model._train_state.apply_fn(
                r_2_model._train_state.params,
                c_state_list[i],
                action_list[i],
                c_ts_list[i],
                training=False,
                attn_mask=am_list[i],
            )
            c_v_2_pred = v_2_model._train_state.apply_fn(
                v_2_model._train_state.params,
                c_state_list[i],
                c_ts_list[i],
                training=False,
                attn_mask=am_list[i],
            )
            n_v_2_pred = v_2_model._train_state.apply_fn(
                v_2_model._train_state.params,
                n_samps_list[i],
                n_ts_list[i],
                training=False,
                attn_mask=n_am_list[i],
            )
            con_r_1.append(
                jnp.nansum(
                    r_1_pred["value"].reshape(1)
                    - c_v_1_pred.reshape(1)
                    - n_v_1_pred.reshape(m, 1)
                )
                / m
            )
            con_r_2.append(
                jnp.nansum(
                    r_2_pred["value"].reshape(1)
                    - c_v_2_pred.reshape(1)
                    - n_v_2_pred.reshape(m, 1)
                )
                / m
            )
        con_r_1 = jnp.array(con_r_1)
        con_r_2 = jnp.array(con_r_2)
        con_r_1 = con_r_1 / jnp.sqrt(jnp.nanmean(con_r_1**2))
        con_r_2 = con_r_2 / jnp.sqrt(jnp.nanmean(con_r_2**2))
        dist = jnp.sqrt(jnp.nanmean((con_r_1 - con_r_2) ** 2))
        dists[r_1, r_2] = dist
        dists[r_2, r_1] = dist
    dists = jnp.asarray(dists)
    with h5py.File(output, "a") as f:
        f.create_dataset("starc_matrix", data=dists, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
