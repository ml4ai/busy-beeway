import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath("../.."))
import jax
import jax.numpy as jnp
from argformat import StructuredFormatter
from jax import random

from transformers.training.utils import load_pickle


def main(argv):
    parser = argparse.ArgumentParser(
        description="Computes Starc metric for two reward functions",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "rewards",
        metavar="R",
        type=str,
        nargs=2,
        help="Reward functions from two test sessions",
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
        "-r",
        "--random_seed",
        type=int,
        default=131387,
        help="Random Seed",
    )
    args = parser.parse_args(argv)
    rewards = args.rewards
    n = args.n_samps
    m = args.m_samps
    w_dir = os.path.expanduser(args.work_dir)
    samps = os.path.expanduser(args.sample_file)
    seed = args.random_seed
    r_1_model = load_pickle(f"{w_dir}/results/no_pretrain/{rewards[0]}/best_model.pkl")[
        "model"
    ]
    v_1_model = load_pickle(f"{w_dir}/results/v_results/{rewards[0]}/best_model.pkl")[
        "model"
    ]
    r_2_model = load_pickle(f"{w_dir}/results/no_pretrain/{rewards[1]}/best_model.pkl")[
        "model"
    ]
    v_2_model = load_pickle(f"{w_dir}/results/v_results/{rewards[1]}/best_model.pkl")[
        "model"
    ]
    keys = random.key(seed)
    with h5py.File(samps) as f:
        s_shape = f["states"].shape
        a_shape = f["actions"].shape
        con_r_1 = []
        con_r_2 = []
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
                            [500, 1, 500, 3, 3, 3, 500, 3, 3, 500, 3, 500, 500, 500, 1]
                        ),
                    )
                )
            n_samps.append(n_state)
            n_samps = jnp.stack(n_samps).reshape(m, 1, n_state.shape[0])
            n_am = jnp.tile(am, (m, 1))
            n_ts = jnp.tile(c_ts + 1, (m, 1))
            r_1_pred, _ = r_1_model._train_state.apply_fn(
                r_1_model._train_state.params,
                c_state,
                action,
                c_ts,
                training=False,
                attn_mask=am,
            )
            c_v_1_pred = v_1_model._train_state.apply_fn(
                v_1_model._train_state.params,
                c_state,
                c_ts,
                training=False,
                attn_mask=am,
            )
            n_v_1_pred = v_1_model._train_state.apply_fn(
                v_1_model._train_state.params,
                n_samps,
                n_ts,
                training=False,
                attn_mask=n_am,
            )

            r_2_pred, _ = r_2_model._train_state.apply_fn(
                r_2_model._train_state.params,
                c_state,
                action,
                c_ts,
                training=False,
                attn_mask=am,
            )
            c_v_2_pred = v_2_model._train_state.apply_fn(
                v_2_model._train_state.params,
                c_state,
                c_ts,
                training=False,
                attn_mask=am,
            )
            n_v_2_pred = v_2_model._train_state.apply_fn(
                v_2_model._train_state.params,
                n_samps,
                n_ts,
                training=False,
                attn_mask=n_am,
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
        print(jnp.sqrt(jnp.nanmean((con_r_1 - con_r_2) ** 2)))
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
