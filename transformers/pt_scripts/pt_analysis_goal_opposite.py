import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import seaborn as sns
from flax import nnx
from matplotlib import colors
from matplotlib.collections import LineCollection
from tqdm import tqdm, trange
from argformat import StructuredFormatter

import h5py

sys.path.insert(0, os.path.abspath("../.."))
from transformers.models.pref_transformer import load_PT
from transformers.data_utils.bb_data_loading import load_list
from transformers.evaluation.eval_episodes import bb_record_opposite_goal


def rclr(d):
    log_d = np.log(d)
    log_d = np.where(np.isfinite(log_d), log_d, 0.0)
    return log_d - log_d.mean()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Runs PT Analysis with Goal Opposite Trajectories",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_list",
        metavar="P",
        type=str,
        help="Text file with list of participants",
    )
    parser.add_argument(
        "-r",
        "--reward_models",
        type=str,
        default="~/busy-beeway/transformers/pt_rewards_bb",
        help="Directory with reward models (must be absolute path)",
    )
    parser.add_argument(
        "-t",
        "--t_samples",
        type=int,
        default=100,
        help="Number of trajectory segment samples",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=4,
        help="Random seed used throughout the program.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=1000,
        help="Number of action samples for EPIC",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers/dist_m_goal_opposite_results",
        help="Output directory for results",
    )
    args = parser.parse_args(argv)
    reward_models = os.path.expanduser(args.reward_models)
    p_ids = load_list(os.path.expanduser(args.p_list))
    output_dir = os.path.expanduser(args.output_dir)
    t_samples = args.t_samples
    n_samps = args.n_samples
    rng = np.random.default_rng(args.seed)
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    r_models = [
        nnx.jit(
            load_PT(
                os.path.expanduser(f"{reward_models}/{p}/best_model.ckpt"),
                checkpointer,
                on_cpu=True,
            ),
            static_argnums=4,
        )
        for p in p_ids
    ]
    checkpointer.close()

    speeds = rng.uniform(0.0, 0.44, n_samps)
    angles = rng.uniform(-180.0, 180.0, n_samps)
    sample_acts = np.stack([speeds, angles]).T
    discount = 0.99
    rad = np.zeros((len(p_ids), len(p_ids)))
    pear_dist = np.zeros((len(p_ids), len(p_ids)))
    w_pear_dist = np.zeros((len(p_ids), len(p_ids)))
    epic = np.zeros((len(p_ids), len(p_ids)))
    for i in range(t_samples):
        e_sts, e_acts, _ = bb_record_opposite_goal(
            0.44, obs_dist=200, seed=args.seed + (i + 1)
        )
        if e_sts.shape[1] % 100 == 0:
            fill_size = e_sts.shape[1]
        else:
            fill_size = e_sts.shape[1] + (100 - (e_sts.shape[1] % 100))
        n_splits = int(fill_size / 100)

        if fill_size > e_sts.shape[1]:
            sts = np.pad(
                e_sts,
                ((0, 0), (0, fill_size - e_sts.shape[1]), (0, 0)),
                constant_values=0,
            )
            acts = np.pad(
                e_acts,
                ((0, 0), (0, fill_size - e_acts.shape[1]), (0, 0)),
                constant_values=0,
            )
        else:
            sts = e_sts[:fill_size, ...]
            acts = e_acts[:fill_size, ...]
        ts = np.arange(fill_size)
        am = np.zeros(fill_size)
        am[: e_sts.shape[1]] = 1

        sts = sts.reshape((n_splits, 100, sts.shape[2]))
        acts = acts.reshape((n_splits, 100, acts.shape[2]))
        ts = ts.reshape((n_splits, 100))
        am = am.reshape((n_splits, 100))
        if n_splits > 1:
            r_idx = rng.choice(n_splits)
            sts = sts[r_idx, ...].reshape(1, 100, 26)
            acts = acts[r_idx, ...].reshape(1, 100, 2)
            ts = ts[r_idx, ...].reshape(1, 100)
            am = am[r_idx, ...].reshape(1, 100)
        am_sum = int(am.sum())
        n_rewards = []
        n_weights = []
        n_clr_weights = []
        n_c_rewards = []

        for r_model in tqdm(
            r_models, total=len(r_models), desc=f"Processing Trajectory Segment {i}"
        ):
            rewards, weights = r_model(sts, acts, ts, am, training=False)
            rewards = rewards["value"].reshape(
                100,
            )
            rewards = rewards[:am_sum]
            n_rewards.append(rewards)

            weights = weights[-1].reshape(1, 100, 100)
            weights = np.mean(weights, axis=1).reshape(
                100,
            )
            weights = weights[:am_sum]
            n_weights.append(weights)

            rclr_w = rclr(weights)
            n_clr_weights.append(rclr_w)

            r_mean = np.zeros(am_sum)
            for j in range(am_sum):
                sts_samps = np.repeat(
                    sts[:, : (j + 1), :].reshape(1, -1, sts.shape[2]),
                    n_samps,
                    axis=0,
                )
                acts_samps = np.repeat(
                    acts[:, : (j + 1), :].reshape(1, -1, acts.shape[2]),
                    n_samps,
                    axis=0,
                )
                acts_samps[:, -1, :] = sample_acts
                ts_samps = np.repeat(ts[:, : (j + 1)].reshape(1, -1), n_samps, axis=0)
                am_samps = np.repeat(am[:, : (j + 1)].reshape(1, -1), n_samps, axis=0)
                rwd_samps, _ = r_model(
                    sts_samps, acts_samps, ts_samps, am_samps, training=False
                )
                rwd_samps = rwd_samps["value"].reshape(n_samps, -1)[:, -1]
                r_mean[j] = rwd_samps.mean()

            c_rewards = np.zeros(am_sum - 1)
            for k in range(am_sum - 1):
                c_rewards[k] = rewards[k] + discount * r_mean[k + 1] - r_mean[k]
            n_c_rewards.append(c_rewards)

        for p_1 in range(len(p_ids)):
            for p_2 in range(len(p_ids)):
                rad[p_1, p_2] += np.sqrt(
                    ((n_clr_weights[p_1] - n_clr_weights[p_2]) ** 2).sum()
                )

                data = {
                    f"{p_ids[p_1]}": n_rewards[p_1],
                    f"{p_ids[p_2]}": n_rewards[p_2],
                }
                df = pd.DataFrame(data)
                corr = df[f"{p_ids[p_1]}"].corr(df[f"{p_ids[p_2]}"])
                pear_dist[p_1, p_2] += np.sqrt(1 - corr) / np.sqrt(2)

                w_data = {
                    f"{p_ids[p_1]}": n_weights[p_1] * n_rewards[p_1],
                    f"{p_ids[p_2]}": n_weights[p_2] * n_rewards[p_2],
                }
                w_df = pd.DataFrame(w_data)
                w_corr = w_df[f"{p_ids[p_1]}"].corr(w_df[f"{p_ids[p_2]}"])
                w_pear_dist[p_1, p_2] += np.sqrt(1 - w_corr) / np.sqrt(2)

                e_data = {
                    f"{p_ids[p_1]}": n_c_rewards[p_1],
                    f"{p_ids[p_2]}": n_c_rewards[p_2],
                }
                e_df = pd.DataFrame(e_data)
                e_corr = e_df[f"{p_ids[p_1]}"].corr(e_df[f"{p_ids[p_2]}"])
                epic[p_1, p_2] += np.sqrt(1 - e_corr) / np.sqrt(2)
    rad = rad / (t_samples * 1.0)
    pear_dist = pear_dist / (t_samples * 1.0)
    w_pear_dist = w_pear_dist / (t_samples * 1.0)
    epic = epic / (t_samples * 1.0)

    with h5py.File(f"{output_dir}/mean_dist_matrices_goal_opposite.hdf5", "a") as g:
        g["robust_aitchison_distance"] = rad
        g["pearson_distance"] = pear_dist
        g["weighted_pearson_distance"] = w_pear_dist
        g["EPIC"] = epic

    fig, axe = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=(15, 10),
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    sns.heatmap(
        rad, cmap="viridis_r", ax=axe["A"], xticklabels=p_ids, yticklabels=p_ids
    )
    sns.heatmap(
        pear_dist,
        cmap="viridis_r",
        ax=axe["B"],
        xticklabels=p_ids,
        yticklabels=p_ids,
    )
    sns.heatmap(
        w_pear_dist,
        cmap="viridis_r",
        ax=axe["C"],
        xticklabels=p_ids,
        yticklabels=p_ids,
    )
    sns.heatmap(
        epic, cmap="viridis_r", ax=axe["D"], xticklabels=p_ids, yticklabels=p_ids
    )
    axe["A"].set_title("Robust Aitchison Distance (Weights)")
    axe["B"].set_title("Pearson Distance (Rewards)")
    axe["C"].set_title("Weighted Pearson Distance")
    axe["D"].set_title("EPIC (Rewards)")
    plt.savefig(f"{output_dir}/mean_dist_matrices_goal_opposite.png")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
