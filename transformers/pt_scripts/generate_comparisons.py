import argparse
import os
import sys

import h5py
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_platforms", "cpu")
jax.clear_caches()

sys.path.insert(0, os.path.abspath("../.."))

from argformat import StructuredFormatter

from transformers.evaluation.eval_episodes import bb_record_episode
from transformers.replayer.replayer import animate_run, load_stats
from transformers.training.utils import load_pickle
from transformers.data_utils.bb_data_loading import load_participant_data_p


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generates comparisons for real versus generated behavior",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "dt",
        metavar="D",
        type=str,
        help="File with Decision Transformer model",
    )
    parser.add_argument(
        "pt",
        metavar="P",
        type=str,
        help="File with Preference Transformer model",
    )
    parser.add_argument(
        "stats",
        metavar="S",
        type=str,
        help="File with Obstacle Stats",
    )
    parser.add_argument(
        "real_returns",
        metavar="R",
        type=str,
        help="File with returns for real data",
    )
    parser.add_argument(
        "auto_id",
        metavar="A",
        type=str,
        help="Participant Auto-ID",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=25102,
        help="Random seed",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=1000,
        help="Number of Episodes",
    )
    parser.add_argument(
        "-t",
        "--target_return",
        type=float,
        default=8000.0,
        help="Target return",
    )
    parser.add_argument(
        "-m",
        "--horizon",
        type=int,
        default=400,
        help="Horizon",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers",
        help="Horizon",
    )
    args = parser.parse_args(argv)
    dt = os.path.expanduser(args.dt)
    pt = os.path.expanduser(args.pt)
    stats = os.path.expanduser(args.stats)
    real_returns = os.path.expanduser(args.real_returns)
    auto_id = args.auto_id
    seed = args.seed
    episodes = args.episodes
    target_return = args.target_return
    horizon = args.horizon
    output_dir = os.path.expanduser(args.output_dir)

    d_model = load_pickle(dt)["model"]
    r_model = load_pickle(pt)["model"]
    move_stats = load_stats(stats)
    key = jax.random.key(seed)
    D_g = []
    for i in tqdm(jax.random.split(key, episodes)):
        jax.clear_caches()
        D_g.append(
            bb_record_episode(
                d_model, r_model, move_stats, i, 100, target_return, horizon
            )
        )

    with h5py.File(real_returns) as f:
        rtns_r = f["ep_returns_2"][:]

    # Assumes data is in certain place
    D_r = load_participant_data_p(auto_id, cores)
    successes_r = 0.0
    end_goal_dist_r = []
    frames_r = []
    for d in D_r:
        if d["reached_goal"]:
            successes_r += 1.0
        posX = d["player"]["posX"].to_numpy()[-1]
        posY = d["player"]["posY"].to_numpy()[-1]
        goal = d["goal"]
        end_goal_dist_r.append(
            np.sqrt(((goal[0] - posX) ** 2) + ((goal[1] - posY) ** 2))
        )
        frames_r.append(len(d["player"]))

    successes_g = 0.0
    end_goal_dist_g = []
    frames_g = []
    rtns_g = []
    for e_r, e_l, d in D_g:
        if d["reached_goal"]:
            successes_g += 1.0
        posX = d["player"]["posX"].to_numpy()[-1]
        posY = d["player"]["posY"].to_numpy()[-1]
        goal = d["goal"]
        end_goal_dist_g.append(
            np.sqrt(((goal[0] - posX) ** 2) + ((goal[1] - posY) ** 2))
        )
        frames_g.append(e_l + 1)
        rtns_g.append(e_r)

    fig, ax = plt.subplots()
    vp = ax.violinplot([rtns_r, rtns_g], showmeans=True, showmedians=True)
    vp["cmeans"].set_color("orange")
    vp["cmeans"].set_label("Mean")
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_label("Median")
    ax.legend()
    ax.set_title("Returns (Real Versus Generated)")
    ax.set_xticks([y + 1 for y in range(2)], labels=["Real", "Generated"])
    ax.set_ylabel("Return")
    ax.scatter(
        np.concatenate([np.ones(rtns_r.shape[0]), np.ones(rtns_g.shape[0]) * 2.0]),
        np.concatenate([rtns_r, rtns_g]),
    )
    fig.savefig(output_dir + "/returns.png")

    fig, ax = plt.subplots()
    real_prop_r = successes_r / len(D_r)
    real_prop_g = successes_g / len(D_g)
    real_var_r = np.sqrt((real_prop_r * (1.0 - real_prop_r)) / len(D_r))
    real_var_g = np.sqrt((real_prop_g * (1.0 - real_prop_g)) / len(D_g))
    ax.bar(
        ["Real", "Generated"], [real_prop_r, real_prop_g], yerr=[real_var_r, real_var_g]
    )
    ax.text(0, real_prop_r / 2, f"N={len(D_r)}", ha="center", va="bottom")
    ax.text(1, real_prop_g / 2, f"N={len(D_g)}", ha="center", va="bottom")
    fig.savefig(output_dir + "/successes.png")

    fig, ax = plt.subplots()
    vp = ax.violinplot(
        [end_goal_dist_r, end_goal_dist_g], showmeans=True, showmedians=True
    )
    vp["cmeans"].set_color("orange")
    vp["cmeans"].set_label("Mean")
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_label("Median")
    ax.legend()
    ax.set_title("End Goal Distance (Real Versus Generated)")
    ax.set_xticks([y + 1 for y in range(2)], labels=["Real", "Generated"])
    ax.set_ylabel("Distance")
    ax.scatter(
        np.concatenate(
            [np.ones(len(end_goal_dist_r)), np.ones(len(end_goal_dist_g)) * 2.0]
        ),
        np.concatenate([end_goal_dist_r, end_goal_dist_g]),
    )
    fig.savefig(output_dir + "/end_goal_dist.png")

    fig, ax = plt.subplots()
    vp = ax.violinplot([frames_r, frames_g], showmeans=True, showmedians=True)
    vp["cmeans"].set_color("orange")
    vp["cmeans"].set_label("Mean")
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_label("Median")
    ax.legend()
    ax.set_title("Frames (Real Versus Generated)")
    ax.set_xticks([y + 1 for y in range(2)], labels=["Real", "Generated"])
    ax.set_ylabel("Frames")
    ax.scatter(
        np.concatenate([np.ones(len(frames_r)), np.ones(len(frames_g)) * 2.0]),
        np.concatenate([frames_r, frames_g]),
    )
    fig.savefig(output_dir + "/frames.png")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
