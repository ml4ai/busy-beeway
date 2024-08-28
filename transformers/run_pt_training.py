import getopt
import sys

import numpy as np

from bb_data_loading import load_participant_data, load_participant_data_p
from data_utils import (
    compute_features,
    compute_features_p,
    create_preference_data,
    load_features_from_parquet,
    load_preference_data,
)
from replayer import generate_stats, goal_only_replay, goal_only_replay_p, load_stats
from train_model import train_pt


def main(argv):
    parallel = False
    try:
        opts, args = getopt.getopt(argv, "hp", ["help,parallel"])
    except getopt.GetoptError:
        print("usage: train_pt.py [-p --parallel]")
        print("Options:")
        print("-h --help        : Prints the current help message")
        print(
            "-p --parallel    : Runs compute_features functions using multiprocessing"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("usage: train_pt.py [-p --parallel]")
            print("Options:")
            print("-h --help        : Prints the current help message")
            print(
                "-p --parallel    : Runs compute_features functions using multiprocessing"
            )
            sys.exit(0)
        elif opt in ("-p", "--parallel"):
            parallel = True

    path = "~/busy-beeway/data/game_data"
    p_id = "auto-1ba807eecf3cf284"

    fill_size = None
    train_split = 23 / 28
    batch_size = 64
    n_epochs = 10
    arc_sweep = (10, 360, 10)
    seed = 2024

    pd_not_loaded = True
    if parallel:
        try:
            P = load_preference_data(f"~/busy-beeway/transformers/{p_id}.npz")
        except FileNotFoundError:
            try:
                RF = load_features_from_parquet(
                    "~/busy-beeway/transformers/rf_save"
                )
            except FileNotFoundError:
                try:
                    stats = load_stats("~/busy-beeway/transformers/p_stats.npy")
                except FileNotFoundError:
                    CD = load_participant_data_p(p_id=p_id, path=path, control=3)
                    stats = generate_stats(
                        CD, save_data="~/busy-beeway/transformers/p_stats.npy"
                    )
                D = load_participant_data_p(p_id=p_id, path=path)
                pd_not_loaded = False
                RD = goal_only_replay_p(D, stats, seed=seed)
                RF = compute_features_p(
                    RD, arc_sweep, save_dir="~/busy-beeway/transformers/rf_save"
                )
            try:
                F = load_features_from_parquet("~/busy-beeway/transformers/f_save")
            except FileNotFoundError:
                if pd_not_loaded:
                    D = load_participant_data_p(p_id=p_id, path=path)
                F = compute_features_p(
                    D, arc_sweep, save_dir="~/busy-beeway/transformers/f_save"
                )
            P = create_preference_data(
                RF, F, fill_size=fill_size, save_data=f"{p_id}.npz"
            )
    else:
        try:
            P = load_preference_data(f"~/busy-beeway/transformers/{p_id}.npz")
        except FileNotFoundError:
            try:
                RF = load_features_from_parquet(
                    "~/busy-beeway/transformers/rf_save"
                )
            except FileNotFoundError:
                try:
                    stats = load_stats("~/busy-beeway/transformers/p_stats.npy")
                except FileNotFoundError:
                    CD = load_participant_data(p_id=p_id, path=path, control=3)
                    stats = generate_stats(
                        CD, save_data="~/busy-beeway/transformers/p_stats.npy"
                    )
                D = load_participant_data(p_id=p_id, path=path)
                pd_not_loaded = False
                RD = goal_only_replay(D, stats, seed=seed)
                RF = compute_features(
                    RD, arc_sweep, save_dir="~/busy-beeway/transformers/rf_save"
                )
            try:
                F = load_features_from_parquet("~/busy-beeway/transformers/f_save")
            except FileNotFoundError:
                if pd_not_loaded:
                    D = load_participant_data(p_id=p_id, path=path)
                F = compute_features(
                    D, arc_sweep, save_dir="~/busy-beeway/transformers/f_save"
                )

            P = create_preference_data(
                RF, F, fill_size=fill_size, save_data=f"{p_id}.npz"
            )

    rng = np.random.default_rng(seed)
    p_size = P["observations"].shape[0]
    shuffle_idx = rng.permutation(p_size)

    t_int = int(p_size * train_split)

    training_data = {}
    test_data = {}
    for k in P.keys():
        P_k = P[k]
        training_data[k] = P_k[shuffle_idx[:t_int]]
        test_data[k] = P_k[shuffle_idx[t_int:]]
    if isinstance(P, np.lib.npyio.NpzFile):
        P.close()
    train_pt(
        training_data, test_data, batch_size=batch_size, n_epochs=n_epochs, seed=seed
    )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
