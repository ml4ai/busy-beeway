import getopt
import sys

from bb_data_loading import load_participant_data, load_participant_data_p
from data_utils import (
    compute_features,
    compute_features_p,
    create_preference_data,
    load_features_from_parquet,
)
from replayer import generate_stats, goal_only_replay, goal_only_replay_p, load_stats


def main(argv):
    parallel = False
    try:
        opts, args = getopt.getopt(argv, "hp", ["help,parallel"])
    except getopt.GetoptError:
        print("usage: save_preference_data.py [-p --parallel]")
        print("Options:")
        print("-h --help        : Prints the current help message")
        print(
            "-p --parallel    : Runs compute_features functions using multiprocessing"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("usage: save_preference_data.py [-p --parallel]")
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
    arc_sweep = (10, 360, 10)
    seed = 2024

    pd_not_loaded = True
    if parallel:
        try:
            RF = load_features_from_parquet("~/busy-beeway/pref_transformer/rf_save")
        except FileNotFoundError:
            try:
                stats = load_stats("~/busy-beeway/pref_transformer/p_stats.npy")
            except FileNotFoundError:
                CD = load_participant_data_p(p_id=p_id, path=path, control=3)
                stats = generate_stats(
                    CD, save_data="~/busy-beeway/pref_transformer/p_stats.npy"
                )
            D = load_participant_data_p(p_id=p_id, path=path)
            pd_not_loaded = False
            RD = goal_only_replay_p(D, stats, seed=seed)
            RF = compute_features_p(
                RD, arc_sweep, save_dir="~/busy-beeway/pref_transformer/rf_save"
            )
        try:
            F = load_features_from_parquet("~/busy-beeway/pref_transformer/f_save")
        except FileNotFoundError:
            if pd_not_loaded:
                D = load_participant_data_p(p_id=p_id, path=path)
            F = compute_features_p(
                D, arc_sweep, save_dir="~/busy-beeway/pref_transformer/f_save"
            )
        create_preference_data(RF, F, fill_size=fill_size, save_data=f"{p_id}.npz")
        sys.exit(0)

    try:
        RF = load_features_from_parquet("~/busy-beeway/pref_transformer/rf_save")
    except FileNotFoundError:
        try:
            stats = load_stats("~/busy-beeway/pref_transformer/p_stats.npy")
        except FileNotFoundError:
            CD = load_participant_data(p_id=p_id, path=path, control=3)
            stats = generate_stats(
                CD, save_data="~/busy-beeway/pref_transformer/p_stats.npy"
            )
        D = load_participant_data(p_id=p_id, path=path)
        pd_not_loaded = False
        RD = goal_only_replay(D, stats, seed=seed)
        RF = compute_features(
            RD, arc_sweep, save_dir="~/busy-beeway/pref_transformer/rf_save"
        )
    try:
        F = load_features_from_parquet("~/busy-beeway/pref_transformer/f_save")
    except FileNotFoundError:
        if pd_not_loaded:
            D = load_participant_data(p_id=p_id, path=path)
        F = compute_features(
            D, arc_sweep, save_dir="~/busy-beeway/pref_transformer/f_save"
        )
    create_preference_data(RF, F, fill_size=fill_size, save_data=f"{p_id}.npz")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])