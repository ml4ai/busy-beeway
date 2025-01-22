import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

from argformat import StructuredFormatter
from tqdm import tqdm
from transformers.data_utils.bb_data_loading import (
    load_list,
    load_participant_data,
    load_participant_data_by_day,
    load_participant_data_p,
)
from transformers.data_utils.data_utils import (
    compute_features,
    compute_features_p,
    create_preference_data,
)
from transformers.replayer.replayer import (
    generate_stats,
    load_stats,
    random_replay,
    random_replay_p,
)
from transformers.training.utils import load_pickle


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compiles preference feature data by day into a digestible form for Preference Transformer",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "reward",
        metavar="R",
        type=str,
        help="File with Reward function (as pickled dictionary)",
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="a single auto ID",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data Directory.",
    )
    parser.add_argument(
        "-e",
        "--exclusion_list",
        type=str,
        help="List of test sessions to exclude for each participant",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=Path.cwd(),
        help="Path for saving output. Saves in \ncurrent working directory if not set. \nCached files also get saved here.",
    )
    parser.add_argument(
        "-f",
        "--split_size",
        type=int,
        default=100,
        help="Trajectory segment size.",
    )
    parser.add_argument(
        "-n",
        "--num_state_features",
        type=int,
        default=15,
        help="Number of state features. \nThis helps seperate state features and actions.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=13138787,
        help="Random seed used throughout the program.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Attempts to run certain processes in parallel.",
    )
    parser.add_argument(
        "-k",
        "--cache_stats",
        action="store_true",
        help="Caches movement statistics (recommended).",
    )
    parser.add_argument(
        "-j",
        "--load_stats",
        action="store_true",
        help="Tries to load movement statistics \nfrom 'p_stats.npy' \nwithin the cache directory of \nthe current working directory (recommended).",
    )
    args = parser.parse_args(argv)
    reward = os.path.expanduser(args.reward)
    r_model = load_pickle(reward)["model"]
    parallel = args.parallel
    path = args.data_dir
    o_path = args.output_dir
    state_features = args.num_state_features
    if not Path(o_path).is_dir():
        raise FileNotFoundError(f"Cannot find output directory {o_path}!")
    split_size = args.split_size
    seed = args.seed
    load_p_stats = args.load_stats
    if args.exclusion_list:
        L = load_list(args.exclusion_list)
    else:
        L = []

    arc_sweep = None
    Path(f"{o_path}/returns_1").mkdir(parents=True, exist_ok=True)
    if args.cache_stats:
        Path(f"{o_path}/cache").mkdir(parents=True, exist_ok=True)
        save_p_stats = f"{o_path}/cache/p_stats.npy"
    else:
        save_p_stats = None
    p_id = args.p_id

    save_pref = f"{o_path}/returns_1/"

    if parallel:
        try:
            if load_stats:
                stats = load_stats(f"{o_path}/cache/p_stats.npy")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            CD = load_participant_data_p(
                p_id=p_id, path=path, control=3, exclusion_list=L, study=study
            )
            stats = generate_stats(CD, save_data=save_p_stats)
            del CD
        D = load_participant_data_p(p_id=p_id, path=path, exclusion_list=L, study=study)

        RD = random_replay_p(D, stats, seed=seed)
        RF = compute_features_p(RD, arc_sweep)
        del RD

        F = compute_features_p(D, arc_sweep)
        del D

        create_return_data(
            RF,
            F,
            r_model,
            state_features=state_features,
            split_size=split_size,
            save_data=save_pref+reward.split("/")[-2]+".hdf5",
        )
        del RF
        del F
        sys.exit(0)

    try:
        if load_stats:
            stats = load_stats(f"{o_path}/cache/p_stats.npy")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        CD = load_participant_data(
            p_id=p_id,
            path=path,
            control=3,
            exclusion_list=L,
            study=study,
        )
        stats = generate_stats(CD, save_data=save_p_stats)
        del CD
    D = load_participant_data(p_id=p_id, path=path, exclusion_list=L, study=study)

    RD = random_replay(D, stats, seed=seed)
    RF = compute_features(RD, arc_sweep)
    del RD

    F = compute_features(D, arc_sweep)
    del D

    create_return_data(
        RF,
        F,
        r_model,
        state_features=state_features,
        split_size=split_size,
        save_data=save_pref+reward.split("/")[-2]+".hdf5",
    )
    del RF
    del F
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
