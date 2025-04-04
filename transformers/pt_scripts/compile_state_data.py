import argparse
import os
import sys
import itertools

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
    create_state_data,
)
from transformers.replayer.replayer import (
    generate_stats,
    load_stats,
    random_replay,
    random_replay_p,
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compiles state feature data by day into a digestible form for Preference Transformer",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="Either a single auto ID or \na .txt file containing a list of \nAuto IDs to process",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data Directory.",
    )
    parser.add_argument(
        "-l",
        "--day_lists",
        type=str,
        default="~/busy-beeway/data/day_lists",
        help="Directory of participant day lists",
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
        "-s",
        "--seed",
        type=int,
        default=13138787,
        help="Random seed used throughout the program.",
    )
    parser.add_argument(
        "-n",
        "--n_min_obstacles",
        type=int,
        default=6,
        help="Number of closest obstacles to consider in state feature space",
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
    parser.add_argument(
        "-b",
        "--bbway",
        type=int,
        default=1,
        help="Busy Beeway study (1 or 2).",
    )
    args = parser.parse_args(argv)
    parallel = args.parallel
    path = args.data_dir
    o_path = args.output_dir
    if not Path(o_path).is_dir():
        raise FileNotFoundError(f"Cannot find output directory {o_path}!")
    split_size = args.split_size
    seed = args.seed
    load_p_stats = args.load_stats
    study = args.bbway
    day_lists = args.day_lists
    if args.exclusion_list:
        L = load_list(args.exclusion_list)
    else:
        L = []

    Path(f"{o_path}/state_data_{study}").mkdir(parents=True, exist_ok=True)
    if args.cache_stats:
        Path(f"{o_path}/cache").mkdir(parents=True, exist_ok=True)
        save_p_stats = f"{o_path}/cache/p_stats.npy"
    else:
        save_p_stats = None
    p_id = args.p_id
    if p_id.endswith(".txt"):
        S = load_list(p_id)
        for p_id in tqdm(S):
            day_list = load_list(day_lists + "/" + p_id + "/day_list.txt")
            save_pref = f"{o_path}/state_data_{study}/"

            if parallel:
                try:
                    if load_p_stats:
                        stats = load_stats(f"{o_path}/cache/p_stats.npy")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    CD = load_participant_data_p(
                        p_id=p_id,
                        path=path,
                        control=3,
                        exclusion_list=L,
                        study=study,
                    )
                    stats = generate_stats(CD, save_data=save_p_stats)
                    del CD
                D = load_participant_data_by_day(
                    p_id=p_id, path=path, exclusion_list=L, study=study
                )

                F = itertools.chain.from_iterable(
                    [
                        compute_features_p(D[key], day=i,n_min_obstacles=args.n_min_obstacles)
                        for i, key in enumerate(day_list)
                        if key in D
                    ]
                )
                save_d_pref = save_pref + f"{p_id}.hdf5"
                create_state_data(
                    F,
                    state_features=8 + 3*n_min_obstacles,
                    split_size=split_size,
                    save_data=save_d_pref,
                )
                del F
            else:
                try:
                    if load_p_stats:
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
                D = load_participant_data_by_day(
                    p_id=p_id, path=path, exclusion_list=L, study=study
                )

                F = itertools.chain.from_iterable(
                    [
                        compute_features_p(D[key], day=i,n_min_obstacles=args.n_min_obstacles)
                        for i, key in enumerate(day_list)
                        if key in D
                    ]
                )

                save_d_pref = save_pref + f"{p_id}.hdf5"
                create_state_data(
                    F,
                    state_features=8 + 3*n_min_obstacles,
                    split_size=split_size,
                    save_data=save_d_pref,
                )
                del F
        sys.exit(0)
    else:
        day_list = load_list(day_lists + "/" + p_id + "/day_list.txt")
        save_pref = f"{o_path}/state_data_{study}/"

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
            D = load_participant_data_by_day(
                p_id=p_id, path=path, exclusion_list=L, study=study
            )

            F = itertools.chain.from_iterable(
                [
                    compute_features_p(D[key], day=i,n_min_obstacles=args.n_min_obstacles)
                    for i, key in enumerate(day_list)
                    if key in D
                ]
            )

            save_d_pref = save_pref + f"{p_id}.hdf5"
            create_state_data(
                F,
                state_features=8 + 3*n_min_obstacles,
                split_size=split_size,
                save_data=save_d_pref,
            )
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
        D = load_participant_data_by_day(
            p_id=p_id, path=path, exclusion_list=L, study=study
        )

        F = itertools.chain.from_iterable(
            [
                compute_features_p(D[key], day=i,n_min_obstacles=args.n_min_obstacles)
                for i, key in enumerate(day_list)
                if key in D
            ]
        )

        save_d_pref = save_pref + f"{p_id}.hdf5"
        create_state_data(
            F,
            state_features=8 + 3*n_min_obstacles,
            split_size=split_size,
            save_data=save_d_pref,
        )
        del F
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
