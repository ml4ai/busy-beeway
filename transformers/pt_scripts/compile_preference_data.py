import argparse
import os, sys

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path
from tqdm import tqdm

from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import (
    load_participant_data,
    load_participant_data_p,
    load_list,
)
from transformers.data_utils.data_utils import (
    compute_features,
    compute_features_p,
    create_preference_data,
    load_features_from_parquet,
)
from transformers.replayer.replayer import (
    generate_stats,
    goal_only_replay,
    goal_only_replay_p,
    load_stats,
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compiles preference feature data into a digestible form for Preference Transformer",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="Either a single Participant ID or \na .txt file containing a list of \nParticipant IDs to process",
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
        default = 100,
        help="Trajectory segment size.",
    )
    parser.add_argument(
        "-a",
        "--arc_sweep",
        type=int,
        default=[10, 360, 10],
        nargs=3,
        help="Settings for arc sizes for feature computation. \n(Starting Size, Ending Size, Increment Size).",
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
        "-c",
        "--cache_features",
        action="store_true",
        help="Caches feature data.",
    )
    parser.add_argument(
        "-k",
        "--cache_stats",
        action="store_true",
        help="Caches movement statistics (recommended).",
    )
    parser.add_argument(
        "-l",
        "--load_features",
        action="store_true",
        help="Tries to load feature data from \ndirectories '<p_id>_f_save' and '<p_id>_rf_save' \nwithin the cache directory of \nthe current working directory.",
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
    arc_sweep = tuple(args.arc_sweep)
    seed = args.seed
    load_features = args.load_features
    load_p_stats = args.load_stats
    study = args.bbway
    if args.exclusion_list:
        L = load_list(args.exclusion_list)
    else:
        L = None

    Path(f"{o_path}/preference_data_{study}").mkdir(parents=True, exist_ok=True)
    if args.cache_stats:
        Path(f"{o_path}/cache").mkdir(parents=True, exist_ok=True)
        save_p_stats = f"{o_path}/cache/p_stats.npy"
    else:
        save_p_stats = None
    p_id = args.p_id
    if p_id.endswith(".txt"):
        S = load_list(p_id)
        for p_id in tqdm(S):
            if args.cache_features:
                Path(f"{o_path}/cache").mkdir(parents=True, exist_ok=True)
                save_f = f"{o_path}/cache/{p_id}_f_save"
                save_rf = f"{o_path}/cache/{p_id}_rf_save"
            else:
                save_f = None
                save_rf = None

            save_pref = f"{o_path}/preference_data_{study}/{p_id}.hdf5"

            pd_not_loaded = True
            if parallel:
                try:
                    if load_features:
                        RF = load_features_from_parquet(
                            f"{o_path}/cache/{p_id}_rf_save"
                        )
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
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
                    D = load_participant_data_p(
                        p_id=p_id, path=path, exclusion_list=L, study=study
                    )
                    pd_not_loaded = False
                    RD = goal_only_replay_p(D, stats, simulate_forward=False, seed=seed)
                    RF = compute_features_p(RD, arc_sweep, save_dir=save_rf)
                    del RD
                try:
                    if load_features:
                        F = load_features_from_parquet(f"{o_path}/cache/{p_id}_f_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    if pd_not_loaded:
                        D = load_participant_data_p(
                            p_id=p_id, path=path, exclusion_list=L, study=study
                        )
                    F = compute_features_p(D, arc_sweep, save_dir=save_f)
                    del D
                create_preference_data(
                    RF, F, split_size=split_size, save_data=save_pref
                )
                del RF
                del F
            else:
                try:
                    if load_features:
                        RF = load_features_from_parquet(
                            f"{o_path}/cache/{p_id}_rf_save"
                        )
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
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
                    D = load_participant_data(
                        p_id=p_id, path=path, exclusion_list=L, study=study
                    )
                    pd_not_loaded = False
                    RD = goal_only_replay(D, stats, simulate_forward=False, seed=seed)
                    RF = compute_features(RD, arc_sweep, save_dir=save_rf)
                    del RD
                try:
                    if load_features:
                        F = load_features_from_parquet(f"{o_path}/cache/{p_id}_f_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    if pd_not_loaded:
                        D = load_participant_data(
                            p_id=p_id, path=path, exclusion_list=L, study=study
                        )
                    F = compute_features(D, arc_sweep, save_dir=save_f)
                    del D
                create_preference_data(
                    RF, F, split_size=split_size, save_data=save_pref
                )
                del RF
                del F
        sys.exit(0)
    else:
        if args.cache_features:
            Path(f"{o_path}/cache").mkdir(parents=True, exist_ok=True)
            save_f = f"{o_path}/cache/{p_id}_f_save"
            save_rf = f"{o_path}/cache/{p_id}_rf_save"
        else:
            save_f = None
            save_rf = None

        save_pref = f"{o_path}/preference_data_{study}/{p_id}.hdf5"

        pd_not_loaded = True
        if parallel:
            try:
                if load_features:
                    RF = load_features_from_parquet(f"{o_path}/cache/{p_id}_rf_save")
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
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
                D = load_participant_data_p(
                    p_id=p_id, path=path, exclusion_list=L, study=study
                )
                pd_not_loaded = False
                RD = goal_only_replay_p(D, stats, simulate_forward=False, seed=seed)
                RF = compute_features_p(RD, arc_sweep, save_dir=save_rf)
                del RD
            try:
                if load_features:
                    F = load_features_from_parquet(f"{o_path}/cache/{p_id}_f_save")
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                if pd_not_loaded:
                    D = load_participant_data_p(
                        p_id=p_id, path=path, exclusion_list=L, study=study
                    )
                F = compute_features_p(D, arc_sweep, save_dir=save_f)
                del D
            create_preference_data(RF, F, split_size=split_size, save_data=save_pref)
            del RF
            del F
            sys.exit(0)

        try:
            if load_features:
                RF = load_features_from_parquet(f"{o_path}/cache/{p_id}_rf_save")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
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
            D = load_participant_data(
                p_id=p_id, path=path, exclusion_list=L, study=study
            )
            pd_not_loaded = False
            RD = goal_only_replay(D, stats, simulate_forward=False, seed=seed)
            RF = compute_features(RD, arc_sweep, save_dir=save_rf)
            del RD
        try:
            if load_features:
                F = load_features_from_parquet(f"{o_path}/cache/{p_id}_f_save")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if pd_not_loaded:
                D = load_participant_data(
                    p_id=p_id, path=path, exclusion_list=L, study=study
                )
            F = compute_features(D, arc_sweep, save_dir=save_f)
            del D
        create_preference_data(RF, F, split_size=split_size, save_data=save_pref)
        del RF
        del F
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
