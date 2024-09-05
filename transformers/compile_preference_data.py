import argparse
import sys
from pathlib import Path

from argformat import StructuredFormatter

from bb_data_loading import (
    load_participant_data,
    load_participant_data_p,
    load_participant_list,
)
from data_utils import (
    compute_features,
    compute_features_p,
    create_preference_data,
    load_features_from_parquet,
)
from replayer import generate_stats, goal_only_replay, goal_only_replay_p, load_stats


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compiles preference feature data into a digestible form for Preference Transformer",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="Either a single Participant ID or \na .txt file containing a list of Participant IDs to process",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data Directory.",
    )
    parser.add_argument(
        "-f",
        "--fill_size",
        type=int,
        help="Padding size for sequences. \nUses max sequence length by default.",
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
    args = parser.parse_args(argv)
    parallel = args.parallel
    path = args.data_dir
    fill_size = args.fill_size
    arc_sweep = tuple(args.arc_sweep)
    seed = args.seed
    load_features = args.load_features
    load_p_stats = args.load_stats
    if args.cache_stats:
        save_p_stats = "p_stats.npy"
    else:
        save_p_stats = None
    p_id = args.p_id
    if Path(p_id).suffix == ".txt":
        S = load_participant_list(p_id)
        for p_id in S:
            if args.cache_features:
                save_f = f"{p_id}_f_save"
                save_rf = f"{p_id}_rf_save"
            else:
                save_f = None
                save_rf = None
                
            save_pref = f"{p_id}"

            pd_not_loaded = True
            if parallel:
                try:
                    if load_features:
                        RF = load_features_from_parquet(f"cache/{p_id}_rf_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    try:
                        if load_p_stats:
                            stats = load_stats("cache/p_stats.npy")
                        else:
                            raise FileNotFoundError
                    except FileNotFoundError:
                        CD = load_participant_data_p(p_id=p_id, path=path, control=3)
                        stats = generate_stats(CD, save_data=save_p_stats)
                    D = load_participant_data_p(p_id=p_id, path=path)
                    pd_not_loaded = False
                    RD = goal_only_replay_p(D, stats, seed=seed)
                    RF = compute_features_p(RD, arc_sweep, save_dir=save_rf)
                try:
                    if load_features:
                        F = load_features_from_parquet(f"cache/{p_id}_f_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    if pd_not_loaded:
                        D = load_participant_data_p(p_id=p_id, path=path)
                    F = compute_features_p(D, arc_sweep, save_dir=save_f)
                create_preference_data(RF, F, fill_size=fill_size, save_data=save_pref)
            else:
                try:
                    if load_features:
                        RF = load_features_from_parquet(f"cache/{p_id}_rf_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    try:
                        if load_p_stats:
                            stats = load_stats("cache/p_stats.npy")
                        else:
                            raise FileNotFoundError
                    except FileNotFoundError:
                        CD = load_participant_data(p_id=p_id, path=path, control=3)
                        stats = generate_stats(CD, save_data=save_p_stats)
                    D = load_participant_data(p_id=p_id, path=path)
                    pd_not_loaded = False
                    RD = goal_only_replay(D, stats, seed=seed)
                    RF = compute_features(RD, arc_sweep, save_dir=save_rf)
                try:
                    if load_features:
                        F = load_features_from_parquet(f"cache/{p_id}_f_save")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    if pd_not_loaded:
                        D = load_participant_data(p_id=p_id, path=path)
                    F = compute_features(D, arc_sweep, save_dir=save_f)
                create_preference_data(RF, F, fill_size=fill_size, save_data=save_pref)
        sys.exit(0)
    else:
        if args.cache_features:
            save_f = f"{p_id}_f_save"
            save_rf = f"{p_id}_rf_save"
        else:
            save_f = None
            save_rf = None
            
        save_pref = f"{p_id}"

        pd_not_loaded = True
        if parallel:
            try:
                if load_features:
                    RF = load_features_from_parquet(f"cache/{p_id}_rf_save")
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                try:
                    if load_stats:
                        stats = load_stats("cache/p_stats.npy")
                    else:
                        raise FileNotFoundError
                except FileNotFoundError:
                    CD = load_participant_data_p(p_id=p_id, path=path, control=3)
                    stats = generate_stats(CD, save_data=save_p_stats)
                D = load_participant_data_p(p_id=p_id, path=path)
                pd_not_loaded = False
                RD = goal_only_replay_p(D, stats, seed=seed)
                RF = compute_features_p(RD, arc_sweep, save_dir=save_rf)
            try:
                if load_features:
                    F = load_features_from_parquet(f"cache/{p_id}_f_save")
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                if pd_not_loaded:
                    D = load_participant_data_p(p_id=p_id, path=path)
                F = compute_features_p(D, arc_sweep, save_dir=save_f)
            create_preference_data(RF, F, fill_size=fill_size, save_data=save_pref)
            sys.exit(0)

        try:
            if load_features:
                RF = load_features_from_parquet(f"cache/{p_id}_rf_save")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            try:
                if load_stats:
                    stats = load_stats("cache/p_stats.npy")
                else:
                    raise FileNotFoundError
            except FileNotFoundError:
                CD = load_participant_data(p_id=p_id, path=path, control=3)
                stats = generate_stats(CD, save_data=save_p_stats)
            D = load_participant_data(p_id=p_id, path=path)
            pd_not_loaded = False
            RD = goal_only_replay(D, stats, seed=seed)
            RF = compute_features(RD, arc_sweep, save_dir=save_rf)
        try:
            if load_features:
                F = load_features_from_parquet(f"cache/{p_id}_f_save")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            if pd_not_loaded:
                D = load_participant_data(p_id=p_id, path=path)
            F = compute_features(D, arc_sweep, save_dir=save_f)
        create_preference_data(RF, F, fill_size=fill_size, save_data=save_pref)
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
