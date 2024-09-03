import argparse
from argformat import StructuredFormatter
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
    parser = argparse.ArgumentParser(
        description="Runs Preference Transformer Training Algorithm given a compiled dataset (in jax.numpy.array form)",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="Location of the data. \nA single .npz file or a directory \ncontaining several .npy files.",
    )
    parser.add_argument(
        "-t",
        "--training_split",
        type=float,
        default=0.7,
        help="Percentage of training data.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size. Powers of 2 work best.",
    )
    parser.add_argument(
        "-e",
        "--eval_period",
        type=int,
        default=1,
        help="Period in which the validation set is ran and evaluated.",
    )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "-s",
        "--seeds",
        type=int,
        default=13138787,
        help="Random seed used throughout the program.",
    )
    parser.add_argument(
        "-c",
        "--cpu",
        type=int,
        default=0,
        help="CPU ID (0 by default) for data loading. \nSetting to -1 lets Jax use a default device \n(a GPU if available).",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=[0.0, 1e-4, 0.0],
        nargs=3,
        help="Learning Rate parameters passed to optimizer. \nIt uses a Cosine Decay Schedule with warmup steps, \nso this option requires 3 arguments \n(initial learning rate, peak learning rate, end learning rate).",
    )
    args = parser.parse_args(argv)
    load_data = args.data

    train_split = args.training_split
    batch_size = args.batch_size
    eval_period = args.eval_period
    n_epochs = args.n_epochs
    seed = args.seed
    cpu = args.cpu
    learing_rate = args.learning_rate
    init_value = learing_rate[0]
    peak_value = learing_rate[1]
    end_value = learing_rate[2]
    if cpu < 0:
        cpu = None
    try:
        P = load_preference_data(load_data, sep_files=True, mmap_mode="r", cpu=cpu)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{load_data} not found, try running compile_preference_data.py first!"
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
        training_data,
        test_data,
        batch_size=batch_size,
        n_epochs=n_epochs,
        eval_period=eval_period,
        seed=seed,
        init_value=init_value,
        peak_value=peak_value,
        end_value=end_value,
    )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
