import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import jax
import jax.numpy as jnp
import numpy as np
import torch.multiprocessing as multiprocessing
from argformat import StructuredFormatter
from transformers.data_utils.data_loader import Dec_H5Dataset
from transformers.training.train_model import train_dt
from transformers.training.utils import load_pickle
from transformers.replayer.replayer import load_stats


def main(argv):
    parser = argparse.ArgumentParser(
        description="Runs Decision Transformer Training Algorithm given a compiled dataset (in jax.numpy.array form)",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="HDF5 file containing data. \nThe file must have the datasets \n'Observations',timesteps',etc.",
    )
    parser.add_argument(
        "reward",
        metavar="R",
        type=str,
        help="File containing reward model",
    )
    parser.add_argument(
        "output_type",
        metavar="O",
        type=str,
        help="This determines what the DT is mainly trying to predict. \nOptions should be Q (state-action value), \nS_D (discrete states), \nS_F (feature-based states), \nA_D (discrete actions), \nA_F (feature-based actions",
    )
    parser.add_argument(
        "-e",
        "--eval_settings",
        type=int,
        default=[1, 10, 100, 500, 0],
        nargs=4,
        help="Eval settings (period, number of episodes, target return, max horizon, eval_type)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size. Powers of 2 work best.",
    )
    parser.add_argument(
        "-n", "--n_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=13138787,
        help="Random seed used throughout the program.",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=[0.0, 1e-4, 0.0],
        nargs=3,
        help="Learning Rate parameters passed to optimizer. \nIt uses a Cosine Decay Schedule with warmup steps, \nso this option requires 3 arguments \n(initial learning rate, peak learning rate, end learning rate).",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=256,
        help="Sets embedding dimensions (with some layer \ndimensions set as a function \nof this value.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="Number of workers assigned to handle data loading.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers/logs",
        help="Output directory for training logs and pickled models",
    )
    parser.add_argument(
        "-p",
        "--pretrained_model",
        type=str,
        default=None,
        help="File with pickled pretrained model",
    )
    parser.add_argument(
        "-m",
        "--move_stats",
        type=str,
        default="~/busy-beeway/transformers/t0012/cache/p_stats.npy",
        help="File with obstacle stats",
    )
    multiprocessing.set_start_method("forkserver")
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
    r_model = load_pickle(os.path.expanduser(args.reward))["model"]
    move_stats = load_stats(args.move_stats)
    batch_size = args.batch_size
    eval_settings = args.eval_settings
    n_epochs = args.n_epochs
    seed = args.seed
    learning_rate = args.learning_rate
    output_dir = os.path.expanduser(args.output_dir)
    init_value = learning_rate[0]
    peak_value = learning_rate[1]
    end_value = learning_rate[2]
    dim = args.dim
    workers = args.workers
    pm = args.pretrained_model
    if pm is not None:
        pm = os.path.expanduser(pm)
        pm = load_pickle(pm)["model"]
        pm = pm._train_state.params
    try:
        data = Dec_H5Dataset(data)
        train_dt(
            data,
            r_model,
            move_stats,
            seed,
            output_type=args.output_type,
            batch_size=batch_size,
            num_workers=workers,
            n_epochs=n_epochs,
            eval_settings=eval_settings,
            save_dir=output_dir,
            init_value=init_value,
            peak_value=peak_value,
            end_value=end_value,
            embd_dim=dim,
            pretrained_params=pm,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{data} not found, try running compile_preference_data.py first!"
        )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
