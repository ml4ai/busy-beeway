import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import jax

import jax.numpy as jnp
import numpy as np
import h5py
from argformat import StructuredFormatter

from transformers.data_utils.data_loader import Pref_H5Dataset, Pref_H5Dataset_minari
from transformers.training.train_model import train_pt
import torch.multiprocessing as multiprocessing


def main(argv):
    parser = argparse.ArgumentParser(
        description="Runs Preference Transformer Training Algorithm given a compiled dataset (in jax.numpy.array form)",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "data_1",
        metavar="D",
        type=str,
        help="HDF5 file containing data. \nThe file must have the datasets \n'Observations',timesteps',etc. \nThis assumes that a comparison set is included, \nif no data_2 is given",
    )
    parser.add_argument(
        "-a",
        "--data_2",
        type=str,
        default=None,
        help="HDF5 file containing comparison data for dataset 1. ",
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
        "-m",
        "--max_episode_length",
        type=int,
        default=None,
        help="Pre-known max episode length",
    )
    multiprocessing.set_start_method("forkserver")
    args = parser.parse_args(argv)
    data_1 = os.path.expanduser(args.data_1)
    if args.data_2 is not None:
        data_2 = os.path.expanduser(args.data_2)
    train_split = args.training_split
    batch_size = args.batch_size
    eval_period = args.eval_period
    n_epochs = args.n_epochs
    seed = args.seed
    learning_rate = args.learning_rate
    output_dir = os.path.expanduser(args.output_dir)
    init_value = learning_rate[0]
    peak_value = learning_rate[1]
    end_value = learning_rate[2]
    dim = args.dim
    workers = args.workers
    rng = np.random.default_rng(seed)
    seed, _ = rng.integers(0, 10000, 2)
    try:
        if args.data_2 is not None:
            m_idxs = []
            with h5py.File(data_1, "r") as f:
                with h5py.File(data_2, "r") as g:
                    m_size = g["states"].shape[0]
                    if args.max_episode_length is None:
                        mep = np.max(
                        [np.max(f["timesteps"][:]), np.max(g["timesteps"][:])]
                        )
                    else:
                        mep = args.max_episode_length
                        
                    for m in range(m_size):
                        m_static = g["states"][m, 0, -4:]
                        t_static = f["states"][:, 0, -4:]
                        matches = np.argwhere(np.all(t_static == m_static, axis=1))[:, 0]
                        if matches.shape[0] > 0:
                            m_idxs.append(rng.choice(matches))
                        else:
                            m_idxs.append(rng.choice(t_static.shape[0]))
            data = Pref_H5Dataset(data_1, data_2, np.asarray(m_idxs), mep)
        else:
            data = Pref_H5Dataset_minari(data_1, args.max_episode_length)
        train_pt(
            data,
            seed,
            train_split=train_split,
            batch_size=batch_size,
            num_workers=workers,
            n_epochs=n_epochs,
            eval_period=eval_period,
            save_dir=output_dir,
            init_value=init_value,
            peak_value=peak_value,
            end_value=end_value,
            embd_dim=dim,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{data} not found, try running compile_preference_data.py first!"
        )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
