import argparse
import os
import sys

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_enable_cudnn_fmha=true"
)

sys.path.insert(0, os.path.abspath("../.."))
import h5py
import numpy as np
from argformat import StructuredFormatter

from transformers.training.train_model import train_pt
import jax
import jax.numpy as jnp


def main(argv):
    parser = argparse.ArgumentParser(
        description="Runs Preference Transformer Training Algorithm given a compiled dataset (in jax.numpy.array form)",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="HDF5 file containing data. \nThe file must have the datasets \n'Observations',timesteps',etc.",
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
        "-o",
        "--output_dir",
        type=str,
        default="~/busy-beeway/transformers/logs",
        help="Output directory for training logs and pickled models",
    )
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
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

    try:
        with h5py.File(data, "r") as f:
            key = jax.random.PRNGKey(seed)
            p_size = f["observations"].shape[0]
            shuffled_idx = jax.random.permutation(key, p_size)

            t_int = int(p_size * train_split)

            train_pt(
                f,
                jnp.sort(shuffled_idx[:t_int]),
                jnp.sort(shuffled_idx[t_int:]),
                key,
                seed,
                batch_size=batch_size,
                n_epochs=n_epochs,
                eval_period=eval_period,
                save_dir=output_dir,
                init_value=init_value,
                peak_value=peak_value,
                end_value=end_value,
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{data} not found, try running compile_preference_data.py first!"
        )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
