import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch.multiprocessing as multiprocessing
from argformat import StructuredFormatter
from flax import nnx

from transformers.data_utils.data_loader import Dec_H5Dataset
from transformers.models.pref_transformer import load_PT
from transformers.replayer.replayer import load_stats
from transformers.training.train_model import train_dt


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
        "output_type",
        metavar="O",
        type=str,
        help="This determines what the DT is mainly trying to predict. \nOptions should be Q (state-action value), \nS_D (discrete states), \nS_F (feature-based states), \nA_D (discrete actions), \nA_F (feature-based actions",
    )
    parser.add_argument(
        "-r",
        "--reward_function",
        type=str,
        default=None,
        help="A .ckpt directory containing reward model. \nThis is required for evaluating with a learned reward function",
    )
    parser.add_argument(
        "-e",
        "--eval_settings",
        type=int,
        default=[1, 10, 100, 0],
        nargs=4,
        help="Eval settings (period, number of episodes, target return, eval_type). \neval_type == 0 for Busy Beeway, 1 for AntMaze_Medium-v4",
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
        "-m",
        "--move_stats",
        type=str,
        default=None,
        help="File with obstacle stats. \nThis is required for Busy Beeway",
    )
    parser.add_argument(
        "-a",
        "--normalized_returns",
        action="store_true",
        help="Uses normalized returns for training. \nWorks with a learned reward function only!",
    )
    multiprocessing.set_start_method("forkserver")
    args = parser.parse_args(argv)
    data = os.path.expanduser(args.data)
    if args.reward_function:
        reward_function = os.path.expanduser(args.reward_function)
        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        r_model = load_PT(reward_function, checkpointer, on_cpu=True)
        r_model = nnx.jit(r_model, static_argnums=4)
        checkpointer.close()
        task_returns = False
    else:
        r_model = None
        task_returns = True
    if args.move_stats:
        move_stats = load_stats(args.move_stats)
    else:
        move_stats = None
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

    try:
        data = Dec_H5Dataset(data, normalized_returns=args.normalized_returns,task_returns=task_returns)
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
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{data} not found, try running compile_preference_data.py first!"
        )
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
