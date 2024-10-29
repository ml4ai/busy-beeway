import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp

jax.config.update("jax_platforms", "cpu")
from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Script for removing NaN returns",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "day_list",
        metavar="DL",
        type=str,
        help="A .txt file containing a list of files (no extensions) to process",
    )
    parser.add_argument(
        "data",
        metavar="D",
        type=str,
        help="Data File",
    )
    args = parser.parse_args(argv)
    dl = load_list(args.day_list)
    with h5py.File(f"{args.data}", "r+") as f:
        for d in dl:
            dat = f[f"return_to_go_{d}"][:]
            dat = jnp.delete(dat, jnp.where(jnp.isnan(dat))[0], axis=0)
            del f[f"return_to_go_{d}"]
            f.create_dataset(f"return_to_go_{d}", data=dat, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
