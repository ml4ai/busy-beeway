import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import h5py
from argformat import StructuredFormatter
from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="This normalizes the observation features \nto the same scale. \nThis alters preference data in-place, \nMake a backup if this is an issue!",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="A .txt file containing a list of Participant IDs to process",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/transformers/preference_data_1",
        help="Data Directory for participant preference data.",
    )
    args = parser.parse_args(argv)
    p_id = load_list(args.p_id)
    data_dir = os.path.expanduser(args.data_dir)
    # first pass collects min max data
    mins_1 = []
    maxs_1 = []
    mins_2 = []
    maxs_2 = []
    for p in p_id:
        with h5py.File(f"{data_dir}/{p}.hdf5", "r") as f:
            ob = f["observations"][:]
            ob_t = ob != 0

            mins_1.append(np.min(np.min(ob, 1, initial=100, where=ob_t), 0))
            maxs_1.append(np.max(np.max(ob, 1, initial=-1, where=ob_t), 0))

            ob = f["observations_2"][:]
            ob_t = ob != 0
            mins_2.append(np.min(np.min(ob, 1, initial=100, where=ob_t), 0))
            maxs_2.append(np.max(np.max(ob, 1, initial=-1, where=ob_t), 0))

    min_1 = np.min(mins_1, 0)
    min_2 = np.min(mins_2, 0)
    max_1 = np.max(maxs_1, 0)
    max_2 = np.max(maxs_2, 0)
    maxmin_1 = max_1 - min_1
    maxmin_2 = max_2 - min_2
    # second pass applies scaling
    for p in p_id:
        with h5py.File(f"{data_dir}/{p}.hdf5", "r+") as f:
            ob = f["observations"][:]
            ob_t = ob != 0
            f["observations"][:, :, :-2] = np.add(
                1,
                np.divide(np.subtract(ob, min_1, where=ob_t), maxmin_1, where=ob_t),
                where=ob_t,
            )[:, :, :-2]

            ob = f["observations_2"][:]
            ob_t = ob != 0
            f["observations_2"][:, :, :-2] = np.add(
                1,
                np.divide(np.subtract(ob, min_2, where=ob_t), maxmin_2, where=ob_t),
                where=ob_t,
            )[:, :, :-2]

    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
