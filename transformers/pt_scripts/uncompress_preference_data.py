import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import h5py
from argformat import StructuredFormatter
from transformers.data_utils.bb_data_loading import load_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Removes compression on preference data for multiple participants. \nThis does not check for compression, but will return \nuncompressed versions of datasets regardless. \nWarning: This edits the existing files, \nyou might want to make a backup of the data!",
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
    for p in p_id:
        with h5py.File(f"{data_dir}/{p}.hdf5", "r+") as f:
            for k in f:
                dat = f[k][:]
                del f[k]
                f.create_dataset(k, data=dat, chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
