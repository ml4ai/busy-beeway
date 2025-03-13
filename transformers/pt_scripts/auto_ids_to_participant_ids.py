import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

import h5py
from argformat import StructuredFormatter
from transformers.data_utils.bb_data_loading import load_list
import pandas as pd
import numpy as np


def main(argv):
    parser = argparse.ArgumentParser(
        description="Converts auto IDs to participant IDs. \nIt also merges data from same participant with \nmultiple auto IDs. \nMerged datasets are chunked and uncompressed automatically",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "study_list",
        metavar="SL",
        type=str,
        help="A .txt file containing a list of participants and \nwhat study (bbway1 or bbway2) they were in.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/transformers/preference_data_1",
        help="Data Directory for participant preference data.",
    )
    parser.add_argument(
        "-b",
        "--bbway",
        type=int,
        default=1,
        help="Busy Beeway study (1 or 2).",
    )
    args = parser.parse_args(argv)
    study_list = args.study_list
    study = args.bbway
    data_dir = os.path.expanduser(args.data_dir)
    sl = pd.read_csv(study_list, header=None)
    sl = sl[sl[2] == f"bbway{study}"]
    gb = sl.groupby(1)
    dfs = [gb.get_group(x) for x in gb.groups]
    for d in dfs:
        if d.shape[0] == 1:
            try:
                os.rename(
                    f"{data_dir}/{d[0].iloc[0]}.hdf5", f"{data_dir}/{d[1].iloc[0]}.hdf5"
                )
            except FileNotFoundError:
                pass
        else:
            sts = []
            acts = []
            ts = []
            am = []
            sts2 = []
            acts2 = []
            ts2 = []
            am2 = []
            lab = []
            for p in d[0]:
                try:
                    with h5py.File(f"{data_dir}/{p}.hdf5", "r") as f:
                        sts.append(f["states"][:])
                        acts.append(f["actions"][:])
                        ts.append(f["timesteps"][:])
                        am.append(f["attn_mask"][:])
                        lab.append(f["labels"][:])
                    os.remove(f"{data_dir}/{p}.hdf5")
                except FileNotFoundError:
                    pass
            with h5py.File(f"{data_dir}/{d[1].iloc[0]}.hdf5", "a") as f:
                f.create_dataset("states", data=np.concatenate(sts), chunks=True)
                f.create_dataset("actions", data=np.concatenate(acts), chunks=True)
                f.create_dataset("timesteps", data=np.concatenate(ts), chunks=True)
                f.create_dataset("attn_mask", data=np.concatenate(am), chunks=True)
                f.create_dataset("labels", data=np.concatenate(lab), chunks=True)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
