import argparse
import sys
from pathlib import Path
import os

from argformat import StructuredFormatter
import pandas as pd


def main(argv):
    parser = argparse.ArgumentParser(
        description="Get participant and exclusion list",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "study_list",
        metavar="SL",
        type=str,
        help="A .txt file containing a list of participants and \nwhat study (bbway1 or bbway2) they were in.",
    )
    parser.add_argument(
        "verified_csv",
        metavar="VC",
        type=str,
        help="A .csv for confirming if a participant successfully \ncompleted a test session.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data directory for exclusion list",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path for saving output",
    )
    args = parser.parse_args(argv)
    study_list = args.study_list
    verified_csv = args.verified_csv
    save_dir = os.path.expanduser(args.save_dir)
    data_dir = os.path.expanduser(args.data_dir)
    sl = pd.read_csv(study_list, header=None)
    sl = list(sl[sl[2] == "bbway1"][0])
    with open(f"{save_dir}/participant_list.txt", "w") as f:
        for line in sl:
            f.write(f"{line}\n")

    vc = pd.read_csv(verified_csv)

    vc = vc[vc["User"].isin(sl)]
    vc = vc[vc["Verify"] < 1]

    Experiments = vc["Experiment"].to_numpy()
    Users = vc["User"].to_numpy()
    Sessions = vc["Session"].to_numpy()
    with open(f"{save_dir}/exclusion_list.txt", "w") as g:
        for i in range(vc.shape[0]):
            g.write(f"{data_dir}/{Experiments[i]}/{Users[i]}/{Sessions[i]}\n")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
