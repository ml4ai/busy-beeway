import argparse
import sys
from pathlib import Path
import os
import re

from argformat import StructuredFormatter
import pandas as pd


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def main(argv):
    parser = argparse.ArgumentParser(
        description="Get sorted list of days for participant",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "data_dir",
        metavar="D",
        type=str,
        help="The directory where the day by day \nproccessed data is located",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path for saving output",
    )
    args = parser.parse_args(argv)
    save_dir = os.path.expanduser(args.save_dir)
    data_dir = os.path.expanduser(args.data_dir)
    alist = []
    for i in os.scandir(data_dir):
        if i.name.endswith(".hdf5"):
            n,_ = os.path.splitext(i.name)
            alist.append(n)

    alist.sort(key=natural_keys)

    with open(f"{save_dir}/day_list.txt", "w") as f:
        for line in alist:
            f.write(f"{line}\n")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
