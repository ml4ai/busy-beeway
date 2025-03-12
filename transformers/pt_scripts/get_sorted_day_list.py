import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from argformat import StructuredFormatter
from tqdm import tqdm


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
        "p_id",
        metavar="PID",
        type=str,
        help="Either a single auto ID or \na .txt file containing a list of \nAuto IDs to process",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data Directory.",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path for saving output",
    )
    parser.add_argument(
        "-b",
        "--bbway",
        type=int,
        default=1,
        help="Busy Beeway study (1 or 2).",
    )
    args = parser.parse_args(argv)
    save_dir = os.path.expanduser(args.save_dir)
    data_dir = os.path.expanduser(args.data_dir)
    study = args.bbway
    alist = []
    p_id = args.p_id
    if p_id.endswith(".txt"):
        S = load_list(p_id)
        for p_id in tqdm(S):
            Path(f"{save_dir}/{p_id}").mkdir(parents=True, exist_ok=True)
            if study == 1:
                e_code = "T5"
            else:
                e_code = "D1"
            alist = []
            dir_list = os.scandir(data_dir)
            for i in dir_list:
                if i.is_dir():
                    if (
                        i.path.endswith(e_code)
                        and not (i.path.endswith("97D1"))
                        and not (i.path.endswith("aiD1"))
                    ):
                        e_path = f"{i.path}/{p_id}"
                        for j in os.scandir(e_path):
                            if j.is_dir():
                                alist.append(j.name)

            alist.sort(key=natural_keys)

            with open(f"{save_dir}/{p_id}/day_list.txt", "w") as f:
                for line in alist:
                    f.write(f"{line}\n")
        sys.exit(0)
    Path(f"{save_dir}/{p_id}").mkdir(parents=True, exist_ok=True)
    if study == 1:
        e_code = "T5"
    else:
        e_code = "D1"
    alist = []
    dir_list = os.scandir(data_dir)
    for i in dir_list:
        if i.is_dir():
            if (
                i.path.endswith(e_code)
                and not (i.path.endswith("97D1"))
                and not (i.path.endswith("aiD1"))
            ):
                e_path = f"{i.path}/{p_id}"
                for j in os.scandir(e_path):
                    if j.is_dir():
                        alist.append(j.name)

    alist.sort(key=natural_keys)

    with open(f"{save_dir}/{p_id}/day_list.txt", "w") as f:
        for line in alist:
            f.write(f"{line}\n")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
