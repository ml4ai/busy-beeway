import argparse
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

from argformat import StructuredFormatter

from transformers.data_utils.bb_data_loading import load_list, load_participant_data_p
from transformers.data_utils.data_utils import max_seq_length


def main(argv):
    parser = argparse.ArgumentParser(
        description="Get max sequence length for a list of participants",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="A .txt file containing a list of participants.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="~/busy-beeway/data/game_data",
        help="Data directory",
    )
    parser.add_argument(
        "-e",
        "--exclusion_list",
        type=str,
        help="List of test sessions to exclude for each participant",
    )
    parser.add_argument(
        "-s",
        "--save_file",
        type=str,
        default=f"{Path.cwd()}/max_seq_length.txt",
        help="File to save output",
    )
    args = parser.parse_args(argv)
    pid = args.p_id
    e = args.exclusion_list
    data_dir = os.path.expanduser(args.data_dir)
    save_file = os.path.expanduser(args.save_file)

    pid = load_list(pid)
    e = load_list(e)
    max_l = []
    for i in pid:
        max_l.append(
            max_seq_length(
                load_participant_data_p(p_id=i, path=data_dir, exclusion_list=e)
            )
        )
    max_seq_l = max(max_l)
    with open(save_file, "w") as f:
        f.write(f"{max_seq_l}")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
