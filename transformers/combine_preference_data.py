import argparse
import sys
from pathlib import Path

from argformat import StructuredFormatter

from bb_data_loading import load_participant_list
from data_utils import load_preference_data


def main(argv):
    parser = argparse.ArgumentParser(
        description="Combines preference data for multiple participants",
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
        default="~/busy-beeway/transformers",
        help="Preference Data Directory. \nLooks for the preference_data folder in this directory.",
    )
    parser.add_argument(
        "-c",
        "--cpu",
        type=int,
        default=0,
        help="CPU ID (0 by default) for data loading. \nSetting to -1 lets Jax use a default device \n(a GPU if available).",
    )
    args = parser.parse_args(argv)
    cpu = args.cpu
    path = args.data_dir
    p_id = args.p_id
    S = load_participant_list(p_id)
    for p_id in S:
        P = load_preference_data(f"{path}/preference_data/{p_id}", sep_files=True, mmap_mode="r", cpu=cpu)
        
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
