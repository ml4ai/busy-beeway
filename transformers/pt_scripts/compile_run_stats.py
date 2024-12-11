import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from pathlib import Path

from argformat import StructuredFormatter
from tqdm import tqdm

from transformers.data_utils.bb_data_loading import (
    load_list,
    load_participant_data_by_day,
)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Compiles preference feature data by day into a digestible form for Preference Transformer",
        formatter_class=StructuredFormatter,
    )
    parser.add_argument(
        "p_id",
        metavar="PID",
        type=str,
        help="A .txt file containing a list of \nAuto IDs to process",
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
        default="~/busy-beeway/data/game_data",
        help="Data Directory.",
    )
    parser.add_argument(
        "-e",
        "--exclusion_list",
        type=str,
        help="List of test sessions to exclude for each participant",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=Path.cwd(),
        help="Path for saving output. Saves in \ncurrent working directory if not set. \nCached files also get saved here.",
    )
    args = parser.parse_args(argv)
    study_list = args.study_list
    sl = pd.read_csv(study_list, header=None, names=["autoID", "subjectID", "study"])
    path = args.data_dir
    o_path = args.output_dir
    if not Path(o_path).is_dir():
        raise FileNotFoundError(f"Cannot find output directory {o_path}!")
    if args.exclusion_list:
        L = load_list(args.exclusion_list)
    else:
        L = []
    p_id = args.p_id
    S = load_list(p_id)
    control_frames = []
    frames = []
    level = []
    ai = []
    collisions = []
    dates = []
    subject_ids = []
    for p_id in tqdm(S):
        n_runs = 0
        D = load_participant_data_by_day(
            p_id=p_id, path=path, exclusion_list=L, study=1
        )
        for i in D.keys():
            n_runs += len(D[i])
            date = i.split(".")
            date = date[1] + "-" + date[2] + "-" + date[3]
            dates += [date] * len(D[i])
            for d in D[i]:
                control_frames.append(np.sum(d["player"]["userControl"].to_numpy()))
                frames.append(d["player"].shape[0])
                collisions.append(not (d["reached_goal"]))
                oc = d["player"]["obstacle_count"][0]
                sig = d["player"]["sigma"][0]
                match oc:
                    case 50:
                        level.append(9)
                    case 100:
                        level.append(10)
                    case 150:
                        level.append(11)
                match sig:
                    case 55:
                        ai.append(1)
                    case 62:
                        ai.append(2)
                    case 140:
                        ai.append(3)
                    case 100:
                        ai.append(4)
        subject_ids += [sl[sl["autoID"] == p_id]["subjectID"].to_numpy()[0]] * n_runs
    df = pd.DataFrame(
        {
            "AI": ai,
            "level": level,
            "n_frames": frames,
            "n_control_frames": control_frames,
            "collision": collisions,
            "subject_id": subject_ids,
            "date": dates,
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["subject_id","date", "level"], inplace=True)
    df.to_csv(f"{o_path}/bbway1_run_stats.csv", index=False)
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
