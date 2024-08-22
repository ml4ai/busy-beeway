import os
from itertools import chain, groupby
from multiprocessing import Pool
from operator import itemgetter

import numpy as np
import pandas as pd


def load_attempt_data(
    lvl,
    attempt,
    path,
    skip=0,
    control=1,
):

    p_file = f"{path}/player.{lvl}.{attempt}.0.data.csv"
    match lvl:
        case 9:
            obs_end = 58
        case 10:
            obs_end = 108
        case 11:
            obs_end = 158
    if control != 2 and control != 3:
        try:
            p_df = pd.read_csv(p_file, usecols=["posX", "posY", "angle"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
            )

        # adjusts negative angles to be within positive 0 to 360
        p_angles = p_df["angle"].to_numpy()
        p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
        p_df["angle"] = p_angles
        o_dfs = []
        for i in range(9, obs_end + 1):
            o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
            try:
                o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                )

            o_dfs.append(o)

        g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
        try:
            g_df = pd.read_csv(g_file, usecols=["At", "From"])
        except:
            raise FileNotFoundError(
                f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
            )

        D = []
        if g_df.shape[0] > 0:
            for i in range(g_df.shape[0]):
                p_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                p_pos = p_pos.reset_index(drop=True)
                p_pos = p_pos[
                    (p_pos.index % (skip + 1) == 0)
                    | (p_pos.index == (p_pos.shape[0] - 1))
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos["t"] = p_pos.index

                O_list = []
                for o_id, o_df in enumerate(o_dfs):
                    O = o_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    O = O.reset_index(drop=True)
                    O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                    O = O.reset_index(drop=True)
                    O["t"] = O.index
                    O["id"] = o_id
                    O_list.append(O)
                o_pos = pd.concat(O_list)
                o_pos.reset_index

                if i < 6:
                    k = i + 2
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                else:
                    k = 1
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                    )
                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                D.append(
                    {
                        "player": p_pos,
                        "obstacles": o_pos,
                        "reached_goal": True,
                        "goal": c_g,
                    }
                )

            if g_df.shape[0] < 7:
                p_pos = p_df.iloc[
                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos = p_pos[
                    (p_pos.index % (skip + 1) == 0)
                    | (p_pos.index == (p_pos.shape[0] - 1))
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos["t"] = p_pos.index

                O_list = []
                for o_id, o_df in enumerate(o_dfs):
                    O = o_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    O = O.reset_index(drop=True)
                    O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                    O = O.reset_index(drop=True)
                    O["t"] = O.index
                    O["id"] = o_id
                    O_list.append(O)
                o_pos = pd.concat(O_list)
                o_pos.reset_index

                if g_df.shape[0] == 6:
                    k = 1
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                else:
                    k = g_df.shape[0] + 2
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                    )
                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                D.append(
                    {
                        "player": p_pos,
                        "obstacles": o_pos,
                        "reached_goal": False,
                        "goal": c_g,
                    }
                )

        else:
            p_df = p_df.iloc[:-1,]
            p_df = p_df[
                (p_df.index % (skip + 1) == 0) | (p_df.index == (p_df.shape[0] - 1))
            ]
            p_df = p_df.reset_index(drop=True)
            p_df["t"] = p_df.index

            O_list = []
            for o_id, o_df in enumerate(o_dfs):
                O = o_df.iloc[:-1,]
                O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                O = O.reset_index(drop=True)
                O["t"] = O.index
                O["id"] = o_id
                O_list.append(O)
            o_pos = pd.concat(O_list)
            o_pos.reset_index

            c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
            try:
                c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
            except:
                raise FileNotFoundError(
                    f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                )
            c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
            D.append(
                {"player": p_df, "obstacles": o_pos, "reached_goal": False, "goal": c_g}
            )

        return D
    else:
        if control == 2:
            try:
                p_df = pd.read_csv(
                    p_file, usecols=["posX", "posY", "angle", "userControl"]
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
                )

            # adjusts negative angles to be within positive 0 to 360
            p_angles = p_df["angle"].to_numpy()
            p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
            p_df["angle"] = p_angles
            o_dfs = []
            for i in range(9, obs_end + 1):
                o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
                try:
                    o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                    )

                o_dfs.append(o)

            g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
            try:
                g_df = pd.read_csv(g_file, usecols=["At", "From"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
                )

            D = []
            if g_df.shape[0] > 0:
                for i in range(g_df.shape[0]):
                    if i < 6:
                        k = i + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    else:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )

                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == True].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": True,
                                    "goal": c_g,
                                }
                            )

                if g_df.shape[0] < 7:
                    if g_df.shape[0] == 6:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    else:
                        k = g_df.shape[0] + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )
                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == True].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (
                                        p_df.shape[0] - 1
                                    ),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": False,
                                    "goal": c_g,
                                }
                            )

            else:
                c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                    )

                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                p_df = p_df.iloc[:-1,]
                h_id = p_df[p_df["userControl"] == True].index.to_list()
                if h_id:
                    h_control = [
                        list(map(itemgetter(1), g))
                        for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                    ]
                    for k in h_control:
                        p_pos = p_df.iloc[k[0] : (k[-1] + 1), 0:3]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos = p_pos[
                            (p_pos.index % (skip + 1) == 0)
                            | (p_pos.index == (p_pos.shape[0] - 1))
                        ]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos["t"] = p_pos.index

                        O_list = []
                        for o_id, o_df in enumerate(o_dfs):
                            O = o_df.iloc[:-1,]
                            O = O.iloc[k[0] : (k[-1] + 1),]
                            O = O[
                                (O.index % (skip + 1) == 0)
                                | (O.index == (O.shape[0] - 1))
                            ]
                            O = O.reset_index(drop=True)
                            O["t"] = O.index
                            O["id"] = o_id
                            O_list.append(O)
                        o_pos = pd.concat(O_list)
                        o_pos.reset_index

                        D.append(
                            {
                                "player": p_pos,
                                "obstacles": o_pos,
                                "reached_goal": False,
                                "goal": c_g,
                            }
                        )

            return D
        else:
            try:
                p_df = pd.read_csv(
                    p_file, usecols=["posX", "posY", "angle", "userControl"]
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
                )

            # adjusts negative angles to be within positive 0 to 360
            p_angles = p_df["angle"].to_numpy()
            p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
            p_df["angle"] = p_angles
            o_dfs = []
            for i in range(9, obs_end + 1):
                o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
                try:
                    o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                    )

                o_dfs.append(o)

            g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
            try:
                g_df = pd.read_csv(g_file, usecols=["At", "From"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
                )

            D = []
            if g_df.shape[0] > 0:
                for i in range(g_df.shape[0]):
                    if i < 6:
                        k = i + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    else:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )

                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == False].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": True,
                                    "goal": c_g,
                                }
                            )

                if g_df.shape[0] < 7:
                    if g_df.shape[0] == 6:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    else:
                        k = g_df.shape[0] + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )
                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == False].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (
                                        p_df.shape[0] - 1
                                    ),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": False,
                                    "goal": c_g,
                                }
                            )

            else:
                c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                    )

                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                p_df = p_df.iloc[:-1,]
                h_id = p_df[p_df["userControl"] == False].index.to_list()
                if h_id:
                    h_control = [
                        list(map(itemgetter(1), g))
                        for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                    ]
                    for k in h_control:
                        p_pos = p_df.iloc[k[0] : (k[-1] + 1), 0:3]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos = p_pos[
                            (p_pos.index % (skip + 1) == 0)
                            | (p_pos.index == (p_pos.shape[0] - 1))
                        ]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos["t"] = p_pos.index

                        O_list = []
                        for o_id, o_df in enumerate(o_dfs):
                            O = o_df.iloc[:-1,]
                            O = O.iloc[k[0] : (k[-1] + 1),]
                            O = O[
                                (O.index % (skip + 1) == 0)
                                | (O.index == (O.shape[0] - 1))
                            ]
                            O = O.reset_index(drop=True)
                            O["t"] = O.index
                            O["id"] = o_id
                            O_list.append(O)
                        o_pos = pd.concat(O_list)
                        o_pos.reset_index

                        D.append(
                            {
                                "player": p_pos,
                                "obstacles": o_pos,
                                "reached_goal": False,
                                "goal": c_g,
                            }
                        )

            return D
    return None


def load_lvl_data(
    lvl,
    path,
    skip=0,
    control=1,
):
    D = []
    a = 0
    while True:
        d = load_attempt_data(lvl, a, path, skip, control)
        if d:
            D += d
        a += 1
        p_file = f"{path}/player.{lvl}.{a}.0.data.csv"
        if not (os.path.isfile(os.path.expanduser(p_file))):
            break
    return D


def load_test_data(
    path,
    skip=0,
    control=1,
):
    D = []
    for i in range(9, 12):
        d = load_lvl_data(i, path, skip, control)
        if d:
            D += d
    return D


# Path is a directory for a participant for a given experiment over several different days (e.g., Experiment_1T5)
def load_experiment_data(path, skip=0, control=1):
    D = []
    dir_path = os.path.expanduser(path)
    dir_list = os.scandir(dir_path)
    for i in dir_list:
        if i.is_dir():
            d = load_test_data(i.path, skip, control)
            if d:
                D += d
    return D


# p_id is a participant id (e.g., auto-1ba807eecf3cf284). This function will look for experiment data for that paricipant for a given path variable.
# It assumes that the format of the directories within the path are set-up as <experiment>/<participant_id>/<test days>/<data files>. There should be
# No other types of directories other than experiment folders or there could be unexplained behavior!
def load_participant_data(p_id, path="~/busy-beeway/data/game_data", skip=0, control=1):
    D = []
    dir_path = os.path.expanduser(path)
    dir_list = os.scandir(dir_path)
    for i in dir_list:
        if i.is_dir():
            e_path = f"{i.path}/{p_id}"
            d = load_experiment_data(e_path, skip, control)
            if d:
                D += d
    return D


def load_attempt_data_p(f):
    lvl, attempt, path, skip, control = f
    p_file = f"{path}/player.{lvl}.{attempt}.0.data.csv"
    match lvl:
        case 9:
            obs_end = 58
        case 10:
            obs_end = 108
        case 11:
            obs_end = 158
    if control != 2 and control != 3:
        try:
            p_df = pd.read_csv(p_file, usecols=["posX", "posY", "angle"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
            )

        # adjusts negative angles to be within positive 0 to 360
        p_angles = p_df["angle"].to_numpy()
        p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
        p_df["angle"] = p_angles
        o_dfs = []
        for i in range(9, obs_end + 1):
            o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
            try:
                o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                )

            o_dfs.append(o)

        g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
        try:
            g_df = pd.read_csv(g_file, usecols=["At", "From"])
        except:
            raise FileNotFoundError(
                f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
            )

        D = []
        if g_df.shape[0] > 0:
            for i in range(g_df.shape[0]):
                p_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                p_pos = p_pos.reset_index(drop=True)
                p_pos = p_pos[
                    (p_pos.index % (skip + 1) == 0)
                    | (p_pos.index == (p_pos.shape[0] - 1))
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos["t"] = p_pos.index

                O_list = []
                for o_id, o_df in enumerate(o_dfs):
                    O = o_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    O = O.reset_index(drop=True)
                    O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                    O = O.reset_index(drop=True)
                    O["t"] = O.index
                    O["id"] = o_id
                    O_list.append(O)
                o_pos = pd.concat(O_list)
                o_pos.reset_index

                if i < 6:
                    k = i + 2
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                else:
                    k = 1
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                    )
                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                D.append(
                    {
                        "player": p_pos,
                        "obstacles": o_pos,
                        "reached_goal": True,
                        "goal": c_g,
                    }
                )

            if g_df.shape[0] < 7:
                p_pos = p_df.iloc[
                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos = p_pos[
                    (p_pos.index % (skip + 1) == 0)
                    | (p_pos.index == (p_pos.shape[0] - 1))
                ]
                p_pos = p_pos.reset_index(drop=True)
                p_pos["t"] = p_pos.index

                O_list = []
                for o_id, o_df in enumerate(o_dfs):
                    O = o_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    O = O.reset_index(drop=True)
                    O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                    O = O.reset_index(drop=True)
                    O["t"] = O.index
                    O["id"] = o_id
                    O_list.append(O)
                o_pos = pd.concat(O_list)
                o_pos.reset_index

                if g_df.shape[0] == 6:
                    k = 1
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                else:
                    k = g_df.shape[0] + 2
                    c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                    )
                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                D.append(
                    {
                        "player": p_pos,
                        "obstacles": o_pos,
                        "reached_goal": False,
                        "goal": c_g,
                    }
                )

        else:
            p_df = p_df.iloc[:-1,]
            p_df = p_df[
                (p_df.index % (skip + 1) == 0) | (p_df.index == (p_df.shape[0] - 1))
            ]
            p_df = p_df.reset_index(drop=True)
            p_df["t"] = p_df.index

            O_list = []
            for o_id, o_df in enumerate(o_dfs):
                O = o_df.iloc[:-1,]
                O = O[(O.index % (skip + 1) == 0) | (O.index == (O.shape[0] - 1))]
                O = O.reset_index(drop=True)
                O["t"] = O.index
                O["id"] = o_id
                O_list.append(O)
            o_pos = pd.concat(O_list)
            o_pos.reset_index

            c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
            try:
                c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
            except:
                raise FileNotFoundError(
                    f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                )
            c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
            D.append(
                {"player": p_df, "obstacles": o_pos, "reached_goal": False, "goal": c_g}
            )

        return D
    else:
        if control == 2:
            try:
                p_df = pd.read_csv(
                    p_file, usecols=["posX", "posY", "angle", "userControl"]
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
                )

            # adjusts negative angles to be within positive 0 to 360
            p_angles = p_df["angle"].to_numpy()
            p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
            p_df["angle"] = p_angles
            o_dfs = []
            for i in range(9, obs_end + 1):
                o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
                try:
                    o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                    )

                o_dfs.append(o)

            g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
            try:
                g_df = pd.read_csv(g_file, usecols=["At", "From"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
                )

            D = []
            if g_df.shape[0] > 0:
                for i in range(g_df.shape[0]):
                    if i < 6:
                        k = i + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    else:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )

                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == True].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": True,
                                    "goal": c_g,
                                }
                            )

                if g_df.shape[0] < 7:
                    if g_df.shape[0] == 6:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    else:
                        k = g_df.shape[0] + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )
                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == True].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (
                                        p_df.shape[0] - 1
                                    ),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": False,
                                    "goal": c_g,
                                }
                            )

            else:
                c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                    )

                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                p_df = p_df.iloc[:-1,]
                h_id = p_df[p_df["userControl"] == True].index.to_list()
                if h_id:
                    h_control = [
                        list(map(itemgetter(1), g))
                        for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                    ]
                    for k in h_control:
                        p_pos = p_df.iloc[k[0] : (k[-1] + 1), 0:3]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos = p_pos[
                            (p_pos.index % (skip + 1) == 0)
                            | (p_pos.index == (p_pos.shape[0] - 1))
                        ]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos["t"] = p_pos.index

                        O_list = []
                        for o_id, o_df in enumerate(o_dfs):
                            O = o_df.iloc[:-1,]
                            O = O.iloc[k[0] : (k[-1] + 1),]
                            O = O[
                                (O.index % (skip + 1) == 0)
                                | (O.index == (O.shape[0] - 1))
                            ]
                            O = O.reset_index(drop=True)
                            O["t"] = O.index
                            O["id"] = o_id
                            O_list.append(O)
                        o_pos = pd.concat(O_list)
                        o_pos.reset_index

                        D.append(
                            {
                                "player": p_pos,
                                "obstacles": o_pos,
                                "reached_goal": False,
                                "goal": c_g,
                            }
                        )

            return D
        else:
            try:
                p_df = pd.read_csv(
                    p_file, usecols=["posX", "posY", "angle", "userControl"]
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find player data for level {lvl}, attempt {attempt}. Check that directory {path} exists!"
                )

            # adjusts negative angles to be within positive 0 to 360
            p_angles = p_df["angle"].to_numpy()
            p_angles = np.where(p_angles < 0, p_angles + 360.0, p_angles)
            p_df["angle"] = p_angles
            o_dfs = []
            for i in range(9, obs_end + 1):
                o_file = f"{path}/entity.{lvl}.{attempt}.{i}.data.csv"
                try:
                    o = pd.read_csv(o_file, usecols=["posX", "posY", "angle"])
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (obstacle) {i} for level {lvl}, attempt {attempt}!"
                    )

                o_dfs.append(o)

            g_file = f"{path}/goal-goal.{lvl}.{attempt}.agg.csv"
            try:
                g_df = pd.read_csv(g_file, usecols=["At", "From"])
            except:
                raise FileNotFoundError(
                    f"Could not find data for goal-goal for level {lvl}, attempt {attempt}!"
                )

            D = []
            if g_df.shape[0] > 0:
                for i in range(g_df.shape[0]):
                    if i < 6:
                        k = i + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    else:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )

                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[(g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == False].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[i, 1] - 1) : (g_df.iloc[i, 0] - 1),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": True,
                                    "goal": c_g,
                                }
                            )

                if g_df.shape[0] < 7:
                    if g_df.shape[0] == 6:
                        k = 1
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.1.data.csv"
                    else:
                        k = g_df.shape[0] + 2
                        c_g_file = f"{path}/entity.{lvl}.{attempt}.{k}.data.csv"
                    try:
                        c_g_df = pd.read_csv(
                            c_g_file, usecols=["posX", "posY"], nrows=1
                        )
                    except:
                        raise FileNotFoundError(
                            f"Could not find data for entity (goal) {k} for level {lvl}, attempt {attempt}!"
                        )
                    c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])

                    p_g_pos = p_df.iloc[
                        (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (p_df.shape[0] - 1),
                    ]
                    p_g_pos = p_g_pos.reset_index(drop=True)
                    h_id = p_g_pos[p_g_pos["userControl"] == False].index.to_list()
                    if h_id:
                        h_control = [
                            list(map(itemgetter(1), g))
                            for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                        ]
                        for k in h_control:
                            p_pos = p_g_pos.iloc[k[0] : (k[-1] + 1), 0:3]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos = p_pos[
                                (p_pos.index % (skip + 1) == 0)
                                | (p_pos.index == (p_pos.shape[0] - 1))
                            ]
                            p_pos = p_pos.reset_index(drop=True)
                            p_pos["t"] = p_pos.index

                            O_list = []
                            for o_id, o_df in enumerate(o_dfs):
                                O = o_df.iloc[
                                    (g_df.iloc[g_df.shape[0] - 1, 0] - 1) : (
                                        p_df.shape[0] - 1
                                    ),
                                ]
                                O = O.reset_index(drop=True)
                                O = O.iloc[k[0] : (k[-1] + 1),]
                                O = O.reset_index(drop=True)
                                O = O[
                                    (O.index % (skip + 1) == 0)
                                    | (O.index == (O.shape[0] - 1))
                                ]
                                O = O.reset_index(drop=True)
                                O["t"] = O.index
                                O["id"] = o_id
                                O_list.append(O)
                            o_pos = pd.concat(O_list)
                            o_pos.reset_index

                            D.append(
                                {
                                    "player": p_pos,
                                    "obstacles": o_pos,
                                    "reached_goal": False,
                                    "goal": c_g,
                                }
                            )

            else:
                c_g_file = f"{path}/entity.{lvl}.{attempt}.2.data.csv"
                try:
                    c_g_df = pd.read_csv(c_g_file, usecols=["posX", "posY"], nrows=1)
                except:
                    raise FileNotFoundError(
                        f"Could not find data for entity (goal) 2 for level {lvl}, attempt {attempt}!"
                    )

                c_g = (c_g_df["posX"].values[0], c_g_df["posY"].values[0])
                p_df = p_df.iloc[:-1,]
                h_id = p_df[p_df["userControl"] == False].index.to_list()
                if h_id:
                    h_control = [
                        list(map(itemgetter(1), g))
                        for k, g in groupby(enumerate(h_id), lambda x: x[0] - x[1])
                    ]
                    for k in h_control:
                        p_pos = p_df.iloc[k[0] : (k[-1] + 1), 0:3]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos = p_pos[
                            (p_pos.index % (skip + 1) == 0)
                            | (p_pos.index == (p_pos.shape[0] - 1))
                        ]
                        p_pos = p_pos.reset_index(drop=True)
                        p_pos["t"] = p_pos.index

                        O_list = []
                        for o_id, o_df in enumerate(o_dfs):
                            O = o_df.iloc[:-1,]
                            O = O.iloc[k[0] : (k[-1] + 1),]
                            O = O[
                                (O.index % (skip + 1) == 0)
                                | (O.index == (O.shape[0] - 1))
                            ]
                            O = O.reset_index(drop=True)
                            O["t"] = O.index
                            O["id"] = o_id
                            O_list.append(O)
                        o_pos = pd.concat(O_list)
                        o_pos.reset_index

                        D.append(
                            {
                                "player": p_pos,
                                "obstacles": o_pos,
                                "reached_goal": False,
                                "goal": c_g,
                            }
                        )

            return D
    return None


def load_lvl_data_p(lvl, path, skip=0, control=1, outer_call=True, cores=None):
    S = []
    a = 0
    while True:
        S.append((lvl, a, path, skip, control))
        a += 1
        p_file = f"{path}/player.{lvl}.{a}.0.data.csv"
        if not (os.path.isfile(os.path.expanduser(p_file))):
            break
    if outer_call:
        if cores is None:
            cores = os.cpu_count()
        with Pool(cores) as p:
            res = p.map(load_attempt_data_p, S)
            D = list(chain.from_iterable(res))
            return D
    return S


def load_test_data_p(path, skip=0, control=1, outer_call=True, cores=None):
    S = []
    for i in range(9, 12):
        s = load_lvl_data_p(i, path, skip, control, False, None)
        S += s
    if outer_call:
        if cores is None:
            cores = os.cpu_count()
        with Pool(cores) as p:
            res = p.map(load_attempt_data_p, S)
            D = list(chain.from_iterable(res))
            return D
    return S


# Path is a directory for a participant for a given experiment over several different days (e.g., Experiment_1T5)
def load_experiment_data_p(path, skip=0, control=1, outer_call=True, cores=None):
    S = []
    dir_path = os.path.expanduser(path)
    dir_list = os.scandir(dir_path)
    for i in dir_list:
        if i.is_dir():
            s = load_test_data_p(i.path, skip, control, False, None)
            S += s
    if outer_call:
        if cores is None:
            cores = os.cpu_count()
        with Pool(cores) as p:
            res = p.map(load_attempt_data_p, S)
            D = list(chain.from_iterable(res))
            return D
    return S


# p_id is a participant id (e.g., auto-1ba807eecf3cf284). This function will look for experiment data for that paricipant for a given path variable.
# It assumes that the format of the directories within the path are set-up as <experiment>/<participant_id>/<test days>/<data files>. There should be
# No other types of directories other than experiment folders or there could be unexplained behavior!
def load_participant_data_p(
    p_id, path="~/busy-beeway/data/game_data", skip=0, control=1, cores=None
):
    S = []
    dir_path = os.path.expanduser(path)
    dir_list = os.scandir(dir_path)
    for i in dir_list:
        if i.is_dir():
            e_path = f"{i.path}/{p_id}"
            s = load_experiment_data_p(e_path, skip, control, False, None)
            S += s
    if cores is None:
        cores = os.cpu_count()
    with Pool(cores) as p:
        res = p.map(load_attempt_data_p, S)
        D = list(chain.from_iterable(res))
        return D
