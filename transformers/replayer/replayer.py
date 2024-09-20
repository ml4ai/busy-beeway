import itertools
import os
from multiprocessing import Pool

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformers.replayer.replayer_utils import (
    collision,
    cos_plus,
    find_direction,
    sin_plus,
)


def generate_stats(D, save_data=None):
    p_dists = []
    o_dists = []
    for d in D:
        p_df = d["player"]
        O = d["obstacles"]
        p_dist = np.sqrt(
            (np.diff(p_df["posX"].to_numpy()) ** 2)
            + (np.diff(p_df["posY"].to_numpy()) ** 2)
        )
        p_dists.append(p_dist)
        ids = np.unique(O["id"])
        for i in ids:
            o_i = O[O["id"] == i]
            o_i_dist = np.sqrt(
                (np.diff(o_i["posX"].to_numpy()) ** 2)
                + (np.diff(o_i["posY"].to_numpy()) ** 2)
            )
            o_dists.append(np.extract(o_i_dist <= 1, o_i_dist))
    if save_data is None:
        return (
            np.mean(np.concatenate(p_dists)),
            np.std(np.concatenate(p_dists), ddof=1),
            np.mean(np.concatenate(o_dists)),
            np.std(np.concatenate(o_dists), ddof=1),
        )
    else:
        save_data = os.path.expanduser(save_data)
        Path(save_data).mkdir(parents=True, exist_ok=True)
        stats = (
            np.mean(np.concatenate(p_dists)),
            np.std(np.concatenate(p_dists), ddof=1),
            np.mean(np.concatenate(o_dists)),
            np.std(np.concatenate(o_dists), ddof=1),
        )
        np.save(save_data, stats, False, False)
        return stats


def load_stats(load_file):
    load_file = os.path.expanduser(load_file)
    return tuple(np.load(load_file, fix_imports=False))


# Replay Run where controller only moves to points closest to goal
def goal_only_run_replay(
    d,
    move_stats,
    simulate_forward=True,
    ignore_collisions=False,
    rng=np.random.default_rng(),
):
    p_df = d["player"]
    O = d["obstacles"]
    g = d["goal"]

    old_p_X = p_df["posX"].to_numpy()[0]
    old_p_Y = p_df["posY"].to_numpy()[0]
    old_O = O[O["t"] == p_df["t"].to_numpy()[0]]
    p_X_list = [old_p_X]
    p_Y_list = [old_p_Y]
    p_A_list = [p_df["angle"].to_numpy()[0]]
    O_list = [old_O]
    success = True
    for t in itertools.count(start=1):
        new_O = O[O["t"] == t]
        if new_O.shape[0] == 0:
            if simulate_forward:
                o_dists = rng.normal(move_stats[2], move_stats[3], old_O.shape[0])
                o_X = old_O["posX"].to_numpy() + (
                    o_dists * cos_plus(old_O["angle"].to_numpy())
                )
                o_Y = old_O["posY"].to_numpy() + (
                    o_dists * sin_plus(old_O["angle"].to_numpy())
                )
                g_o_dist = np.sqrt((o_X**2) + (o_Y**2))
                o_X = np.where(g_o_dist > 50.0, -o_X, o_X)
                o_Y = np.where(g_o_dist > 50.0, -o_Y, o_Y)
                new_O = pd.DataFrame(
                    {
                        "posX": o_X,
                        "posY": o_Y,
                        "angle": old_O["angle"].to_numpy(),
                        "t": t,
                        "id": old_O["id"].to_numpy(),
                    }
                )
            else:
                success = False
                break
        p_dist = rng.normal(move_stats[0], move_stats[1])
        g_dir = find_direction(old_p_X, old_p_Y, g[0], g[1])
        new_p_X = old_p_X + (p_dist * cos_plus(g_dir))
        new_p_Y = old_p_Y + (p_dist * sin_plus(g_dir))
        p_X_list.append(new_p_X)
        p_Y_list.append(new_p_Y)
        p_A_list.append(g_dir)
        O_list.append(new_O)
        if not ignore_collisions:
            coll, _, _ = collision(
                old_O["posX"].to_numpy(),
                old_O["posY"].to_numpy(),
                new_O["posX"].to_numpy(),
                new_O["posY"].to_numpy(),
                old_p_X,
                old_p_Y,
            )

            if coll:
                success = False
                break

            coll, _, _ = collision(
                old_p_X,
                old_p_Y,
                new_p_X,
                new_p_Y,
                new_O["posX"].to_numpy(),
                new_O["posY"].to_numpy(),
            )

            if coll:
                success = False
                break
        coll, _, _ = collision(
            old_p_X, old_p_Y, new_p_X, new_p_Y, g[0], g[1], radius_2=1.0
        )

        if coll:
            break
        old_p_X = new_p_X
        old_p_Y = new_p_Y
        old_O = new_O

    new_p_df = pd.DataFrame(
        {
            "posX": p_X_list,
            "posY": p_Y_list,
            "angle": p_A_list,
            "obstacle_count": p_df["obstacle_count"].to_numpy(),
            "sigma": 0,
            "repel_factor": 0,
            "attempt": p_df["attempt"].to_numpy(),
            "userControl": 0,
            "t": range(len(p_X_list)),
        }
    )
    o_pos = pd.concat(O_list)
    o_pos.reset_index
    return {
        "player": new_p_df,
        "obstacles": o_pos,
        "reached_goal": success,
        "goal": g,
    }, rng


def goal_only_replay(
    D, move_stats, simulate_forward=True, ignore_collisions=False, seed=None
):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    dat = []
    for d in D:
        rd, rng = goal_only_run_replay(
            d, move_stats, simulate_forward, ignore_collisions, rng
        )
        dat.append(rd)
    return dat


# Replay Run where controller only moves to points closest to goal
def goal_only_run_replay_p(f):
    d, move_stats, simulate_forward, ignore_collisions, rng = f
    p_df = d["player"]
    O = d["obstacles"]
    g = d["goal"]

    old_p_X = p_df["posX"].to_numpy()[0]
    old_p_Y = p_df["posY"].to_numpy()[0]
    old_O = O[O["t"] == p_df["t"].to_numpy()[0]]
    p_X_list = [old_p_X]
    p_Y_list = [old_p_Y]
    p_A_list = [p_df["angle"].to_numpy()[0]]
    O_list = [old_O]
    success = True
    for t in itertools.count(start=1):
        new_O = O[O["t"] == t]
        if new_O.shape[0] == 0:
            if simulate_forward:
                o_dists = rng.normal(move_stats[2], move_stats[3], old_O.shape[0])
                o_X = old_O["posX"].to_numpy() + (
                    o_dists * cos_plus(old_O["angle"].to_numpy())
                )
                o_Y = old_O["posY"].to_numpy() + (
                    o_dists * sin_plus(old_O["angle"].to_numpy())
                )
                g_o_dist = np.sqrt((o_X**2) + (o_Y**2))
                o_X = np.where(g_o_dist > 50.0, -o_X, o_X)
                o_Y = np.where(g_o_dist > 50.0, -o_Y, o_Y)
                new_O = pd.DataFrame(
                    {
                        "posX": o_X,
                        "posY": o_Y,
                        "angle": old_O["angle"].to_numpy(),
                        "t": t,
                        "id": old_O["id"].to_numpy(),
                    }
                )
            else:
                success = False
                break
        p_dist = rng.normal(move_stats[0], move_stats[1])
        g_dir = find_direction(old_p_X, old_p_Y, g[0], g[1])
        new_p_X = old_p_X + (p_dist * cos_plus(g_dir))
        new_p_Y = old_p_Y + (p_dist * sin_plus(g_dir))
        p_X_list.append(new_p_X)
        p_Y_list.append(new_p_Y)
        p_A_list.append(g_dir)
        O_list.append(new_O)
        if not ignore_collisions:
            coll, _, _ = collision(
                old_O["posX"].to_numpy(),
                old_O["posY"].to_numpy(),
                new_O["posX"].to_numpy(),
                new_O["posY"].to_numpy(),
                old_p_X,
                old_p_Y,
            )

            if coll:
                success = False
                break

            coll, _, _ = collision(
                old_p_X,
                old_p_Y,
                new_p_X,
                new_p_Y,
                new_O["posX"].to_numpy(),
                new_O["posY"].to_numpy(),
            )

            if coll:
                success = False
                break
        coll, _, _ = collision(
            old_p_X, old_p_Y, new_p_X, new_p_Y, g[0], g[1], radius_2=1.0
        )

        if coll:
            break
        old_p_X = new_p_X
        old_p_Y = new_p_Y
        old_O = new_O

    new_p_df = pd.DataFrame(
        {
            "posX": p_X_list,
            "posY": p_Y_list,
            "angle": p_A_list,
            "obstacle_count": p_df["obstacle_count"].to_numpy(),
            "sigma": 0,
            "repel_factor": 0,
            "attempt": p_df["attempt"].to_numpy(),
            "userControl": 0,
            "t": range(len(p_X_list)),
        }
    )
    o_pos = pd.concat(O_list)
    o_pos.reset_index
    return {
        "player": new_p_df,
        "obstacles": o_pos,
        "reached_goal": success,
        "goal": g,
    }


def goal_only_replay_p(
    D,
    move_stats,
    simulate_forward=True,
    ignore_collisions=False,
    seed=None,
    cores=None,
):
    if cores is None:
        cores = os.cpu_count()
    if seed is None:
        with Pool(cores) as p:
            dat = list(
                p.map(
                    goal_only_run_replay_p,
                    [
                        (
                            d,
                            move_stats,
                            simulate_forward,
                            ignore_collisions,
                            np.random.default_rng(),
                        )
                        for d in D
                    ],
                )
            )
            return dat
    else:
        with Pool(cores) as p:
            dat = list(
                p.map(
                    goal_only_run_replay_p,
                    [
                        (
                            d,
                            move_stats,
                            simulate_forward,
                            ignore_collisions,
                            np.random.default_rng(seed + i * 1000),
                        )
                        for i, d in enumerate(D)
                    ],
                )
            )
            return dat


def animate_run(d, interval=80, filename="run.gif"):
    p_df = d["player"]
    O = d["obstacles"]
    g = d["goal"]
    p_X = p_df["posX"].to_numpy()
    p_Y = p_df["posY"].to_numpy()
    p_t = p_df["t"].to_numpy()
    fig, ax = plt.subplots()

    scat1 = ax.scatter([], [], c="b", s=5, label="bee")
    scat2 = ax.scatter([], [], c="r", s=5, label="wasps")
    scat3 = ax.scatter([], [], c="g", s=5, label="goal")
    ax.set(xlim=[-30, 30], ylim=[-30, 30], xlabel="X", ylabel="Y")
    ax.legend(loc="upper right")

    def update(frame):
        p_data = np.stack([p_X[frame], p_Y[frame]]).T
        scat1.set_offsets(p_data)

        o_t = O[O["t"] == p_t[frame]]
        o_data = np.stack([o_t["posX"].to_numpy(), o_t["posY"].to_numpy()]).T
        scat2.set_offsets(o_data)

        g_data = np.stack([g[0], g[1]]).T
        scat3.set_offsets(g_data)

        return scat1, scat2, scat3

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=p_t.shape[0], interval=interval
    )
    ani.save(filename=filename, writer="pillow")
