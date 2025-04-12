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
                o_X = np.where(g_o_dist > 50.0, -old_O["posX"].to_numpy(), o_X)
                o_Y = np.where(g_o_dist > 50.0, -old_O["posY"].to_numpy(), o_Y)
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
            "obstacle_count": p_df["obstacle_count"].to_numpy()[0],
            "sigma": p_df["sigma"].to_numpy()[0],
            "repel_factor": p_df["repel_factor"].to_numpy()[0],
            "attempt": p_df["attempt"].to_numpy()[0],
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


# Replay Run where controller moves randomly
def random_run_replay(
    d,
    move_stats,
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
            success = False
            break
        p_dist = rng.normal(move_stats[0], move_stats[1])
        r_dir = rng.uniform(0, 360)
        new_p_X = old_p_X + (p_dist * cos_plus(r_dir))
        new_p_Y = old_p_Y + (p_dist * sin_plus(r_dir))
        p_X_list.append(new_p_X)
        p_Y_list.append(new_p_Y)
        p_A_list.append(r_dir)
        O_list.append(new_O)

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
            "obstacle_count": p_df["obstacle_count"].to_numpy()[0],
            "sigma": p_df["sigma"].to_numpy()[0],
            "repel_factor": p_df["repel_factor"].to_numpy()[0],
            "attempt": p_df["attempt"].to_numpy()[0],
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


def random_replay(D, move_stats, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    dat = []
    for d in D:
        rd, rng = random_run_replay(d, move_stats, rng)
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
                o_X = np.where(g_o_dist > 50.0, -old_O["posX"].to_numpy(), o_X)
                o_Y = np.where(g_o_dist > 50.0, -old_O["posY"].to_numpy(), o_Y)
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
            "obstacle_count": p_df["obstacle_count"].to_numpy()[0],
            "sigma": p_df["sigma"].to_numpy()[0],
            "repel_factor": p_df["repel_factor"].to_numpy()[0],
            "attempt": p_df["attempt"].to_numpy()[0],
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


# Replay Run where controller moves randomly
def random_run_replay_p(f):
    d, move_stats, rng = f
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
            success = False
            break
        p_dist = rng.normal(move_stats[0], move_stats[1])
        r_dir = rng.uniform(0, 360)
        new_p_X = old_p_X + (p_dist * cos_plus(r_dir))
        new_p_Y = old_p_Y + (p_dist * sin_plus(r_dir))
        p_X_list.append(new_p_X)
        p_Y_list.append(new_p_Y)
        p_A_list.append(r_dir)
        O_list.append(new_O)
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
            "obstacle_count": p_df["obstacle_count"].to_numpy()[0],
            "sigma": p_df["sigma"].to_numpy()[0],
            "repel_factor": p_df["repel_factor"].to_numpy()[0],
            "attempt": p_df["attempt"].to_numpy()[0],
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


def random_replay_p(
    D,
    move_stats,
    seed=None,
    cores=None,
):
    if cores is None:
        cores = os.cpu_count()
    if seed is None:
        with Pool(cores) as p:
            dat = list(
                p.map(
                    random_run_replay_p,
                    [
                        (
                            d,
                            move_stats,
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
                    random_run_replay_p,
                    [
                        (
                            d,
                            move_stats,
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
    ax.set(xlim=[-55, 55], ylim=[-55, 55], xlabel="X", ylabel="Y")
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


# We assume segment is a an array of shape of (seq_length,num_state_feature)
def animate_segment(
    segment,
    aux_data=None,
    ts=None,
    interval=80,
    n_min_obstacles=6,
    labels=["reward", "Importance Weight"],
    filename="run.gif",
):
    if aux_data is None:
        fig, ax = plt.subplots()
        obs_offset = 2 + 3 * n_min_obstacles
        g = segment[-1, obs_offset : obs_offset + 2]
        o_x = segment[:, 2:obs_offset:3]
        o_y = segment[:, 3:obs_offset:3]

        scat1 = ax.scatter([], [], c="b", s=15, label="bee")
        scat2 = ax.scatter([], [], c="r", s=15, label="wasps")
        scat3 = ax.scatter([], [], c="g", s=65, label="goal")
        ax.set(
            xlim=[segment[0, 0] - 30, segment[0, 0] + 30],
            ylim=[segment[0, 1] - 30, segment[0, 1] + 30],
            xlabel="X",
            ylabel="Y",
        )
        ax.legend(loc="upper right")

        def update(frame):
            scat1.set_offsets(segment[frame, 0:2])

            o_data = np.stack([o_x[frame], o_y[frame]]).T
            scat2.set_offsets(o_data)

            scat3.set_offsets(g)

            return scat1, scat2, scat3

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=segment.shape[0], interval=interval
        )
        ani.save(filename=filename, writer="pillow")
    else:
        if not isinstance(aux_data, list):
            fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
            obs_offset = 2 + 3 * n_min_obstacles
            g = segment[-1, obs_offset : obs_offset + 2]
            o_x = segment[:, 2:obs_offset:3]
            o_y = segment[:, 3:obs_offset:3]

            scat1 = ax1.scatter([], [], c="b", s=15, label="bee")
            scat2 = ax1.scatter([], [], c="r", s=15, label="wasps")
            scat3 = ax1.scatter([], [], c="g", s=65, label="goal")
            ax1.set(
                xlim=[segment[0, 0] - 30, segment[0, 0] + 30],
                ylim=[segment[0, 1] - 30, segment[0, 1] + 30],
                xlabel="X",
                ylabel="Y",
            )
            if ts is None:
                ax2.set(
                    xlim=[0, aux_data.shape[0]],
                    ylim=[np.min(aux_data), np.max(aux_data)],
                    xlabel="Timesteps",
                    ylabel=labels[0],
                )
            else:
                ax2.set(
                    xlim=[ts[0], ts[-1]],
                    ylim=[np.min(aux_data), np.max(aux_data)],
                    xlabel="Timesteps",
                    ylabel=labels[0],
                )
            ax1.legend(loc="upper right")

            (p,) = ax2.plot([], [])

            def update(frame):
                scat1.set_offsets(segment[frame, 0:2])

                o_data = np.stack([o_x[frame], o_y[frame]]).T
                scat2.set_offsets(o_data)

                scat3.set_offsets(g)
                if ts is None:
                    p.set_data(np.arange(frame), aux_data[:frame])
                else:
                    p.set_data(ts[:frame], aux_data[:frame])

                return scat1, scat2, scat3, p

            ani = animation.FuncAnimation(
                fig=fig, func=update, frames=segment.shape[0], interval=interval
            )
            ani.save(filename=filename, writer="pillow")
        else:
            assert len(aux_data) == 2
            fig = plt.figure(figsize=(10, 5))
            plt.subplots_adjust(hspace=0.4)
            axs = fig.subplot_mosaic([["game", "aux_1"], ["game", "aux_2"]])
            obs_offset = 2 + 3 * n_min_obstacles
            g = segment[-1, obs_offset : obs_offset + 2]
            o_x = segment[:, 2:obs_offset:3]
            o_y = segment[:, 3:obs_offset:3]

            scat1 = axs["game"].scatter([], [], c="b", s=15, label="bee")
            scat2 = axs["game"].scatter([], [], c="r", s=15, label="wasps")
            scat3 = axs["game"].scatter([], [], c="g", s=65, label="goal")
            axs["game"].set(
                xlim=[segment[0, 0] - 30, segment[0, 0] + 30],
                ylim=[segment[0, 1] - 30, segment[0, 1] + 30],
                xlabel="X",
                ylabel="Y",
            )
            # axs["game"].legend(loc="upper right")
            if ts is None:
                axs["aux_1"].set(
                    xlim=[0, aux_data[0].shape[0]],
                    ylim=[np.min(aux_data[0]), np.max(aux_data[0])],
                    xlabel="Timestep",
                    ylabel=labels[0],
                )
            else:
                axs["aux_1"].set(
                    xlim=[ts[0], ts[-1]],
                    ylim=[np.min(aux_data[0]), np.max(aux_data[0])],
                    xlabel="Timestep",
                    ylabel=labels[0],
                )

            (p_1,) = axs["aux_1"].plot([], [])

            if ts is None:
                axs["aux_2"].set(
                    xlim=[0, aux_data[1].shape[0]],
                    ylim=[0, 1],
                    xlabel="Timestep",
                    ylabel=labels[1],
                )
            else:
                axs["aux_2"].set(
                    xlim=[ts[0], ts[-1]],
                    ylim=[0, 1],
                    xlabel="Timestep",
                    ylabel=labels[1],
                )                

            (p_2,) = axs["aux_2"].plot([], [])

            def update(frame):
                scat1.set_offsets(segment[frame, 0:2])

                o_data = np.stack([o_x[frame], o_y[frame]]).T
                scat2.set_offsets(o_data)

                scat3.set_offsets(g)
                if ts is None:
                    p_1.set_data(np.arange(frame), aux_data[0][:frame])

                    p_2.set_data(np.arange(frame), aux_data[1][:frame])
                else:
                    p_1.set_data(ts[:frame], aux_data[0][:frame])

                    p_2.set_data(ts[:frame], aux_data[1][:frame])                    
                return scat1, scat2, scat3, p_1, p_2

            ani = animation.FuncAnimation(
                fig=fig, func=update, frames=segment.shape[0], interval=interval
            )
            ani.save(filename=filename, writer="pillow")
