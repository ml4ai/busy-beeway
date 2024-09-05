import os
from multiprocessing import Pool
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Finds distance between a set of coordinates and a single coordinate. vecX and vecY are numpy arrays, px and py are scalars (floats/ints/etc.)
# Outputs numpy array
def point_dist(vecX, vecY, px, py):
    return np.sqrt(((vecX - px) ** 2) + ((vecY - py) ** 2)) * 1


def cos_plus(degrees):
    res = np.cos(degrees * (np.pi / 180.0))
    res = np.where(np.isclose(degrees, 90), 0.0, res)
    res = np.where(np.isclose(degrees, 270), 0.0, res)
    return res * 1


def sin_plus(degrees):
    res = np.sin(degrees * (np.pi / 180.0))
    res = np.where(np.isclose(degrees, 360), 0.0, res)
    res = np.where(np.isclose(degrees, 180), 0.0, res)
    return res * 1


def find_direction(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    degs = np.arctan2(y, x) * (180.0 / np.pi)
    degs = np.where(np.isclose(degs, 0.0), 360.0, degs)
    degs = np.where(degs < 0, degs + 360.0, degs)
    return degs * 1


# Made for testing sets ray pairs rather than just a single pair of rays
def rays_intersect(as_x, as_y, ad_x, ad_y, bs_x, bs_y, bd_x, bd_y):
    dx = bs_x - as_x
    dy = bs_y - as_y

    det = (bd_x * ad_y) - (bd_y * ad_x)

    u_num = (dy * bd_x) - (dx * bd_y)
    v_num = (dy * ad_x) - (dx * ad_y)

    u_num_det = u_num * det
    v_num_det = v_num * det

    dx_ad_x = dx / ad_x
    dy_ad_y = dy / ad_y

    normal_int = (
        ((u_num_det > 0) | np.isclose(u_num_det, 0))
        & ((v_num_det > 0) | np.isclose(v_num_det, 0))
        & (~np.isclose(det, 0))
    )
    coincide = np.isclose(u_num, 0) & np.isclose(v_num, 0) & np.isclose(det, 0)
    collide = ((dx_ad_x > 0) | np.isclose(dx_ad_x, 0)) & np.isclose(dx_ad_x, dy_ad_y)
    return normal_int | (coincide & collide)


def max_seq_length(F):
    if isinstance(F[0], dict):
        # Assumes F is a list of test session data dictionaries
        return max([f["player"].shape[0] for f in F])
    else:
        return max([f.shape[0] for f in F])


# sa should always be clockwise from player direction, ea should be counter-clockwise from player direction.
def points_in_arc(cx, cy, xs, ys, sa, ea, r=100.0):
    if np.isclose(sa, ea):
        return np.repeat(True, xs.shape[0])
    X = xs - cx
    Y = ys - cy
    pr = np.sqrt((X**2) + (Y**2))
    a = np.arctan2(Y, X) * (180.0 / np.pi)
    a = np.where(np.isclose(a, 0.0), 360.0, a)
    a = np.where(a < 0, a + 360, a)
    if ea > sa:
        return (
            ((pr < r) | np.isclose(pr, r))
            & ((a > sa) | np.isclose(a, sa))
            & ((a < ea) | np.isclose(a, ea))
        )
    if ea < sa:
        return ((pr < r) | np.isclose(pr, r)) & (
            ((a > sa) | np.isclose(a, sa)) | ((a < ea) | np.isclose(a, ea))
        )


# arc_sweep = (starting arc degree, ending arc degree, increase step for arc degree)
# An arc is defined centered at the players current direction. For each incremented size of the arc, obstacle-based features are computed for obstacles included in the arc.
def compute_run_features(p_df, g, O, arc_sweep=(10, 360, 10)):
    p_X = p_df["posX"].to_numpy()
    p_Y = p_df["posY"].to_numpy()
    p_A = p_df["angle"].to_numpy()
    features = {}
    # ===GOAL RELATED FEATURES===
    # 1. Players distance from goal
    goal_distances = point_dist(p_X, p_Y, g[0], g[1])

    goal_directions = find_direction(p_X, p_Y, g[0], g[1])
    # 2. Cosine similarity between players direction and goal direction
    goal_headings = cos_plus(goal_directions - p_A)

    features["goal_distances"] = goal_distances
    features["goal_headings"] = goal_headings
    for a in range(arc_sweep[0], arc_sweep[1] + 1, arc_sweep[2]):
        # ===MINIMUM DISTANCE OBSTACLE FEATURES===

        # 3. Minimum obstacle distance form player
        features[f"min_obstacle_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 4. Cosine Similarity between players direction and minimum distance obstacle
        features[f"min_distance_obstacle_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 5. cosine similarity between minimum distance obstacle direction and player
        features[f"min_distance_op_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # ===MAX COSINE SIMILARITY BETWEEN PLAYER AND OBSTACLES FEATURES===

        # 6. max cosine similarity between player direction and obstacles
        features[f"max_obstacle_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 7. distance of obstacle with max cosine similarity between player direction and obstacles
        features[f"max_heading_obstacle_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 8. Cosine Similarity of obstacle direction to player for the obstacle that has the max cosine similarity between the player and that obstacle.
        features[f"max_heading_obstacle_op_headings_{a}"] = np.repeat(
            0.0, p_df.shape[0]
        )

        # ===MAX COSINE SIMILARITY BETWEEN OBSTACLES AND PLAYER FEATURES===

        # 9. max cosine similarity between obstacle direction and player
        features[f"max_op_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 10. Distance of obstacle with max cosine similarity of its direction and player
        features[f"max_heading_op_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 11. Cosine Similarity of player direction to obstacle for the obstacle that has the max cosine similarity between the obstacle direction and player.
        features[f"max_heading_op_obstacle_headings_{a}"] = np.repeat(
            0.0, p_df.shape[0]
        )

        for index, row in p_df.iterrows():
            o_t = O[O["t"] == row["t"]]

            sa = row["angle"] - (a / 2.0)
            ea = row["angle"] + (a / 2.0)
            in_arc = points_in_arc(
                row["posX"],
                row["posY"],
                o_t["posX"].to_numpy(),
                o_t["posY"].to_numpy(),
                sa,
                ea,
            )
            if np.any(in_arc):
                o_t = o_t[in_arc]
                if o_t.shape[0] > 1:
                    o_t_X = o_t["posX"].to_numpy()
                    o_t_Y = o_t["posY"].to_numpy()
                    o_t_A = o_t["angle"].to_numpy()
                    obs_distances = point_dist(
                        o_t_X,
                        o_t_Y,
                        row["posX"],
                        row["posY"],
                    )
                    min_dist_obs = np.argmin(obs_distances)
                    features[f"min_obstacle_distances_{a}"][index] = obs_distances[
                        min_dist_obs
                    ]

                    obs_directions = find_direction(
                        row["posX"], row["posY"], o_t_X, o_t_Y
                    )
                    obs_headings = cos_plus(obs_directions - row["angle"]) + 2

                    features[f"min_distance_obstacle_headings_{a}"][index] = (
                        obs_headings[min_dist_obs]
                    )

                    max_heading_obs = np.argmax(obs_headings)
                    features[f"max_obstacle_headings_{a}"][index] = obs_headings[
                        max_heading_obs
                    ]

                    features[f"max_heading_obstacle_distances_{a}"][index] = (
                        obs_distances[max_heading_obs]
                    )

                    op_directions = find_direction(
                        o_t_X, o_t_Y, row["posX"], row["posY"]
                    )

                    op_headings = cos_plus(op_directions - o_t_A) + 2
                    max_heading_op = np.argmax(op_headings)
                    features[f"max_op_headings_{a}"][index] = op_headings[
                        max_heading_op
                    ]

                    features[f"min_distance_op_headings_{a}"][index] = op_headings[
                        min_dist_obs
                    ]

                    features[f"max_heading_obstacle_op_headings_{a}"][index] = (
                        op_headings[max_heading_obs]
                    )

                    features[f"max_heading_op_distances_{a}"][index] = obs_distances[
                        max_heading_op
                    ]

                    features[f"max_heading_op_obstacle_headings_{a}"][index] = (
                        obs_headings[max_heading_op]
                    )
                else:
                    o_t_X = o_t["posX"].to_numpy()
                    o_t_Y = o_t["posY"].to_numpy()
                    o_t_A = o_t["angle"].to_numpy()
                    obs_distances = point_dist(
                        o_t_X,
                        o_t_Y,
                        row["posX"],
                        row["posY"],
                    )

                    features[f"min_obstacle_distances_{a}"][index] = obs_distances[0]

                    obs_directions = find_direction(
                        row["posX"], row["posY"], o_t_X, o_t_Y
                    )
                    obs_headings = cos_plus(obs_directions - row["angle"]) + 2
                    features[f"min_distance_obstacle_headings_{a}"][index] = (
                        obs_headings[0]
                    )

                    features[f"max_obstacle_headings_{a}"][index] = obs_headings[0]

                    features[f"max_heading_obstacle_distances_{a}"][index] = (
                        obs_distances[0]
                    )

                    op_directions = find_direction(
                        o_t_X, o_t_Y, row["posX"], row["posY"]
                    )

                    op_headings = cos_plus(op_directions - o_t_A) + 2

                    features[f"max_op_headings_{a}"][index] = op_headings[0]

                    features[f"min_distance_op_headings_{a}"][index] = op_headings[0]

                    features[f"max_heading_obstacle_op_headings_{a}"][index] = (
                        op_headings[0]
                    )

                    features[f"max_heading_op_distances_{a}"][index] = obs_distances[0]

                    features[f"max_heading_op_obstacle_headings_{a}"][index] = (
                        obs_headings[0]
                    )
    features["userControl"] = p_df["userControl"].to_numpy().astype(int)
    features["t"] = p_df["t"].to_numpy()
    return pd.DataFrame(features)


# save_dir is a string containing the path to the directory where we want feature files saved.
def compute_features(D, arc_sweep=(10, 360, 10), save_dir=None):
    dat = []
    if save_dir is None:
        for d in D:
            p_df = d["player"]
            g = d["goal"]
            O = d["obstacles"]
            dat.append(compute_run_features(p_df, g, O, arc_sweep))
        return dat
    else:
        dir_path = os.path.expanduser(save_dir)
        try:
            os.mkdir("cache")
            try:
                os.mkdir(f"cache/{dir_path}")
                for i, d in enumerate(D):
                    p_df = d["player"]
                    g = d["goal"]
                    O = d["obstacles"]
                    res = compute_run_features(p_df, g, O, arc_sweep)
                    res.to_parquet(f"cache/{dir_path}/sequence_{i}.parquet")
                    dat.append(res)
            except FileExistsError:
                for i, d in enumerate(D):
                    p_df = d["player"]
                    g = d["goal"]
                    O = d["obstacles"]
                    res = compute_run_features(p_df, g, O, arc_sweep)
                    res.to_parquet(f"{dir_path}/sequence_{i}.parquet")
                    dat.append(res)
            return dat
        except FileExistsError:
            try:
                os.mkdir(f"cache/{dir_path}")
                for i, d in enumerate(D):
                    p_df = d["player"]
                    g = d["goal"]
                    O = d["obstacles"]
                    res = compute_run_features(p_df, g, O, arc_sweep)
                    res.to_parquet(f"cache/{dir_path}/sequence_{i}.parquet")
                    dat.append(res)
            except FileExistsError:
                for i, d in enumerate(D):
                    p_df = d["player"]
                    g = d["goal"]
                    O = d["obstacles"]
                    res = compute_run_features(p_df, g, O, arc_sweep)
                    res.to_parquet(f"cache/{dir_path}/sequence_{i}.parquet")
                    dat.append(res)
            return dat


def compute_run_features_p(d):
    p_df, g, O, arc_sweep, save_data = d
    p_X = p_df["posX"].to_numpy()
    p_Y = p_df["posY"].to_numpy()
    p_A = p_df["angle"].to_numpy()
    features = {}
    # ===GOAL RELATED FEATURES===
    # 1. Players distance from goal
    goal_distances = point_dist(p_X, p_Y, g[0], g[1])

    goal_directions = find_direction(p_X, p_Y, g[0], g[1])
    # 2. Cosine similarity between players direction and goal direction
    goal_headings = cos_plus(goal_directions - p_A)

    features["goal_distances"] = goal_distances
    features["goal_headings"] = goal_headings
    for a in range(arc_sweep[0], arc_sweep[1] + 1, arc_sweep[2]):
        # ===MINIMUM DISTANCE OBSTACLE FEATURES===

        # 3. Minimum obstacle distance form player
        features[f"min_obstacle_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 4. Cosine Similarity between players direction and minimum distance obstacle
        features[f"min_distance_obstacle_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 5. cosine similarity between minimum distance obstacle direction and player
        features[f"min_distance_op_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # ===MAX COSINE SIMILARITY BETWEEN PLAYER AND OBSTACLES FEATURES===

        # 6. max cosine similarity between player direction and obstacles
        features[f"max_obstacle_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 7. distance of obstacle with max cosine similarity between player direction and obstacles
        features[f"max_heading_obstacle_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 8. Cosine Similarity of obstacle direction to player for the obstacle that has the max cosine similarity between the player and that obstacle.
        features[f"max_heading_obstacle_op_headings_{a}"] = np.repeat(
            0.0, p_df.shape[0]
        )

        # ===MAX COSINE SIMILARITY BETWEEN OBSTACLES AND PLAYER FEATURES===

        # 9. max cosine similarity between obstacle direction and player
        features[f"max_op_headings_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 10. Distance of obstacle with max cosine similarity of its direction and player
        features[f"max_heading_op_distances_{a}"] = np.repeat(0.0, p_df.shape[0])

        # 11. Cosine Similarity of player direction to obstacle for the obstacle that has the max cosine similarity between the obstacle direction and player.
        features[f"max_heading_op_obstacle_headings_{a}"] = np.repeat(
            0.0, p_df.shape[0]
        )

        for index, row in p_df.iterrows():
            o_t = O[O["t"] == row["t"]]

            sa = row["angle"] - (a / 2.0)
            ea = row["angle"] + (a / 2.0)
            in_arc = points_in_arc(
                row["posX"],
                row["posY"],
                o_t["posX"].to_numpy(),
                o_t["posY"].to_numpy(),
                sa,
                ea,
            )
            if np.any(in_arc):
                o_t = o_t[in_arc]
                if o_t.shape[0] > 1:
                    o_t_X = o_t["posX"].to_numpy()
                    o_t_Y = o_t["posY"].to_numpy()
                    o_t_A = o_t["angle"].to_numpy()
                    obs_distances = point_dist(
                        o_t_X,
                        o_t_Y,
                        row["posX"],
                        row["posY"],
                    )
                    min_dist_obs = np.argmin(obs_distances)
                    features[f"min_obstacle_distances_{a}"][index] = obs_distances[
                        min_dist_obs
                    ]

                    obs_directions = find_direction(
                        row["posX"], row["posY"], o_t_X, o_t_Y
                    )
                    obs_headings = cos_plus(obs_directions - row["angle"]) + 2

                    features[f"min_distance_obstacle_headings_{a}"][index] = (
                        obs_headings[min_dist_obs]
                    )

                    max_heading_obs = np.argmax(obs_headings)
                    features[f"max_obstacle_headings_{a}"][index] = obs_headings[
                        max_heading_obs
                    ]

                    features[f"max_heading_obstacle_distances_{a}"][index] = (
                        obs_distances[max_heading_obs]
                    )

                    op_directions = find_direction(
                        o_t_X, o_t_Y, row["posX"], row["posY"]
                    )

                    op_headings = cos_plus(op_directions - o_t_A) + 2
                    max_heading_op = np.argmax(op_headings)
                    features[f"max_op_headings_{a}"][index] = op_headings[
                        max_heading_op
                    ]

                    features[f"min_distance_op_headings_{a}"][index] = op_headings[
                        min_dist_obs
                    ]

                    features[f"max_heading_obstacle_op_headings_{a}"][index] = (
                        op_headings[max_heading_obs]
                    )

                    features[f"max_heading_op_distances_{a}"][index] = obs_distances[
                        max_heading_op
                    ]

                    features[f"max_heading_op_obstacle_headings_{a}"][index] = (
                        obs_headings[max_heading_op]
                    )
                else:
                    o_t_X = o_t["posX"].to_numpy()
                    o_t_Y = o_t["posY"].to_numpy()
                    o_t_A = o_t["angle"].to_numpy()
                    obs_distances = point_dist(
                        o_t_X,
                        o_t_Y,
                        row["posX"],
                        row["posY"],
                    )

                    features[f"min_obstacle_distances_{a}"][index] = obs_distances[0]

                    obs_directions = find_direction(
                        row["posX"], row["posY"], o_t_X, o_t_Y
                    )
                    obs_headings = cos_plus(obs_directions - row["angle"]) + 2
                    features[f"min_distance_obstacle_headings_{a}"][index] = (
                        obs_headings[0]
                    )

                    features[f"max_obstacle_headings_{a}"][index] = obs_headings[0]

                    features[f"max_heading_obstacle_distances_{a}"][index] = (
                        obs_distances[0]
                    )

                    op_directions = find_direction(
                        o_t_X, o_t_Y, row["posX"], row["posY"]
                    )

                    op_headings = cos_plus(op_directions - o_t_A) + 2

                    features[f"max_op_headings_{a}"][index] = op_headings[0]

                    features[f"min_distance_op_headings_{a}"][index] = op_headings[0]

                    features[f"max_heading_obstacle_op_headings_{a}"][index] = (
                        op_headings[0]
                    )

                    features[f"max_heading_op_distances_{a}"][index] = obs_distances[0]

                    features[f"max_heading_op_obstacle_headings_{a}"][index] = (
                        obs_headings[0]
                    )
    features["userControl"] = p_df["userControl"].to_numpy().astype(int)
    features["t"] = p_df["t"].to_numpy()
    df = pd.DataFrame(features)
    if save_data:
        df.to_parquet(save_data)
    return df


def compute_features_p(D, arc_sweep=(10, 360, 10), save_dir=None, cores=None):
    if cores is None:
        cores = os.cpu_count()
    with Pool(cores) as p:
        if save_dir is None:
            dat = list(
                p.map(
                    compute_run_features_p,
                    [
                        (d["player"], d["goal"], d["obstacles"], arc_sweep, None)
                        for d in D
                    ],
                )
            )
        else:
            dir_path = os.path.expanduser(save_dir)
            try:
                os.mkdir("cache")
                try:
                    os.mkdir(f"cache/{dir_path}")
                    dat = list(
                        p.map(
                            compute_run_features_p,
                            [
                                (
                                    d["player"],
                                    d["goal"],
                                    d["obstacles"],
                                    arc_sweep,
                                    f"cache/{dir_path}/sequence_{i}.parquet",
                                )
                                for i, d in enumerate(D)
                            ],
                        )
                    )
                except FileExistsError:
                    dat = list(
                        p.map(
                            compute_run_features_p,
                            [
                                (
                                    d["player"],
                                    d["goal"],
                                    d["obstacles"],
                                    arc_sweep,
                                    f"cache/{dir_path}/sequence_{i}.parquet",
                                )
                                for i, d in enumerate(D)
                            ],
                        )
                    )
            except FileExistsError:
                try:
                    os.mkdir(f"cache/{dir_path}")
                    dat = list(
                        p.map(
                            compute_run_features_p,
                            [
                                (
                                    d["player"],
                                    d["goal"],
                                    d["obstacles"],
                                    arc_sweep,
                                    f"cache/{dir_path}/sequence_{i}.parquet",
                                )
                                for i, d in enumerate(D)
                            ],
                        )
                    )
                except FileExistsError:
                    dat = list(
                        p.map(
                            compute_run_features_p,
                            [
                                (
                                    d["player"],
                                    d["goal"],
                                    d["obstacles"],
                                    arc_sweep,
                                    f"cache/{dir_path}/sequence_{i}.parquet",
                                )
                                for i, d in enumerate(D)
                            ],
                        )
                    )
        return dat


# This will load in any parquet file in the directory. This loads files in order by
# index (number at end of filename) so that real feature trajectories match with their corresponding pair of synthetic
# feature trajectories.
def load_features_from_parquet(load_dir):
    dat = []
    dir_path = os.path.expanduser(load_dir)
    dir_list = os.scandir(dir_path)
    count = 0
    for i in dir_list:
        if i.is_file():
            if i.path.endswith(".parquet") and i.path.startswith(
                f"{dir_path}/sequence"
            ):
                count += 1

    if count:
        for c in range(count):
            dat.append(pd.read_parquet(f"{load_dir}/sequence_{c}.parquet"))
    return dat


# This pads feature dataframe up to the fill size with zeros. Returns original dataframe if fill_size <= dataframe.shape[0]
# Also fixes time index
def pad_run_features(f, fill_size=500):
    if fill_size > f.shape[0]:
        p_f = f.reindex(range(fill_size), fill_value=0)
        p_f["t"] = p_f.index
        return p_f
    return f


# If fill_size is none, then it computes the max seq length and uses that for the fill_size
def pad_features(F, fill_size=None):
    new_F = []
    if not fill_size:
        fill_size = max_seq_length(F)
    for f in F:
        new_F.append(pad_run_features(f, fill_size))
    return new_F


# Takes feature dataframe and transforms into jnp arrays with padding. Returns tuple of features matrix and timestep array for a state sequence
def run_to_jnp(f, fill_size=500, with_attn_mask=True):
    p_f = pad_run_features(f, fill_size)
    nf = p_f.to_numpy()
    if with_attn_mask:
        attn_mask = np.zeros(nf.shape[0])
        attn_mask[: f.shape[0]] = 1.0
        return (
            jnp.array(nf[:, 0:-1]),
            jnp.array(nf[:, -1], dtype=jnp.int32),
            jnp.array(attn_mask, dtype=jnp.float32),
        )
    return jnp.array(nf[:, 0:-1]), jnp.array(nf[:, -1], dtype=jnp.int32)


# Takes list of feature dataframes and transforms them into jnp arrays with padding. Returns a dictionary with array of "observations" and array of "timesteps"
def to_jnp(
    F,
    fill_size=None,
    labels=("observations", "timesteps", "attn_mask"),
    with_attn_mask=True,
):
    if not fill_size:
        fill_size = max_seq_length(F)

    if with_attn_mask:
        obs = []
        ts = []
        ams = []
        for f in F:
            o, t, am = run_to_jnp(f, fill_size, with_attn_mask)
            obs.append(o)
            ts.append(t)
            ams.append(am)
        return {
            labels[0]: jnp.stack(obs),
            labels[1]: jnp.stack(ts),
            labels[2]: jnp.stack(ams),
        }
    obs = []
    ts = []
    for f in F:
        o, t = run_to_jnp(f, fill_size, with_attn_mask)
        obs.append(o)
        ts.append(t)
    return {labels[0]: jnp.stack(obs), labels[1]: jnp.stack(ts)}


# Takes two lists of feature dataframes transforms them into jnp arrays with padding and matching preference label y.
# One list is assumed to be real data and the other is generated data based on the real data.
# Its assumed that F_1 is the generated data and F_2 is the real data.
# y = 1 for most pairs, but can be 0.5 if the real data mimics the same behavior as the generate data.
# _2 is appended to the dict labels for F_2
# This is specifically for real data matched with generated goal only trajectory data.
# save_data takes in a group path for a .hdf5 file
def create_preference_data(
    F_1,
    F_2,
    fill_size=None,
    labels=("observations", "timesteps", "attn_mask"),
    with_attn_mask=True,
    save_data=None,
):

    assert len(F_1) == len(F_2), "F_1 and F_2 should be equal sizes!"

    if not fill_size:
        fill_size = max(max_seq_length(F_1), max_seq_length(F_2))

    if with_attn_mask:
        obs = []
        ts = []
        ams = []
        obs_2 = []
        ts_2 = []
        ams_2 = []
        lbs = []
        for i, f in enumerate(F_1):
            o, t, am = run_to_jnp(f, fill_size, with_attn_mask)
            o_2, t_2, am_2 = run_to_jnp(F_2[i], fill_size, with_attn_mask)

            obs.append(o)
            ts.append(t)
            ams.append(am)

            obs_2.append(o_2)
            ts_2.append(t_2)
            ams_2.append(am_2)

            if np.all(np.isclose(F_2[i]["goal_headings"].to_numpy(), 1)):
                lbs.append(0.5)
            else:
                lbs.append(1.0)
        if save_data is None:
            return {
                labels[0]: jnp.stack(obs),
                labels[1]: jnp.stack(ts),
                labels[2]: jnp.stack(ams),
                f"{labels[0]}_2": jnp.stack(obs_2),
                f"{labels[1]}_2": jnp.stack(ts_2),
                f"{labels[2]}_2": jnp.stack(ams_2),
                "labels": jnp.array(lbs),
            }
        else:
            data = {
                labels[0]: jnp.stack(obs),
                labels[1]: jnp.stack(ts),
                labels[2]: jnp.stack(ams),
                f"{labels[0]}_2": jnp.stack(obs_2),
                f"{labels[1]}_2": jnp.stack(ts_2),
                f"{labels[2]}_2": jnp.stack(ams_2),
                "labels": jnp.array(lbs),
            }
            with h5py.File("preference_data.hdf5", "a") as f:
                if save_data in f:
                    # WARNING if this group already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
                    # will be overwritten with this new datasets.
                    g = f[save_data]
                    for k in data:
                        if k in g:
                            del g[k]
                            g[k] = data[k]
                        else:
                            g[k] = data[k]
                else:
                    g = f.create_group(save_data)
                    for k in data:
                        g[k] = data[k]
            return data
        obs = []
        ts = []
        obs_2 = []
        ts_2 = []
        lbs = []
        for i, f in enumerate(F_1):
            o, t = run_to_jnp(f, fill_size, with_attn_mask)
            o_2, t_2 = run_to_jnp(F_2[i], fill_size, with_attn_mask)

            obs.append(o)
            ts.append(t)

            obs_2.append(o_2)
            ts_2.append(t_2)

            if np.all(np.isclose(F_2[i]["goal_headings"].to_numpy(), 1)):
                lbs.append(0.5)
            else:
                lbs.append(1.0)

        if save_data is None:
            return {
                labels[0]: jnp.stack(obs),
                labels[1]: jnp.stack(ts),
                f"{labels[0]}_2": jnp.stack(obs_2),
                f"{labels[1]}_2": jnp.stack(ts_2),
                "labels": jnp.array(lbs),
            }
        else:
            data = {
                labels[0]: jnp.stack(obs),
                labels[1]: jnp.stack(ts),
                f"{labels[0]}_2": jnp.stack(obs_2),
                f"{labels[1]}_2": jnp.stack(ts_2),
                "labels": jnp.array(lbs),
            }
            with h5py.File("preference_data.hdf5", "a") as f:
                if save_data in f:
                    # WARNING if this group already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
                    # will be overwritten with this new datasets.
                    g = f[save_data]
                    for k in data:
                        if k in g:
                            del g[k]
                            g[k] = data[k]
                        else:
                            g[k] = data[k]
                else:
                    g = f.create_group(save_data)
                    for k in data:
                        g[k] = data[k]
            return data


def plot_training_validation_loss(load_log, eval_period=5, save_file=None, **kwargs):
    L = pd.read_csv(load_log)
    L = L[(L.index % eval_period == 0) & (L.index >= eval_period)]

    x = L.index.to_numpy()
    y = L["training_loss"].to_numpy()

    y2 = L["eval_loss"].to_numpy()

    xb = L["best_epoch"].to_numpy()[-1]
    yb = L["eval_loss_best"].to_numpy()[-1]

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Training Loss")
    ax.plot(x, y2, label="Validation Loss")
    ax.plot(
        xb,
        yb,
        "*",
        markersize=kwargs.get("markersize", 10),
        label="Best Validation Loss",
    )
    if "ylim" in kwargs:
        ax.set(
            ylim=kwargs["ylim"],
            xlabel="Epoch",
            ylabel="Preference Predictor Cross-Entropy Loss",
            title="Preference Transformer: Training vs. Validation Loss",
        )
    else:
        ax.set(
            xlabel="Epoch",
            ylabel="Preference Predictor Cross-Entropy Loss",
            title="Preference Transformer: Training vs. Validation Loss",
        )
    ax.legend()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
