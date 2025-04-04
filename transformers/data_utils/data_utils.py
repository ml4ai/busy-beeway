import os
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


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

def first_nth_argmins(arr, n):
    """
    Returns the indices of the 0 to nth minimum values in a NumPy array.

    Parameters:
    arr (numpy.ndarray): The input NumPy array.
    n (int): The number of minimum values to consider (inclusive of 0th minimum).

    Returns:
    numpy.ndarray: An array containing the indices of the 0 to nth minimum values.
                   Returns an empty array if n is negative or greater than or equal to the array size.
    """
    if n < 0 or n > arr.size:
        return np.array([])
    
    indices = np.argpartition(arr, np.arange(n))[:n]
    return indices

# n_min_obstacles is number of obstacles to include in features per state starting from closest obstacle to the n_min_obstacles-th closest obstacle
def compute_run_features(p_df, g, O, day=None, n_min_obstacles=6):
    features = {}
    features["posX"] = p_df["posX"].to_numpy()
    features["posY"] = p_df["posY"].to_numpy()
    features["angle"] = p_df["angle"].to_numpy()
    feature["velocityX"] = p_df["velocityX"].to_numpy()
    feature["velocityY"] = p_df["velocityY"].to_numpy()

    for i in range(n_min_obstacles):

        features[f"O_{i}_posX"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_posY"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_angle"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_velocityX"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_veloctiyX"] = np.repeat(0.0, p_df.shape[0])

    for index, row in p_df.iterrows():
        o_t = O[O["t"] == row["t"]]
        o_t_X = o_t["posX"].to_numpy()
        o_t_Y = o_t["posY"].to_numpy()
        o_t_A = o_t["angle"].to_numpy()
        o_t_vX = o_t["velocityX"].to_numpy()
        o_t_vY = o_t["velocityY"].to_numpy()
        obs_distances = point_dist(
            o_t_X,
            o_t_Y,
            row["posX"],
            row["posY"],
        )
        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)
        for i in range(n_min_obstacles):
            features[f"O_{i}_posX"][index] = o_t_X[min_dist_obs[i]]
    
            features[f"O_{i}_posY"][index] = o_t_Y[min_dist_obs[i]]
    
            features[f"O_{i}_angle"][index] = o_t_A[min_dist_obs[i]]
    
            features[f"O_{i}_velocityX"][index] = o_t_vX[min_dist_obs[i]]
    
            features[f"O_{i}_veloctiyX"][index] = o_t_vY[min_dist_obs[i]]
    features["goalX"] = np.repeat(g[0], p_df.shape[0])
    features["goalY"] = np.repeat(g[1], p_df.shape[0])
    features["level"] = p_df["level"].to_numpy()
    features["ai"] = p_df["ai"].to_numpy()
    features["attempt"] = p_df["attempt"].to_numpy()
    if day is not None:
        features["day"] = np.repeat(day, p_df.shape[0])
    features["controlX"] = p_df["controlX"].to_numpy()
    features["controlY"] = p_df["controlY"].to_numpy()
    features["t"] = p_df["t"].to_numpy()
    return pd.DataFrame(features)


# save_dir is a string containing the path to the directory where we want feature files saved.
def compute_features(D, day=None, n_min_obstacles=6,save_dir=None):
    dat = []
    if save_dir is None:
        for d in D:
            p_df = d["player"]
            g = d["goal"]
            O = d["obstacles"]
            dat.append(compute_run_features(p_df, g, O, day,n_min_obstacles))
        return dat
    else:
        dir_path = os.path.expanduser(save_dir)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(D):
            p_df = d["player"]
            g = d["goal"]
            O = d["obstacles"]
            res = compute_run_features(p_df, g, O, day,n_min_obstacles)
            res.to_parquet(f"{dir_path}/sequence_{i}.parquet")
            dat.append(res)
        return dat


def compute_run_features_p(d):
    p_df, g, O, day, n_min_obstacles,save_data = d
    features = {}
    features["posX"] = p_df["posX"].to_numpy()
    features["posY"] = p_df["posY"].to_numpy()
    features["angle"] = p_df["angle"].to_numpy()
    feature["velocityX"] = p_df["velocityX"].to_numpy()
    feature["velocityY"] = p_df["velocityY"].to_numpy()

    for i in range(n_min_obstacles):

        features[f"O_{i}_posX"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_posY"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_angle"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_velocityX"] = np.repeat(0.0, p_df.shape[0])

        features[f"O_{i}_veloctiyX"] = np.repeat(0.0, p_df.shape[0])

    for index, row in p_df.iterrows():
        o_t = O[O["t"] == row["t"]]
        o_t_X = o_t["posX"].to_numpy()
        o_t_Y = o_t["posY"].to_numpy()
        o_t_A = o_t["angle"].to_numpy()
        o_t_vX = o_t["velocityX"].to_numpy()
        o_t_vY = o_t["velocityY"].to_numpy()
        obs_distances = point_dist(
            o_t_X,
            o_t_Y,
            row["posX"],
            row["posY"],
        )
        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)
        for i in range(n_min_obstacles):
            features[f"O_{i}_posX"][index] = o_t_X[min_dist_obs[i]]
    
            features[f"O_{i}_posY"][index] = o_t_Y[min_dist_obs[i]]
    
            features[f"O_{i}_angle"][index] = o_t_A[min_dist_obs[i]]
    
            features[f"O_{i}_velocityX"][index] = o_t_vX[min_dist_obs[i]]
    
            features[f"O_{i}_veloctiyX"][index] = o_t_vY[min_dist_obs[i]]
            
    features["goalX"] = np.repeat(g[0], p_df.shape[0])
    features["goalY"] = np.repeat(g[1], p_df.shape[0])
    features["level"] = p_df["level"].to_numpy()
    features["ai"] = p_df["ai"].to_numpy()
    features["attempt"] = p_df["attempt"].to_numpy()
    if day is not None:
        features["day"] = np.repeat(day, p_df.shape[0])
    features["controlX"] = p_df["controlX"].to_numpy()
    features["controlY"] = p_df["controlY"].to_numpy()
    features["t"] = p_df["t"].to_numpy()
    df = pd.DataFrame(features)
    if save_data:
        df.to_parquet(save_data)
    return df


def compute_features_p(D, day=None, n_min_obstacles=6,save_dir=None, cores=None):
    if cores is None:
        cores = os.cpu_count()
    with Pool(cores) as p:
        if save_dir is None:
            dat = list(
                p.map(
                    compute_run_features_p,
                    [(d["player"], d["goal"], d["obstacles"], day, n_min_obstacles,None) for d in D],
                )
            )
        else:
            dir_path = os.path.expanduser(save_dir)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            dat = list(
                p.map(
                    compute_run_features_p,
                    [
                        (
                            d["player"],
                            d["goal"],
                            d["obstacles"],
                            day,
                            n_min_obstacles,
                            f"{dir_path}/sequence_{i}.parquet",
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


# Takes feature dataframe and transforms into np arrays with padding. Returns tuple of features matrix and timestep array for a state sequence
def run_to_np(f, state_features=41, fill_size=500, with_attn_mask=True):
    p_f = pad_run_features(f, fill_size)
    nf = p_f.to_numpy()
    if with_attn_mask:
        attn_mask = np.zeros(nf.shape[0])
        attn_mask[: f.shape[0]] = 1.0
        return (
            np.array(nf[:, 0:state_features]),
            np.array(nf[:, state_features:-1]),
            np.array(nf[:, -1], dtype=np.int32),
            np.array(attn_mask, dtype=np.float32),
        )
    return (
        np.array(nf[:, 0:state_features]),
        np.array(nf[:, state_features:-1]),
        np.array(nf[:, -1], dtype=np.int32),
    )


def get_pref_labels(o, gh_idx=1):
    x = o.shape[0]
    labels = np.ones(x)
    for i in range(x):
        if np.all(np.isclose(o[i, :, gh_idx], 1)):
            labels[i] = 0.5
    return labels


# Takes two lists of feature dataframes transforms them into np arrays with padding and matching preference label y.
# One list is assumed to be real data and the other is generated data based on the real data.
# Its assumed that F_1 is the generated data and F_2 is the real data.
# y = 1 for most pairs, but can be 0.5 if the real data mimics the same behavior as the generate data.
# _2 is appended to the dict labels for F_2
# This is specifically for real data matched with generated goal only trajectory data.
# save_data takes in a group path for a .hdf5 file
# state_features is number of state features where the rest are actions and timesteps
def create_preference_data(
    F_1,
    F_2,
    split_size=100,
    gh_idx=1,
    state_features=41,
    labels=("states", "actions", "timesteps", "attn_mask"),
    with_attn_mask=True,
    save_data=None,
):

    assert len(F_1) == len(F_2), "F_1 and F_2 should be equal sizes!"

    if with_attn_mask:
        sts = []
        acts = []
        ts = []
        ams = []
        sts_2 = []
        acts_2 = []
        ts_2 = []
        ams_2 = []
        lbs = []
        for i, f in enumerate(F_1):
            fill_size = F_2[i].shape[0] + (split_size - (F_2[i].shape[0] % split_size))
            n_splits = int(fill_size / split_size)
            s, a, t, am = run_to_np(f, state_features, fill_size, with_attn_mask)
            s = s.reshape((n_splits, split_size, s.shape[1]))
            a = a.reshape((n_splits, split_size, a.shape[1]))
            t = t.reshape((n_splits, split_size))
            am = am.reshape((n_splits, split_size))

            s_2, a_2, t_2, am_2 = run_to_np(
                F_2[i], state_features, fill_size, with_attn_mask
            )
            s_2 = s_2.reshape((n_splits, split_size, s_2.shape[1]))
            a_2 = a_2.reshape((n_splits, split_size, a_2.shape[1]))
            t_2 = t_2.reshape((n_splits, split_size))
            am_2 = am_2.reshape((n_splits, split_size))

            sts.append(s)
            acts.append(a)
            ts.append(t)
            ams.append(am)

            sts_2.append(s_2)
            acts_2.append(a_2)
            ts_2.append(t_2)
            ams_2.append(am_2)

            lbs.append(get_pref_labels(s_2, gh_idx))
        if save_data is None:
            return {
                labels[0]: np.concatenate(sts),
                labels[1]: np.concatenate(acts),
                labels[2]: np.concatenate(ts),
                labels[3]: np.concatenate(ams),
                f"{labels[0]}_2": np.concatenate(sts_2),
                f"{labels[1]}_2": np.concatenate(acts_2),
                f"{labels[2]}_2": np.concatenate(ts_2),
                f"{labels[3]}_2": np.concatenate(ams_2),
                "labels": np.concatenate(lbs),
            }
        else:
            data = {
                labels[0]: np.concatenate(sts),
                labels[1]: np.concatenate(acts),
                labels[2]: np.concatenate(ts),
                labels[3]: np.concatenate(ams),
                f"{labels[0]}_2": np.concatenate(sts_2),
                f"{labels[1]}_2": np.concatenate(acts_2),
                f"{labels[2]}_2": np.concatenate(ts_2),
                f"{labels[3]}_2": np.concatenate(ams_2),
                "labels": np.concatenate(lbs),
            }
            with h5py.File(save_data, "a") as f:
                # WARNING if this file already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
                # will be overwritten with new datasets.
                for k in data:
                    if k in f:
                        del f[k]
                        f.create_dataset(k, data=data[k], chunks=True)
                    else:
                        f.create_dataset(k, data=data[k], chunks=True)
            return data
    sts = []
    acts = []
    ts = []
    sts_2 = []
    acts_2 = []
    ts_2 = []
    lbs = []
    for i, f in enumerate(F_1):
        fill_size = F_2[i].shape[0] + (split_size - (F_2[i].shape[0] % split_size))
        n_splits = int(fill_size / split_size)
        s, a, t = run_to_np(f, state_features, fill_size, with_attn_mask)
        s = s.reshape((n_splits, split_size, s.shape[1]))
        a = a.reshape((n_splits, split_size, a.shape[1]))
        t = t.reshape((n_splits, split_size))

        s_2, a_2, t_2 = run_to_np(F_2[i], state_features, fill_size, with_attn_mask)
        s_2 = s_2.reshape((n_splits, split_size, s_2.shape[1]))
        a_2 = a_2.reshape((n_splits, split_size, a_2.shape[1]))
        t_2 = t_2.reshape((n_splits, split_size))

        sts.append(s)
        acts.append(a)
        ts.append(t)

        sts_2.append(s_2)
        acts_2.append(a_2)
        ts_2.append(t_2)

        lbs.append(get_pref_labels(s_2, gh_idx))

    if save_data is None:
        return {
            labels[0]: np.concatenate(sts),
            labels[1]: np.concatenate(acts),
            labels[2]: np.concatenate(ts),
            f"{labels[0]}_2": np.concatenate(sts_2),
            f"{labels[1]}_2": np.concatenate(acts_2),
            f"{labels[2]}_2": np.concatenate(ts_2),
            "labels": np.concatenate(lbs),
        }
    else:
        data = {
            labels[0]: np.concatenate(sts),
            labels[1]: np.concatenate(acts),
            labels[2]: np.concatenate(ts),
            f"{labels[0]}_2": np.concatenate(sts_2),
            f"{labels[1]}_2": np.concatenate(acts_2),
            f"{labels[2]}_2": np.concatenate(ts_2),
            "labels": np.concatenate(lbs),
        }
        with h5py.File(save_data, "a") as f:
            # WARNING if this group already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
            # will be overwritten with this new datasets.
            for k in data:
                if k in f:
                    del f[k]
                    f.create_dataset(k, data=data[k], chunks=True)
                else:
                    f.create_dataset(k, data=data[k], chunks=True)
        return data


def create_state_data(
    F,
    split_size=100,
    state_features=41,
    labels=("states", "actions", "timesteps", "attn_mask"),
    with_attn_mask=True,
    save_data=None,
):

    if with_attn_mask:
        sts = []
        acts = []
        ts = []
        ams = []
        lbs = []
        for f in F:
            fill_size = f.shape[0] + (split_size - (f.shape[0] % split_size))
            n_splits = int(fill_size / split_size)
            s, a, t, am = run_to_np(f, state_features, fill_size, with_attn_mask)
            s = s.reshape((n_splits, split_size, s.shape[1]))
            a = a.reshape((n_splits, split_size, a.shape[1]))
            t = t.reshape((n_splits, split_size))
            am = am.reshape((n_splits, split_size))

            sts.append(s)
            acts.append(a)
            ts.append(t)
            ams.append(am)

            lbs.append(np.ones(s.shape[0]))
        if save_data is None:
            return {
                labels[0]: np.concatenate(sts),
                labels[1]: np.concatenate(acts),
                labels[2]: np.concatenate(ts),
                labels[3]: np.concatenate(ams),
                "labels": np.concatenate(lbs),
            }
        else:
            data = {
                labels[0]: np.concatenate(sts),
                labels[1]: np.concatenate(acts),
                labels[2]: np.concatenate(ts),
                labels[3]: np.concatenate(ams),
                "labels": np.concatenate(lbs),
            }
            with h5py.File(save_data, "a") as f:
                # WARNING if this file already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
                # will be overwritten with new datasets.
                for k in data:
                    if k in f:
                        del f[k]
                        f.create_dataset(k, data=data[k], chunks=True)
                    else:
                        f.create_dataset(k, data=data[k], chunks=True)
            return data
    sts = []
    acts = []
    ts = []
    lbs = []
    for f in F:
        fill_size = f.shape[0] + (split_size - (f.shape[0] % split_size))
        n_splits = int(fill_size / split_size)
        s, a, t = run_to_np(f, state_features, fill_size, with_attn_mask)
        s = s.reshape((n_splits, split_size, s.shape[1]))
        a = a.reshape((n_splits, split_size, a.shape[1]))
        t = t.reshape((n_splits, split_size))

        sts.append(s)
        acts.append(a)
        ts.append(t)

        lbs.append(np.ones(s.shape[0]))

    if save_data is None:
        return {
            labels[0]: np.concatenate(sts),
            labels[1]: np.concatenate(acts),
            labels[2]: np.concatenate(ts),
            "labels": np.concatenate(lbs),
        }
    else:
        data = {
            labels[0]: np.concatenate(sts),
            labels[1]: np.concatenate(acts),
            labels[2]: np.concatenate(ts),
            "labels": np.concatenate(lbs),
        }
        with h5py.File(save_data, "a") as f:
            # WARNING if this group already exists, datasets of the same name of "observations","timesteps","attn_mask", etc.
            # will be overwritten with this new datasets.
            for k in data:
                if k in f:
                    del f[k]
                    f.create_dataset(k, data=data[k], chunks=True)
                else:
                    f.create_dataset(k, data=data[k], chunks=True)
        return data


# 0 for loss, otherwise accuracy
def plot_training_validation_stats(
    load_log, ptype=0, eval_period=1, save_file=None, **kwargs
):
    L = pd.read_csv(load_log)
    L = L[(L.index % eval_period == 0) & (L.index >= eval_period)]

    x = L.index.to_numpy()
    if ptype:
        y = L["training_acc"].to_numpy()

        y2 = L["eval_acc"].to_numpy()

        fig, ax = plt.subplots()
        ax.plot(x, y, label="Training Accuracy")
        ax.plot(x, y2, label="Validation Accuracy")
        if "ylim" in kwargs:
            ax.set(
                ylim=kwargs["ylim"],
                xlabel="Epoch",
                ylabel="Preference Predictor Accuracy",
                title="Preference Transformer: Training vs. Validation Accuracy",
            )
        else:
            ax.set(
                xlabel="Epoch",
                ylabel="Preference Predictor Accuracy",
                title="Preference Transformer: Training vs. Validation Accuracy",
            )
    else:
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
