import jax.numpy as jnp
import numpy as np
import pandas as pd


# Finds distance between a set of coordinates and a single coordinate. vecX and vecY are numpy arrays, px and py are scalars (floats/ints/etc.)
# Outputs numpy array
def vec_point_dist(vecX, vecY, px, py):
    return np.sqrt(((vecX - px) ** 2) + ((vecY - py) ** 2))


def max_seq_length(F):
    return max([f.shape[0] for f in F])


def compute_run_features(p_df, g, O):
    goal_distance = vec_point_dist(
        p_df["posX"].to_numpy(), p_df["posY"].to_numpy(), g[0], g[1]
    )

    min_obstacle_distance = np.zeros(p_df.shape[0])
    for index, row in p_df.iterrows():
        o_t = O[O["t"] == row["t"]]
        obs_distance = vec_point_dist(
            o_t["posX"].to_numpy(),
            o_t["posY"].to_numpy(),
            row["posX"],
            row["posY"],
        )
        min_dist = min(obs_distance)
        min_obstacle_distance[index] = min_dist

    return pd.DataFrame(
        {
            "goal_distance": goal_distance,
            "min_obstacle_distance": min_obstacle_distance,
            "t": p_df["t"].to_numpy(),
        }
    )


def compute_features(D):
    dat = []
    for d in D:
        p_df = d["player"]
        g = d["goal"]
        O = d["obstacles"]
        dat.append(compute_run_features(p_df, g, O))
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
        return jnp.array(nf[:, 0:-1]), jnp.array(nf[:, -1], dtype=jnp.int32), jnp.array(attn_mask, dtype=jnp.float32)
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
