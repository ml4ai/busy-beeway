import jax
import jax.numpy as jnp
import minari
import numpy as np
import pandas as pd
from flax import nnx
from tqdm import tqdm
from transformers.models.policy import sample_actions


def rand_circle(R, N, C=(0, 0), rng=np.random.default_rng()):
    r = R * np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    return C[0] + r * np.cos(theta), C[1] + r * np.sin(theta)


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


# Find closest point on line a-b to point p
def closest_point_on_line(ax, ay, bx, by, px, py, thres):
    apx = px - ax
    apy = py - ay
    abx = bx - ax
    aby = by - ay

    ab2 = (abx**2) + (aby**2)
    # Accounts for obstacles wrapping around map
    cond = ab2 < (thres**2)
    apab = apx * abx + apy * aby
    if isinstance(cond, bool):
        t = np.asarray(apab) / np.asarray(ab2)
        t = np.where(np.isnan(t), 0.0, t)
        t = np.where(t < 0, 0.0, t)
        t = np.where(t > 1, 1.0, t)
        return ax + abx * t, ay + aby * t

    t = apab[cond] / ab2[cond]
    t = np.where(np.isnan(t), 0.0, t)
    t = np.where(t < 0, 0.0, t)
    t = np.where(t > 1, 1.0, t)
    return (ax[cond] + abx[cond] * t) * 1, (ay[cond] + aby[cond] * t) * 1


def point_collide(x1, y1, x2, y2, radius_1, radius_2=None):
    if radius_2 is None:
        radius_2 = radius_1
    dist = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
    tol = (radius_1 + radius_2) ** 2
    l = dist < tol
    e = np.isclose(dist, tol)
    return l | e


def collision(
    old_x, old_y, new_x, new_y, px, py, radius_1=0.3, radius_2=None, thres=2.0
):
    cpx, cpy = closest_point_on_line(old_x, old_y, new_x, new_y, px, py, thres)
    return (
        np.any(point_collide(cpx, cpy, px, py, radius_1, radius_2)),
        cpx * 1,
        cpy * 1,
    )


def find_direction(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return np.rad2deg(np.arctan2(y, x))


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


def bb_run_episode(
    d_model,
    r_model,
    move_stats,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    n_min_obstacles=6,
    days=181,
    rng=np.random.default_rng(),
):
    level = rng.choice([9, 10, 11])
    n_obstacles = 50 if level == 9 else 100 if level == 10 else 150
    ai = rng.choice([1, 2, 3, 4])
    attempt = rng.choice(4)
    if days is not None:
        day = rng.choice(days)

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state():
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        if days is not None:
            s.append(day * 1.0)

        return jnp.asarray(s)

    R = jnp.array(target_return).reshape(1, 1, -1)

    s = create_new_state().reshape(1, 1, -1)

    a = jnp.zeros((1, 0, 2))
    t = jnp.zeros((1, 1), dtype=jnp.int32)
    d_model = nnx.jit(d_model, static_argnums=5)
    episode_return = 0
    for i in tqdm(range(max_horizon), desc="Timesteps"):
        a = jnp.concat([a, jnp.zeros((1, 1, 2))], axis=1)
        a = a[:, -context_length:, :]
        _, _, action = d_model(
            R,
            s,
            a,
            t,
            jnp.ones((1, R.shape[1]), dtype=jnp.float32),
            training=False,
        )

        action = action[-1][-1]
        action = jnp.where(jnp.array([True, False]), jnp.clip(action, 0.0), action)
        a = a.at[-1, -1].set(action)

        reward, _ = r_model(
            s,
            a,
            t,
            jnp.ones((1, R.shape[1]), dtype=jnp.float32),
            training=False,
        )
        reward = reward["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )
        o_dists = rng.normal(move_stats[2], move_stats[3], n_obstacles)
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = np.sqrt((O_posX**2) + (O_posY**2))
        O_posX = np.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = np.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

        s = s[:, -context_length:, :]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, -1)], axis=1)
        R = R[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        if coll:
            break
    return -1, -1, episode_return


def bb_record_episode(
    d_model,
    r_model,
    move_stats,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    n_min_obstacles=6,
    days=181,
    level=None,
    ai=None,
    attempt=None,
    day=None,
    rng=np.random.default_rng(),
):

    if level is None:
        level = rng.choice([9, 10, 11])

    n_obstacles = 50 if level == 9 else 100 if level == 10 else 150

    if ai is None:
        ai = rng.choice([1, 2, 3, 4])
    if attempt is None:
        attempt = rng.choice(4)
    if days is not None:
        if day is None:
            day = rng.choice(days)

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state():
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        if days is not None:
            s.append(day * 1.0)

        return jnp.asarray(s)

    R = jnp.array(target_return).reshape(1, 1, -1)
    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 2))
    t = jnp.zeros((1, 1), dtype=jnp.int32)

    episode_return, episode_length = 0.0, 0

    p_X_list = [p_posX]
    p_Y_list = [p_posY]
    p_t_list = [0]

    O_list = [
        pd.DataFrame(
            {
                "posX": O_posX,
                "posY": O_posY,
                "t": [0] * int(n_obstacles),
            }
        )
    ]
    success = False
    d_model = nnx.jit(d_model, static_argnums=5)
    for i in tqdm(range(max_horizon), desc="Timesteps"):
        a = jnp.concat([a, jnp.zeros((1, 1, 2))], axis=1)
        a = a[:, -context_length:, :]
        _, _, action = d_model(
            R,
            s,
            a,
            t,
            jnp.ones((1, R.shape[1]), dtype=jnp.float32),
            training=False,
        )

        action = action[-1][-1]
        action = jnp.where(jnp.array([True, False]), jnp.clip(action, 0.0), action)
        a = a.at[-1, -1].set(action)

        reward, _ = r_model(
            s,
            a,
            t,
            jnp.ones((1, R.shape[1]), dtype=jnp.float32),
            training=False,
        )
        reward = reward["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )

        o_dists = rng.normal(move_stats[2], move_stats[3], n_obstacles)
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = np.sqrt((O_posX**2) + (O_posY**2))
        O_posX = np.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = np.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        g_coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)
        s = s[:, -context_length:, :]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, -1)], axis=1)
        R = R[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        episode_length += 1

        p_X_list.append(p_posX)
        p_Y_list.append(p_posY)
        p_t_list.append(i + 1)

        O_list.append(
            pd.DataFrame(
                {
                    "posX": O_posX,
                    "posY": O_posY,
                    "t": [i + 1] * int(n_obstacles),
                }
            )
        )

        if coll:
            break
        if g_coll:
            success = True
            break

    p_df = pd.DataFrame(
        {
            "posX": p_X_list,
            "posY": p_Y_list,
            "t": p_t_list,
        }
    )

    o_pos = pd.concat(O_list)
    o_pos.reset_index
    return (
        episode_return,
        episode_length,
        {
            "player": p_df,
            "obstacles": o_pos,
            "reached_goal": success,
            "goal": g,
        },
    )


def bb_run_episode_IQL(
    policy,
    r_model,
    move_stats,
    max_horizon=500,
    n_min_obstacles=6,
    days=181,
    context_length=100,
    rngs=nnx.Rngs(sample=4),
):
    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    rng = np.random.default_rng(int(t_keys[0]))
    level = rng.choice([9, 10, 11])
    n_obstacles = 50 if level == 9 else 100 if level == 10 else 150
    ai = rng.choice([1, 2, 3, 4])
    attempt = rng.choice(4)
    if days is not None:
        day = rng.choice(days)

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state():
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        if days is not None:
            s.append(day * 1.0)

        return jnp.asarray(s)

    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 2))
    t = jnp.zeros((1, 1), dtype=jnp.int32)

    episode_return = 0
    for i in tqdm(range(max_horizon), desc="Timesteps"):
        action = sample_actions(policy, s[-1, -1], 0.0, rngs)
        action = jnp.where(jnp.array([True, False]), jnp.clip(action, 0.0), action)
        a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
        a = a[:, -context_length:, :]

        reward, _ = r_model(
            s,
            a,
            t,
            jnp.ones((1, t.shape[1]), dtype=jnp.float32),
            training=False,
        )
        reward = reward["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )
        o_dists = rng.normal(move_stats[2], move_stats[3], n_obstacles)
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = np.sqrt((O_posX**2) + (O_posY**2))
        O_posX = np.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = np.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

        s = s[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        if coll:
            break
    return -1, -1, episode_return


def bb_record_episode_IQL(
    policy,
    r_model,
    move_stats,
    max_horizon=500,
    n_min_obstacles=6,
    days=181,
    level=None,
    ai=None,
    attempt=None,
    day=None,
    context_length=100,
    rngs=nnx.Rngs(sample=4),
):

    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    rng = np.random.default_rng(int(t_keys[0]))
    if level is None:
        level = rng.choice([9, 10, 11])

    n_obstacles = 50 if level == 9 else 100 if level == 10 else 150

    if ai is None:
        ai = rng.choice([1, 2, 3, 4])
    if attempt is None:
        attempt = rng.choice(4)
    if days is not None:
        if day is None:
            day = rng.choice(days)

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state():
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        if days is not None:
            s.append(day * 1.0)

        return jnp.asarray(s)

    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 2))
    t = jnp.zeros((1, 1), dtype=jnp.int32)

    episode_return, episode_length = 0.0, 0

    p_X_list = [p_posX]
    p_Y_list = [p_posY]
    p_t_list = [0]

    O_list = [
        pd.DataFrame(
            {
                "posX": O_posX,
                "posY": O_posY,
                "t": [0] * int(n_obstacles),
            }
        )
    ]
    success = False

    for i in tqdm(range(max_horizon), desc="Timesteps"):
        action = sample_actions(policy, s[-1, -1], 0.0, rngs)
        action = jnp.where(jnp.array([True, False]), jnp.clip(action, 0.0), action)
        a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
        a = a[:, -context_length:, :]

        reward, _ = r_model(
            s,
            a,
            t,
            jnp.ones((1, t.shape[1]), dtype=jnp.float32),
            training=False,
        )
        reward = reward["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )

        o_dists = rng.normal(move_stats[2], move_stats[3], n_obstacles)
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = np.sqrt((O_posX**2) + (O_posY**2))
        O_posX = np.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = np.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        g_coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

        s = s[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        episode_length += 1

        p_X_list.append(p_posX)
        p_Y_list.append(p_posY)
        p_t_list.append(i + 1)

        O_list.append(
            pd.DataFrame(
                {
                    "posX": O_posX,
                    "posY": O_posY,
                    "t": [i + 1] * int(n_obstacles),
                }
            )
        )

        if coll:
            break
        if g_coll:
            success = True
            break

    p_df = pd.DataFrame(
        {
            "posX": p_X_list,
            "posY": p_Y_list,
            "t": p_t_list,
        }
    )

    o_pos = pd.concat(O_list)
    o_pos.reset_index
    return (
        episode_return,
        episode_length,
        {
            "player": p_df,
            "obstacles": o_pos,
            "reached_goal": success,
            "goal": g,
        },
    )


# If no r_model is given, then episode_return == task_episode_return (if set to True)
# normalized_score == True does nothing if compute_task_return != True
def run_antmaze_medium(
    d_model,
    r_model,
    move_stats,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    compute_task_return=False,
    normalized_score=False,
    rng=np.random.default_rng(),
):
    dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")
    env = dataset.recover_environment()
    m_seed = rng.integer(0, 10000)
    obs, info = env.reset(seed=m_seed)

    episode_over = False
    d_model = nnx.jit(d_model, static_argnums=5)

    R = jnp.array(target_return).reshape(1, 1, 1)
    s = jnp.concatenate(
        [
            obs["desired_goal"],
            obs["achieved_goal"],
            obs["observation"],
        ],
    ).reshape(1, 1, -1)
    a = jnp.zeros((1, 0, env.action_space.shape[0]))
    t = jnp.zeros((1, 1), dtype=jnp.int32)
    episode_return = 0
    task_episode_return = 0
    if r_model is None:
        while not episode_over:
            a = jnp.concat([a, jnp.zeros((1, 1, env.action_space.shape[0]))], axis=1)
            a = a[:, -context_length:, :]
            _, _, action = d_model(
                R,
                s,
                a,
                t,
                jnp.ones((1, R.shape[1]), dtype=jnp.float32),
                training=False,
            )
            action = action[-1][-1]
            a = a.at[-1, -1].set(action)
            obs, reward, terminated, truncated, info = env.step(action)
            s = jnp.concatenate(
                [
                    s,
                    jnp.concatenate(
                        [
                            obs["desired_goal"],
                            obs["achieved_goal"],
                            obs["observation"],
                        ],
                    ).reshape(1, 1, -1),
                ],
                axis=1,
            )
            s = s[:, -context_length:, :]
            R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, 1)], axis=1)
            R = R[:, -context_length:, :]
            t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, 1)], axis=1)
            t = t[:, -context_length:]
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += reward
    else:
        while not episode_over:
            a = jnp.concat([a, jnp.zeros((1, 1, env.action_space.shape[0]))], axis=1)
            a = a[:, -context_length:, :]
            _, _, action = d_model(
                R,
                s,
                a,
                t,
                jnp.ones((1, R.shape[1]), dtype=jnp.float32),
                training=False,
            )
            action = action[-1][-1]
            a = a.at[-1, -1].set(action)

            reward, _ = r_model(
                s,
                a,
                t,
                jnp.ones((1, R.shape[1]), dtype=jnp.float32),
                training=False,
            )
            reward = reward["value"][:, 0, -1]
            obs, t_reward, terminated, truncated, info = env.step(action)
            s = jnp.concatenate(
                [
                    s,
                    jnp.concatenate(
                        [
                            obs["desired_goal"],
                            obs["achieved_goal"],
                            obs["observation"],
                        ],
                    ).reshape(1, 1, -1),
                ],
                axis=1,
            )
            s = s[:, -context_length:, :]
            R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, 1)], axis=1)
            R = R[:, -context_length:, :]
            t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, 1)], axis=1)
            t = t[:, -context_length:]
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += t_reward
    env.close()
    if compute_task_return:
        if normalized_score:
            return (
                episode_return,
                task_episode_return,
                minari.get_normalized_score(dataset, task_episode_return),
            )
        return -1, episode_return, task_episode_return
    return -1, -1, episode_return


# If no r_model is given, then episode_return == task_episode_return (if set to True)
# normalized_score == True does nothing if compute_task_return != True
def run_antmaze_medium_IQL(
    policy,
    r_model=None,
    move_stats=None,
    max_horizon=500,
    compute_task_return=True,
    normalized_score=True,
    context_length=100,
    animate=False,
    rngs=nnx.Rngs(sample=4),
):
    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")
    if animate:
        env = dataset.recover_environment(render_mode="human")
    else:
        env = dataset.recover_environment()

    obs, info = env.reset(seed=int(t_keys[0]))

    episode_over = False
    episode_return = 0
    task_episode_return = 0
    if r_model is None:
        s = jnp.concatenate(
            [
                obs["desired_goal"],
                obs["achieved_goal"],
                obs["observation"],
            ],
        )
        while not episode_over:
            action = sample_actions(policy, s, 0.0, rngs)
            action = jnp.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            s = jnp.concatenate(
                [
                    obs["desired_goal"],
                    obs["achieved_goal"],
                    obs["observation"],
                ],
            )
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += reward
    else:
        s = jnp.concatenate(
            [
                obs["desired_goal"],
                obs["achieved_goal"],
                obs["observation"],
            ],
        ).reshape(1, 1, -1)
        a = jnp.zeros((1, 0, env.action_space.shape[0]))
        t = jnp.zeros((1, 1), dtype=jnp.int32)
        while not episode_over:
            action = sample_actions(policy, s[-1, -1], 0.0, rngs)
            action = jnp.clip(action, -1.0, 1.0)
            a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
            a = a[:, -context_length:, :]

            reward, _ = r_model(
                s,
                a,
                t,
                jnp.ones((1, t.shape[1]), dtype=jnp.float32),
                training=False,
            )
            reward = reward["value"][:, 0, -1]
            obs, t_reward, terminated, truncated, info = env.step(action)
            s = jnp.concatenate(
                [
                    s,
                    jnp.concatenate(
                        [
                            obs["desired_goal"],
                            obs["achieved_goal"],
                            obs["observation"],
                        ],
                    ).reshape(1, 1, -1),
                ],
                axis=1,
            )
            s = s[:, -context_length:, :]
            t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
            t = t[:, -context_length:]
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += t_reward
    env.close()
    if compute_task_return:
        if normalized_score:
            return (
                episode_return,
                task_episode_return,
                minari.get_normalized_score(dataset, task_episode_return),
            )
        return -1, episode_return, task_episode_return
    return -1, -1, episode_return


# If no r_model is given, then episode_return == task_episode_return (if set to True)
# normalized_score == True does nothing if compute_task_return != True
def run_pen_human_IQL(
    policy,
    r_model=None,
    move_stats=None,
    max_horizon=500,
    compute_task_return=True,
    normalized_score=True,
    context_length=100,
    animate=False,
    rngs=nnx.Rngs(sample=4),
):
    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    dataset = minari.load_dataset("D4RL/pen/human-v2")
    if animate:
        env = dataset.recover_environment(render_mode="human")
    else:
        env = dataset.recover_environment()

    obs, info = env.reset(seed=int(t_keys[0]))

    episode_over = False
    episode_return = 0
    task_episode_return = 0
    if r_model is None:
        s = obs
        while not episode_over:
            action = sample_actions(policy, s, 0.0, rngs)
            action = jnp.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            s = obs
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += reward
    else:
        s = obs.reshape(1, 1, -1)
        a = jnp.zeros((1, 0, env.action_space.shape[0]))
        t = jnp.zeros((1, 1), dtype=jnp.int32)
        while not episode_over:
            action = sample_actions(policy, s[-1, -1], 0.0, rngs)
            action = jnp.clip(action, -1.0, 1.0)
            a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
            a = a[:, -context_length:, :]

            reward, _ = r_model(
                s,
                a,
                t,
                jnp.ones((1, t.shape[1]), dtype=jnp.float32),
                training=False,
            )
            reward = reward["value"][:, 0, -1]
            obs, t_reward, terminated, truncated, info = env.step(action)
            s = jnp.concatenate(
                [
                    s,
                    obs.reshape(1, 1, -1),
                ],
                axis=1,
            )
            s = s[:, -context_length:, :]
            t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
            t = t[:, -context_length:]
            episode_over = terminated or truncated
            episode_return += reward
            task_episode_return += t_reward
    env.close()
    if compute_task_return:
        if normalized_score:
            return (
                episode_return,
                task_episode_return,
                minari.get_normalized_score(dataset, task_episode_return),
            )
        return -1, episode_return, task_episode_return
    return -1, -1, episode_return


def bb_record_only_goal(
    max_speed,
    obs_dist=10,
    n_min_obstacles=6,
    max_horizon=500,
    days=181,
    seed=4,
):
    rng = np.random.default_rng(seed)
    level = rng.choice([9, 10, 11])
    ai = rng.choice([1, 2, 3, 4])
    attempt = rng.choice(4)
    day = rng.choice(days)
    p_samp = rand_circle(50, None, rng=rng)

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    theta = rng.random(n_min_obstacles) * 2 * np.pi

    O_posX = p_posX + obs_dist * np.cos(theta)
    O_posY = p_posY + obs_dist * np.sin(theta)

    O_angle = rng.uniform(0.0, 360.0, n_min_obstacles)

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state(
        g,
        p_posX,
        p_posY,
        O_posX,
        O_posY,
        O_angle,
        level,
        ai,
        attempt,
        day,
        n_min_obstacles,
    ):
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        s.append(day * 1.0)

        return np.asarray(s)

    s = create_new_state(
        g,
        p_posX,
        p_posY,
        O_posX,
        O_posY,
        O_angle,
        level,
        ai,
        attempt,
        day,
        n_min_obstacles,
    ).reshape(1, 1, -1)
    a = np.zeros((1, 0, 2))
    t = np.zeros((1, 1), dtype=np.int32)

    while True:
        g_dir = find_direction(p_posX, p_posY, g[0], g[1])
        action = np.asarray([max_speed, g_dir])
        a = np.concatenate([a, action.reshape(1, 1, -1)], axis=1)

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (0 * cos_plus(O_angle))
        O_posY = O_posY + (0 * sin_plus(O_angle))

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )
        if coll or a.shape[1] == max_horizon:
            break

        s = np.concatenate(
            [
                s,
                create_new_state(
                    g,
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                    O_angle,
                    level,
                    ai,
                    attempt,
                    day,
                    n_min_obstacles,
                ).reshape(1, 1, -1),
            ],
            axis=1,
        )

        t = np.concatenate([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
    return s, a, t


def bb_record_opposite_goal(
    max_speed,
    obs_dist=10,
    n_min_obstacles=6,
    max_horizon=100,
    days=181,
    seed=4,
):
    rng = np.random.default_rng(seed)
    level = rng.choice([9, 10, 11])
    ai = rng.choice([1, 2, 3, 4])
    attempt = rng.choice(4)
    day = rng.choice(days)
    p_samp = rand_circle(50, None, rng=rng)

    p_posX = float(p_samp[0])
    p_posY = float(p_samp[1])

    theta = rng.random(n_min_obstacles) * 2 * np.pi

    O_posX = p_posX + obs_dist * np.cos(theta)
    O_posY = p_posY + obs_dist * np.sin(theta)

    O_angle = rng.uniform(0.0, 360.0, n_min_obstacles)

    while True:
        g_h = rng.uniform(0.0, 360.0)
        g_r = rng.normal(30)
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break

    def create_new_state(
        g,
        p_posX,
        p_posY,
        O_posX,
        O_posY,
        O_angle,
        level,
        ai,
        attempt,
        day,
        n_min_obstacles,
    ):
        s = [p_posX, p_posY]
        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

        for i in range(n_min_obstacles):
            s += [
                O_posX[min_dist_obs[i]],
                O_posY[min_dist_obs[i]],
                O_angle[min_dist_obs[i]],
            ]

        s += [g[0], g[1]]

        s.append(level * 1.0)
        s.append(ai * 1.0)
        s.append(attempt * 1.0)

        s.append(day * 1.0)

        return np.asarray(s)

    s = create_new_state(
        g,
        p_posX,
        p_posY,
        O_posX,
        O_posY,
        O_angle,
        level,
        ai,
        attempt,
        day,
        n_min_obstacles,
    ).reshape(1, 1, -1)
    a = np.zeros((1, 0, 2))
    t = np.zeros((1, 1), dtype=np.int32)
    while True:
        og_dir = find_direction(p_posX, p_posY, g[0], g[1])
        og_dir = og_dir + 180
        og_dir = og_dir - 360 if og_dir > 180 else og_dir
        action = np.asarray([max_speed, og_dir])
        a = np.concatenate([a, action.reshape(1, 1, -1)], axis=1)

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
        p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            O_posX,
            O_posY,
        )
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (0 * cos_plus(O_angle))
        O_posY = O_posY + (0 * sin_plus(O_angle))

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )

        coll, _, _ = collision(
            old_p_posX,
            old_p_posY,
            p_posX,
            p_posY,
            g[0],
            g[1],
            radius_2=1.0,
        )
        if coll or a.shape[1] == max_horizon:
            break
        s = np.concatenate(
            [
                s,
                create_new_state(
                    g,
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                    O_angle,
                    level,
                    ai,
                    attempt,
                    day,
                    n_min_obstacles,
                ).reshape(1, 1, -1),
            ],
            axis=1,
        )

        t = np.concatenate([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
    return s, a, t
