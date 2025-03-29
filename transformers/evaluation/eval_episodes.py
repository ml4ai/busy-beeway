import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from tqdm import tqdm
import minari
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
    ax = np.asarray(ax)
    ay = np.asarray(ay)
    bx = np.asarray(bx)
    by = np.asarray(by)
    px = np.asarray(px)
    py = np.asarray(py)
    apx = px - ax
    apy = py - ay
    abx = bx - ax
    aby = by - ay

    ab2 = (abx**2) + (aby**2)
    # Accounts for obstacles wrapping around map
    cond = ab2 < (thres**2)
    apab = apx * abx + apy * aby
    if isinstance(cond, bool):
        t = apab / ab2
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
    degs = np.arctan2(y, x) * (180.0 / np.pi)
    degs = np.where(np.isclose(degs, 0.0), 360.0, degs)
    degs = np.where(degs < 0, degs + 360.0, degs)
    return degs * 1


def bb_run_episode(
    d_model,
    r_model,
    move_stats,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    days=153,
    rng=np.random.default_rng(),
):
    n_obstacles = rng.choice([50, 100, 150])
    ai = rng.choice(4)
    p_attempt = rng.choice(4)
    if days is not None:
        day = rng.choice(days)
    match ai:
        case 0:
            sig = 55
            repel = 72
        case 1:
            sig = 62
            repel = 98
        case 2:
            sig = 140
            repel = 98
        case 3:
            sig = 100
            repel = 75

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = p_samp[0]
    p_posY = p_samp[1]

    p_angle = rng.uniform(0.0, 360.0)

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
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = np.argmin(obs_distances)
        min_obstacle_distances = obs_distances[min_dist_obs]

        obs_headings = (
            cos_plus(
                find_direction(
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                )
                - p_angle
            )
            + 2
        )
        min_distance_obstacle_headings = obs_headings[min_dist_obs]

        max_heading_obs = np.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = np.argmax(op_headings)

        max_op_headings = op_headings[max_heading_op]

        min_distance_op_headings = op_headings[min_dist_obs]

        max_heading_obstacle_op_headings = op_headings[max_heading_obs]

        max_heading_op_distances = obs_distances[max_heading_op]

        max_heading_op_obstacle_headings = obs_headings[max_heading_op]

        obstacle_count = n_obstacles * 1.0

        sigma = sig * 1.0

        repel_factor = repel * 1.0

        attempt = p_attempt * 1.0
        if days is not None:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                    day,
                ]
            )
        else:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                ]
            )
        return s

    R = jnp.array(target_return).reshape(1, 1, -1)

    s = create_new_state().reshape(1, 1, -1)

    a = jnp.zeros((1, 0, 3))
    t = jnp.zeros((1, 1), dtype=jnp.int32)
    d_model = nnx.jit(d_model, static_argnums=5)
    episode_return = 0
    for i in tqdm(range(max_horizon), desc="Timesteps"):
        a = jnp.concat([a, jnp.zeros((1, 1, 3))], axis=1)
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
        p_angle = float(action[1])

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

        s = s[:, -context_length:, :]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, -1)], axis=1)
        R = R[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        if coll:
            break
    return episode_return


def bb_record_episode(
    d_model,
    r_model,
    move_stats,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    days=153,
    n_obstacles=None,
    ai=None,
    p_attempt=None,
    day=None,
    rng=np.random.default_rng(),
):

    if n_obstacles is None:
        n_obstacles = rng.choice([50, 100, 150])

    if ai is None:
        ai = rng.choice(4)
    if p_attempt is None:
        p_attempt = rng.choice(4)
    if days is not None:
        if day is None:
            day = rng.choice(days)
    match ai:
        case 0:
            sig = 55
            repel = 72
        case 1:
            sig = 62
            repel = 98
        case 2:
            sig = 140
            repel = 98
        case 3:
            sig = 100
            repel = 75

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = p_samp[0]
    p_posY = p_samp[1]

    p_angle = rng.uniform(0.0, 360.0)

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
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = np.argmin(obs_distances)
        min_obstacle_distances = obs_distances[min_dist_obs]

        obs_headings = (
            cos_plus(
                find_direction(
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                )
                - p_angle
            )
            + 2
        )
        min_distance_obstacle_headings = obs_headings[min_dist_obs]

        max_heading_obs = np.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = np.argmax(op_headings)

        max_op_headings = op_headings[max_heading_op]

        min_distance_op_headings = op_headings[min_dist_obs]

        max_heading_obstacle_op_headings = op_headings[max_heading_obs]

        max_heading_op_distances = obs_distances[max_heading_op]

        max_heading_op_obstacle_headings = obs_headings[max_heading_op]

        obstacle_count = n_obstacles * 1.0

        sigma = sig * 1.0

        repel_factor = repel * 1.0

        attempt = p_attempt * 1.0
        if days is not None:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                    day,
                ]
            )
        else:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                ]
            )
        return s

    R = jnp.array(target_return).reshape(1, 1, -1)
    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 3))
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
        a = jnp.concat([a, jnp.zeros((1, 1, 3))], axis=1)
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
        p_angle = float(action[1])
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
    days=153,
    rngs=nnx.Rngs(sample=4),
):
    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    rng = np.random.default_rng(t_keys[0])
    n_obstacles = rng.choice([50, 100, 150])
    ai = rng.choice(4)
    p_attempt = rng.choice(4)
    if days is not None:
        day = rng.choice(days)
    match ai:
        case 0:
            sig = 55
            repel = 72
        case 1:
            sig = 62
            repel = 98
        case 2:
            sig = 140
            repel = 98
        case 3:
            sig = 100
            repel = 75

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = p_samp[0]
    p_posY = p_samp[1]

    p_angle = rng.uniform(0.0, 360.0)

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
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = np.argmin(obs_distances)
        min_obstacle_distances = obs_distances[min_dist_obs]

        obs_headings = (
            cos_plus(
                find_direction(
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                )
                - p_angle
            )
            + 2
        )
        min_distance_obstacle_headings = obs_headings[min_dist_obs]

        max_heading_obs = np.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = np.argmax(op_headings)

        max_op_headings = op_headings[max_heading_op]

        min_distance_op_headings = op_headings[min_dist_obs]

        max_heading_obstacle_op_headings = op_headings[max_heading_obs]

        max_heading_op_distances = obs_distances[max_heading_op]

        max_heading_op_obstacle_headings = obs_headings[max_heading_op]

        obstacle_count = n_obstacles * 1.0

        sigma = sig * 1.0

        repel_factor = repel * 1.0

        attempt = p_attempt * 1.0
        if days is not None:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                    day,
                ]
            )
        else:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                ]
            )
        return s

    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 3))
    t = jnp.zeros((1, 1), dtype=jnp.int32)

    episode_return = 0
    for i in tqdm(range(max_horizon), desc="Timesteps"):
        action = sample_actions(policy, s[-1, -1], 0.0, rngs)
        action = jnp.where(
            jnp.array([False, False, True]), jnp.round(jnp.clip(action, 0, 1)), action
        )
        action = jnp.where(
            jnp.array([False, True, False]), jnp.clip(action, 0.0, 360.0), action
        )
        a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
        a = a[:, -context_length:, :]

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
        p_angle = float(action[1])

        s = jnp.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

        s = s[:, -context_length:, :]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, -1)], axis=1)
        R = R[:, -context_length:, :]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
        t = t[:, -context_length:]

        episode_return += reward
        if coll:
            break
    return episode_return


def bb_record_episode_IQL(
    policy,
    r_model,
    move_stats,
    max_horizon=500,
    days=153,
    n_obstacles=None,
    ai=None,
    p_attempt=None,
    day=None,
    rngs=nnx.Rngs(sample=4),
):

    key = rngs.sample()
    t_keys = jax.random.randint(key, 2, 0, 10000)
    rng = np.random.default_rng(t_keys[0])
    if n_obstacles is None:
        n_obstacles = rng.choice([50, 100, 150])

    if ai is None:
        ai = rng.choice(4)
    if p_attempt is None:
        p_attempt = rng.choice(4)
    if days is not None:
        if day is None:
            day = rng.choice(days)
    match ai:
        case 0:
            sig = 55
            repel = 72
        case 1:
            sig = 62
            repel = 98
        case 2:
            sig = 140
            repel = 98
        case 3:
            sig = 100
            repel = 75

    O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

    O_angle = rng.uniform(0.0, 360.0, n_obstacles)

    while True:
        p_samp = rand_circle(50, None, rng=rng)
        if np.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break

    p_posX = p_samp[0]
    p_posY = p_samp[1]

    p_angle = rng.uniform(0.0, 360.0)

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
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = np.argmin(obs_distances)
        min_obstacle_distances = obs_distances[min_dist_obs]

        obs_headings = (
            cos_plus(
                find_direction(
                    p_posX,
                    p_posY,
                    O_posX,
                    O_posY,
                )
                - p_angle
            )
            + 2
        )
        min_distance_obstacle_headings = obs_headings[min_dist_obs]

        max_heading_obs = np.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = np.argmax(op_headings)

        max_op_headings = op_headings[max_heading_op]

        min_distance_op_headings = op_headings[min_dist_obs]

        max_heading_obstacle_op_headings = op_headings[max_heading_obs]

        max_heading_op_distances = obs_distances[max_heading_op]

        max_heading_op_obstacle_headings = obs_headings[max_heading_op]

        obstacle_count = n_obstacles * 1.0

        sigma = sig * 1.0

        repel_factor = repel * 1.0

        attempt = p_attempt * 1.0
        if days is not None:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                    day,
                ]
            )
        else:
            s = jnp.array(
                [
                    goal_distances,
                    goal_headings,
                    min_obstacle_distances,
                    min_distance_obstacle_headings,
                    max_obstacle_headings,
                    max_heading_obstacle_distances,
                    max_op_headings,
                    min_distance_op_headings,
                    max_heading_obstacle_op_headings,
                    max_heading_op_distances,
                    max_heading_op_obstacle_headings,
                    obstacle_count,
                    sigma,
                    repel_factor,
                    attempt,
                ]
            )
        return s

    s = create_new_state().reshape(1, 1, -1)
    a = jnp.zeros((1, 0, 3))
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
        action = jnp.where(
            jnp.array([False, False, True]), jnp.round(jnp.clip(action, 0, 1)), action
        )
        action = jnp.where(
            jnp.array([False, True, False]), jnp.clip(action, 0.0, 360.0), action
        )
        a = jnp.concat([a, action.reshape(1, 1, -1)], axis=1)
        a = a[:, -context_length:, :]

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
        p_angle = float(action[1])

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

    obs, info = env.reset()

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
        return episode_return, task_episode_return
    return episode_return


# If no r_model is given, then episode_return == task_episode_return (if set to True)
# normalized_score == True does nothing if compute_task_return != True
def run_antmaze_medium_IQL(
    policy,
    r_model,
    move_stats,
    max_horizon=500,
    compute_task_return=False,
    normalized_score=False,
    rngs=nnx.Rngs(sample=4),
):
    dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")
    env = dataset.recover_environment()

    obs, info = env.reset()

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
            R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, -1)], axis=1)
            R = R[:, -context_length:, :]
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
        return episode_return, task_episode_return
    return episode_return
