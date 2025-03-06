import jax
import jax.numpy as jnp
import pandas as pd


def point_dist(vecX, vecY, px, py):
    return jnp.sqrt(((vecX - px) ** 2) + ((vecY - py) ** 2)) * 1


def cos_plus(degrees):
    res = jnp.cos(degrees * (jnp.pi / 180.0))
    res = jnp.where(jnp.isclose(degrees, 90), 0.0, res)
    res = jnp.where(jnp.isclose(degrees, 270), 0.0, res)
    return res * 1


def sin_plus(degrees):
    res = jnp.sin(degrees * (jnp.pi / 180.0))
    res = jnp.where(jnp.isclose(degrees, 360), 0.0, res)
    res = jnp.where(jnp.isclose(degrees, 180), 0.0, res)
    return res * 1


# Find closest point on line a-b to point p
def closest_point_on_line(ax, ay, bx, by, px, py, thres):
    ax = jnp.asarray(ax)
    ay = jnp.asarray(ay)
    bx = jnp.asarray(bx)
    by = jnp.asarray(by)
    px = jnp.asarray(px)
    py = jnp.asarray(py)
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
        t = jnp.where(jnp.isnan(t), 0.0, t)
        t = jnp.where(t < 0, 0.0, t)
        t = jnp.where(t > 1, 1.0, t)
        return ax + abx * t, ay + aby * t

    t = apab[cond] / ab2[cond]
    t = jnp.where(jnp.isnan(t), 0.0, t)
    t = jnp.where(t < 0, 0.0, t)
    t = jnp.where(t > 1, 1.0, t)
    return (ax[cond] + abx[cond] * t) * 1, (ay[cond] + aby[cond] * t) * 1


def point_collide(x1, y1, x2, y2, radius_1, radius_2=None):
    if radius_2 is None:
        radius_2 = radius_1
    dist = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
    tol = (radius_1 + radius_2) ** 2
    l = dist < tol
    e = jnp.isclose(dist, tol)
    return l | e


def collision(
    old_x, old_y, new_x, new_y, px, py, radius_1=0.3, radius_2=None, thres=2.0
):
    cpx, cpy = closest_point_on_line(old_x, old_y, new_x, new_y, px, py, thres)
    return (
        jnp.any(point_collide(cpx, cpy, px, py, radius_1, radius_2)),
        cpx * 1,
        cpy * 1,
    )


def find_direction(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    degs = jnp.arctan2(y, x) * (180.0 / jnp.pi)
    degs = jnp.where(jnp.isclose(degs, 0.0), 360.0, degs)
    degs = jnp.where(degs < 0, degs + 360.0, degs)
    return degs * 1


def bb_run_episode(
    d_model,
    r_model,
    move_stats,
    rng_key,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    days=153,
):
    key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11 = (
        jax.random.split(rng_key, 11)
    )
    n_obstacles = jax.random.choice(key1, jnp.asarray([50, 100, 150]))
    ai = jax.random.choice(key2, 4)
    p_attempt = jax.random.choice(key3, 4)
    if days is not None:
        day = jax.random.choice(key11, days)
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

    O_samps = jax.random.ball(key4, 2, shape=(n_obstacles,)) * 50

    O_posX = O_samps[:, 0]
    O_posY = O_samps[:, 1]
    O_angle = jax.random.uniform(key5, shape=(n_obstacles,), maxval=360.0)

    while True:
        p_samp = jax.random.ball(key6, 2) * 50
        if jnp.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break
        _, key6 = jax.random.split(key6)

    p_posX = p_samp[0]
    p_posY = p_samp[1]
    p_angle = jax.random.uniform(key7, maxval=360.0)

    while True:
        g_h = jax.random.uniform(key8, maxval=360.0)
        g_r = jax.random.normal(key9) + 30
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break
        _, key8 = jax.random.split(key8)
        _, key9 = jax.random.split(key9)

    def create_new_state():
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = jnp.argmin(obs_distances)
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

        max_heading_obs = jnp.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = jnp.argmax(op_headings)

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

    R = jnp.array(target_return).reshape(1, 1, 1)
    if days is not None:
        s = create_new_state().reshape(1, 1, 16)
    else:
        s = create_new_state().reshape(1, 1, 15)
    a = jnp.zeros((1, 0, 3))
    t = jnp.zeros((1, 1), dtype=jnp.int32)

    episode_return, episode_length = 0, 0
    keys = jax.random.split(key10, max_horizon + 1)
    key = keys[0]
    data_keys = keys[1:]
    for i in range(max_horizon):
        a = jnp.concat([a, jnp.zeros((1, 1, 3))], axis=1)
        a = a[-context_length:]
        _, _, action = d_model._train_state.apply_fn(
            d_model._train_state.params,
            R,
            s,
            a,
            t,
            training=False,
            attn_mask=jnp.ones((1, R.shape[1]), dtype=jnp.float32),
        )

        action = action[-1][-1]
        a = a.at[-1, -1].set(action)

        preds, _ = r_model._train_state.apply_fn(
            r_model._train_state.params,
            s,
            a,
            t,
            training=False,
            attn_mask=jnp.ones((1, R.shape[1]), dtype=jnp.float32),
        )
        reward = preds["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = p_posX + (action[0] * cos_plus(action[1]))
        p_posY = p_posY + (action[0] * sin_plus(action[1]))

        coll, _, _ = collision(
            float(old_p_posX),
            float(old_p_posY),
            float(p_posX),
            float(p_posY),
            O_posX,
            O_posY,
        )

        o_dists = (
            move_stats[3] * jax.random.normal(data_keys[i], shape=(n_obstacles,))
        ) + move_stats[2]
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = jnp.sqrt((O_posX**2) + (O_posY**2))
        O_posX = jnp.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = jnp.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            float(p_posX),
            float(p_posY),
        )

        coll, _, _ = collision(
            float(old_p_posX),
            float(old_p_posY),
            float(p_posX),
            float(p_posY),
            g[0],
            g[1],
            radius_2=1.0,
        )
        p_angle = action[1]
        if days is not None:
            s = jnp.concat([s, create_new_state().reshape(1, 1, 16)], axis=1)
        else:
            s = jnp.concat([s, create_new_state().reshape(1, 1, 15)], axis=1)
        s[-context_length:]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, 1)], axis=1)
        R[-context_length:]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, 1)], axis=1)
        t[-context_length:]

        episode_return += reward
        episode_length += 1
        if coll:
            break
    return episode_return, episode_length


def bb_record_episode(
    d_model,
    r_model,
    move_stats,
    rng_key,
    context_length=100,
    target_return=100.0,
    max_horizon=500,
    days=153,
    n_obstacles=None,
    ai=None,
    p_attempt=None,
    day=None,
):
    key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, key11 = (
        jax.random.split(rng_key, 11)
    )

    if n_obstacles is None:
        n_obstacles = jax.random.choice(key1, jnp.asarray([50, 100, 150]))

    if ai is None:
        ai = jax.random.choice(key2, 4)
    if p_attempt is None:
        p_attempt = jax.random.choice(key3, 4)
    if days is not None:
        if day is None:
            day = jax.random.choice(key11, days)
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

    O_samps = jax.random.ball(key4, 2, shape=(n_obstacles,)) * 50

    O_posX = O_samps[:, 0]
    O_posY = O_samps[:, 1]
    O_angle = jax.random.uniform(key5, shape=(n_obstacles,), maxval=360.0)

    while True:
        p_samp = jax.random.ball(key6, 2) * 50
        if jnp.all(((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1):
            break
        _, key6 = jax.random.split(key6)

    p_posX = p_samp[0]
    p_posY = p_samp[1]
    p_angle = jax.random.uniform(key7, maxval=360.0)

    while True:
        g_h = jax.random.uniform(key8, maxval=360.0)
        g_r = jax.random.normal(key9) + 30
        g = (
            float(p_posX + g_r * cos_plus(g_h)),
            float(p_posY + g_r * sin_plus(g_h)),
        )
        if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
            break
        _, key8 = jax.random.split(key8)
        _, key9 = jax.random.split(key9)

    def create_new_state():
        goal_distances = point_dist(g[0], g[1], p_posX, p_posY)
        goal_headings = cos_plus(find_direction(p_posX, p_posY, g[0], g[1]) - p_angle)

        obs_distances = point_dist(
            O_posX,
            O_posY,
            p_posX,
            p_posY,
        )
        min_dist_obs = jnp.argmin(obs_distances)
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

        max_heading_obs = jnp.argmax(obs_headings)
        max_obstacle_headings = obs_headings[max_heading_obs]

        max_heading_obstacle_distances = obs_distances[max_heading_obs]

        op_headings = (
            cos_plus(find_direction(O_posX, O_posY, p_posX, p_posY) - O_angle) + 2
        )

        max_heading_op = jnp.argmax(op_headings)

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

    R = jnp.array(target_return).reshape(1, 1, 1)
    if days is not None:
        s = create_new_state().reshape(1, 1, 16)
    else:
        s = create_new_state().reshape(1, 1, 15)
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
    keys = jax.random.split(key10, max_horizon + 1)
    key = keys[0]
    data_keys = keys[1:]
    for i in range(max_horizon):
        a = jnp.concat([a, jnp.zeros((1, 1, 3))], axis=1)
        a = a[-context_length:]
        _, _, action = d_model._train_state.apply_fn(
            d_model._train_state.params,
            R,
            s,
            a,
            t,
            training=False,
            attn_mask=jnp.ones((1, R.shape[1]), dtype=jnp.float32),
        )

        action = action[-1][-1]
        a = a.at[-1, -1].set(action)

        preds, _ = r_model._train_state.apply_fn(
            r_model._train_state.params,
            s,
            a,
            t,
            training=False,
            attn_mask=jnp.ones((1, R.shape[1]), dtype=jnp.float32),
        )
        reward = preds["value"][:, 0, -1]

        old_p_posX = p_posX
        old_p_posY = p_posY
        p_posX = p_posX + (action[0] * cos_plus(action[1]))
        p_posY = p_posY + (action[0] * sin_plus(action[1]))

        coll, _, _ = collision(
            float(old_p_posX),
            float(old_p_posY),
            float(p_posX),
            float(p_posY),
            O_posX,
            O_posY,
        )

        o_dists = (
            move_stats[3] * jax.random.normal(data_keys[i], shape=(n_obstacles,))
        ) + move_stats[2]
        old_O_posX = O_posX
        old_O_posY = O_posY
        O_posX = O_posX + (o_dists * cos_plus(O_angle))
        O_posY = O_posY + (o_dists * sin_plus(O_angle))

        g_o_dist = jnp.sqrt((O_posX**2) + (O_posY**2))
        O_posX = jnp.where(g_o_dist > 50.0, -old_O_posX, O_posX)
        O_posY = jnp.where(g_o_dist > 50.0, -old_O_posY, O_posY)

        coll, _, _ = collision(
            old_O_posX,
            old_O_posY,
            O_posX,
            O_posY,
            float(p_posX),
            float(p_posY),
        )

        g_coll, _, _ = collision(
            float(old_p_posX),
            float(old_p_posY),
            float(p_posX),
            float(p_posY),
            g[0],
            g[1],
            radius_2=1.0,
        )
        p_angle = action[1]
        if days is not None:
            s = jnp.concat([s, create_new_state().reshape(1, 1, 16)], axis=1)
        else:
            s = jnp.concat([s, create_new_state().reshape(1, 1, 15)], axis=1)
        s[-context_length:]

        R = jnp.concat([R, (R[-1][-1] - reward).reshape(1, 1, 1)], axis=1)
        R[-context_length:]

        t = jnp.concat([t, (t[-1][-1] + 1).reshape(1, 1)], axis=1)
        t[-context_length:]

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
