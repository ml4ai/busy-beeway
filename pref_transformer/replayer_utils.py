import numpy as np


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
    np.seterr(divide="ignore", invalid="ignore")
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
    return np.any(point_collide(cpx, cpy, px, py, radius_1, radius_2)), cpx*1, cpy*1


def find_direction(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    degs = np.arctan2(y, x) * (180.0 / np.pi)
    degs = np.where(np.isclose(degs, 0.0), 360.0, degs)
    degs = np.where(degs < 0, degs + 360.0, degs)
    return degs * 1
