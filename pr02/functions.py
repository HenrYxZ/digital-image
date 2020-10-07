import numpy as np
import random

# Local modules
import utils


def random_2d_point(x_range, y_range):
    x = random.uniform(0, x_range)
    y = random.uniform(0, y_range)
    p = np.array([x, y])
    return p


def random_unit_vector2d():
    x = random.random()
    y = random.random()
    n = utils.normalize(np.array([x, y]))
    return n


def create_half_plane(width, height, n=None, p0=None):
    if not n:
        n = random_unit_vector2d()
    if not p0:
        p0 = random_2d_point(width, height)

    def f(x, y):
        p = np.array([x, y])
        distance = np.dot(n, p - p0)
        return distance
    return f


def create_circle(r=None, center=None):
    def f(x, y):
        distance_sqr = (x - center[0]) ** 2 + (y - center[1]) ** 2
        dist = distance_sqr - r ** 2
        return dist
    return f


def create_polygon(center):
    def f(x, y):
        x -= center[0]
        y -= center[1]
        c0 = -30
        c1 = 0
        c2 = 2
        c3 = 2
        c4 = 0.4
        return (
            c4 * y ** 3 * x ** 3 + c3 * x ** 2 + c2 * y ** 2 +
            c1 * x + c0
        )
    return f
