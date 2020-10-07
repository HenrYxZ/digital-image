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


def normal_from_angle(angle):
    """
    Calculate the normal for a given angle thinking of a circle of r = 1
    Args:
        angle: Angle in degrees

    Returns:
        darray: the normal
    """
    theta = utils.degrees2radians(angle)
    nx = np.cos(theta)
    ny = np.sin(theta)
    n = utils.normalize(np.array([nx, ny]))
    return n


def create_convex(width, height):
    center = np.array([width / 2, height / 2])
    num_points = 8
    angle_step = 360 / num_points
    dist_to_center = 100
    ns = [normal_from_angle(angle_step * i) for i in range(num_points)]
    p0s = [center + n * dist_to_center for n in ns]

    def f(x, y):
        max_distance = -np.inf
        p = np.array([x, y])
        for i in range(num_points):
            n = ns[i]
            p0 = p0s[i]
            distance = np.dot(n, p - p0)
            if distance > max_distance:
                max_distance = distance
        return max_distance
    return f


def create_star(width, height):
    center = np.array([width / 2, height / 2])
    theta1 = 45
    theta2 = 180 - theta1
    theta3 = -30
    theta4 = 180 - theta3
    theta0 = 90
    angles = [theta0, theta1, theta2, theta3, theta4]
    distances = [30, 50, 50, 20, 20]
    ns = [normal_from_angle(angle) for angle in angles]
    p0s = [(center + (ns[i] * distances[i])) for i in range(len(distances))]
    print(ns)
    print(p0s)

    def f(x, y):
        p = np.array([x, y])
        count = 0
        num_planes = len(ns)
        for i in range(len(ns)):
            n = ns[i]
            p0 = p0s[i]
            if np.dot(n, p - p0) < 0:
                count += 1
        if count >= num_planes - 1:
            return -1
        else:
            return 1
    return f


def create_line(center):
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
