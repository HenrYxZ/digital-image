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
    scale = 30
    center = np.array([width / 2, height / 2])
    theta0 = 90
    theta1 = 25
    theta2 = 180 - theta1
    theta3 = -50
    theta4 = 180 - theta3
    angles = [theta0, theta1, theta2, theta3, theta4]
    distances = scale * np.array([1, 1, 1, 0.75, 0.75])
    ns = [normal_from_angle(angle) for angle in angles]
    p0s = [center + ns[i] * distances[i] for i in range(len(distances))]

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


def create_line(p0, p1):
    def f(x, y):
        return 0
    return f
