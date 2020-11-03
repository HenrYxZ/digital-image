import numpy as np
import time


MAX_COLOR = 255


def normalize(arr):
    """
    Normalize a vector using numpy.
    Args:
        arr(darray): Input vector
    Returns:
        darray: Normalized input vector
    """
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def humanize_time(secs):
    """
    Extracted from http://testingreflections.com/node/6534
    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02f' % (hours, mins, secs)


def degrees2radians(degrees):
    return (degrees / 360) * 2 * np.pi


def normalize_color(color):
    return color / MAX_COLOR


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def __str__(self):
        return humanize_time(self.elapsed_time)
