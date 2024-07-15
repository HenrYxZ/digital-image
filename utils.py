from constants import RGB_CHANNELS
from PIL import Image
from numba import njit
import numpy as np
import time
from typing import Callable


MAX_COLOR = 255


def open_image(img_filename):
    img = Image.open(img_filename)
    img_arr = np.array(img)
    return img_arr


def backwards_mapping(
    img_arr: np.ndarray,
    h1: int,
    w1: int,
    sample_func: Callable[[np.ndarray, float, float], np.ndarray]
) -> np.ndarray:
    if len(img_arr.shape) == 2:
        new_shape = [h1, w1]
    else:
        new_shape = [h1, w1, img_arr.shape[2]]
    new_arr = np.zeros(new_shape, dtype=np.uint8)
    num_pixels = h1 * w1

    for counter in range(num_pixels):
        j = int(counter / w1)
        i = int(counter % w1)
        # sample the corresponding pixel in the original array
        u = (i + 0.5) / w1
        v = (h1 - (j + 0.5)) / h1   # y would be going from bottom to top
        new_arr[j, i] = sample_func(img_arr, u, v)
        counter += 1
    return new_arr


def normalize(arr):
    """
    Normalize a vector using numpy.
    Args:
        arr(ndarray): Input vector
    Returns:
        ndarray: Normalized input vector
    """
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def distance(p1, p2):
    """
    Get the distance between points p1 and p2
    Args:
        p1(ndarray): Point 1
        p2(ndarray): Point 2
    Returns:
         float: Distance
    """
    dist = np.linalg.norm(p1 - p2)
    return dist


def humanize_time(secs):
    minutes, secs = divmod(secs, 60)
    hours, minutes = divmod(minutes, 60)
    return '%02d:%02d:%02d' % (hours, minutes, secs)


def degrees2radians(degrees):
    return (degrees / 360) * 2 * np.pi


def normalize_color(color):
    return color / MAX_COLOR


@njit
def scale_blerp_njit(img_arr: np.ndarray, h1: int, w1: int) -> np.ndarray:
    new_arr = np.zeros((h1, w1, RGB_CHANNELS), dtype=np.uint8)

    for j in range(h1):
        for i in range(w1):
            # sample the corresponding pixel in the original array
            u = (i + 0.5) / w1
            v = (h1 - (j + 0.5)) / h1  # y would be going from bottom to top
            new_arr[j, i] = blerp_uv_njit(img_arr, u, v)
    return new_arr


@njit
def scale_nn_njit(img_arr: np.ndarray, h1: int, w1: int) -> np.ndarray:
    new_arr = np.zeros((h1, w1, RGB_CHANNELS), dtype=np.uint8)

    for j in range(h1):
        for i in range(w1):
            # sample the corresponding pixel in the original array
            u = (i + 0.5) / w1
            v = (h1 - (j + 0.5)) / h1  # y would be going from bottom to top
            new_arr[j, i] = nearest_neighbor_uv_njit(img_arr, u, v)
    return new_arr


# Sample functions
# -----------------------------------------------------------------------------
def blerp(img_arr, x, y):
    height, width = img_arr.shape[:2]
    # Interpolate values of pixel neighborhood of x and y
    i = int(x)
    # Flip y value to go from top to bottom
    y = height - y
    j = int(y)
    # But not in the borders
    if i == 0 or j == 0 or i == width or j == height:
        if i == width:
            i -= 1
        if j == height:
            j -= 1
        return img_arr[j][i]
    # t and s are interpolation parameters that go from 0 to 1
    t = x - i
    s = y - j
    # Bi-linear interpolation
    color = np.round(
        img_arr[j - 1][i - 1] * (1 - t) * (1 - s)
        + img_arr[j - 1][i] * t * (1 - s)
        + img_arr[j][i - 1] * (1 - t) * s
        + img_arr[j][i] * t * s
    ).astype(np.uint8)
    return color


def blerp_uv(img_arr, u, v):
    height, width = img_arr.shape[:2]
    x = u * width
    y = v * height
    return blerp(img_arr, x, y)


def nearest_neighbor(img_arr, x, y):
    height, width = img_arr.shape[:2]
    i = int(x)
    # Flip y value to go from top to bottom
    y = height - y
    j = int(y)
    return img_arr[j, i]


def nearest_neighbor_uv(img_arr, u, v):
    height, width = img_arr.shape[:2]
    x = u * width
    y = v * height
    return nearest_neighbor(img_arr, x, y)


@njit
def nearest_neighbor_uv_njit(
    img_arr: np.ndarray, u: float, v: float
) -> np.ndarray:
    height, width = img_arr.shape[:2]
    x = u * width
    y = v * height
    i = int(x)
    # Flip y value to go from top to bottom
    y = height - y
    j = int(y)
    return img_arr[j, i]


@njit
def blerp_uv_njit(img_arr: np.ndarray, u: float, v: float) -> np.ndarray:
    height, width = img_arr.shape[:2]
    x = u * width
    y = v * height
    # Interpolate values of pixel neighborhood of x and y
    i = int(x)
    # Flip y value to go from top to bottom
    y = height - y
    j = int(y)
    # But not in the borders
    if i == 0 or j == 0 or i == width or j == height:
        if i == width:
            i -= 1
        if j == height:
            j -= 1
        return img_arr[j][i]
    # t and s are interpolation parameters that go from 0 to 1
    t = x - i
    s = y - j
    # Bi-linear interpolation
    color = np.round(
        img_arr[j - 1][i - 1] * (1 - t) * (1 - s)
        + img_arr[j - 1][i] * t * (1 - s)
        + img_arr[j][i - 1] * (1 - t) * s
        + img_arr[j][i] * t * s
    ).astype(np.uint8)
    return color


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
