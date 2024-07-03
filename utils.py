from PIL import Image
import numpy as np
import time


MAX_COLOR = 255


def open_image(img_filename):
    img = Image.open(img_filename)
    img_arr = np.array(img)
    return img_arr


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
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


def degrees2radians(degrees):
    return (degrees / 360) * 2 * np.pi


def normalize_color(color):
    return color / MAX_COLOR


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
    # Bilinear interpolation
    color = (
        img_arr[j - 1][i - 1] * (1 - t) * (1 - s)
        + img_arr[j - 1][i] * t * (1 - s)
        + img_arr[j][i - 1] * (1 - t) * s
        + img_arr[j][i] * t * s
    )
    return color


def blerp_uv(img_arr, u, v):
    height, width = img_arr.shape[:2]
    x = u * width
    y = v * height
    return blerp(img_arr, x, y)


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
