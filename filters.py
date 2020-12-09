import numpy as np
from progress.bar import Bar

# Local Modules
from constants import MAX_COLOR
import utils


DEFAULT_SIZE = 9
DEFAULT_THETA = np.pi / 4
RGB_CHANNELS = 3
DX_KERNEL = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
DY_KERNEL = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])
DERIVATIVE = [-1, 0, 1]


def box_blur_kernel(size=DEFAULT_SIZE):
    kernel = np.ones((size, size))
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel


def motion_blur_kernel(size=DEFAULT_SIZE, thickness=1.5, theta=DEFAULT_THETA):
    kernel = np.zeros((size, size))
    n = utils.normalize(
        np.array([np.cos(-(np.pi / 2 - theta)), np.sin(-(np.pi / 2 - theta))])
    )
    for j in range(size):
        for i in range(size):
            x = i + 0.5
            y = size - (j + 0.5)
            p = np.array([x, y])
            dist = np.abs(np.dot(p, n))
            if dist < thickness:
                kernel[j][i] = 1
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel


def convolve(img_arr, kernel):
    h, w, _ = img_arr.shape
    filter_size, _ = kernel.shape
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    iterations = h * w
    step_size = np.ceil(iterations / 100).astype(int)
    counter = 0
    bar = Bar("Using convolution...", max=100, suffix='%(percent)d%%')
    bar.check_tty = False
    for j in range(h):
        for i in range(w):
            color = np.zeros(RGB_CHANNELS)
            # flatten the kernel and the portion of the image
            # row_start = max(0, j - filter_size // 2)
            # col_start = max(0, i - filter_size // 2)
            # row_end = min(h - 1, j + filter_size // 2)
            # col_end = min(w - 1, i + filter_size // 2)
            # weights = kernel.flatten()
            # pixels = []

            # color = np.sum(np.dot())
            # For loops approach
            for n in range(filter_size):
                for m in range(filter_size):
                    x = i + m - filter_size // 2
                    y = j + n - filter_size // 2
                    if x < 0:
                        x = 0
                    if x > w - 1:
                        x = w - 1
                    if y < 0:
                        y = 0
                    if y > h - 1:
                        y = h - 1
                    color += np.round(img_arr[y][x] * kernel[n][m])
            output[j][i] = np.clip(color, 0, MAX_COLOR)
            counter += 1
            if counter % step_size == 0:
                bar.next()
    bar.finish()
    return output


def morphological_filter(img_arr, compare_function):
    size = 2
    shape = [
        (i, j) for j in range(-size, size + 1) for i in range(-size, size + 1)
    ]
    h, w, _ = img_arr.shape
    output = np.copy(img_arr)
    for j in range(size, h - size):
        for i in range(size, w - size):
            value = img_arr[j][i]
            for point in shape:
                color = img_arr[j + point[1]][i + point[0]]
                if compare_function(value, color):
                    value = color
            output[j][i] = value
    return output


def erode(img_arr):
    def compare_function(value, color):
        return np.dot(value, value) < np.dot(color, color)
    return morphological_filter(img_arr, compare_function)


def dilate(img_arr):
    def compare_function(value, color):
        return np.dot(value, value) > np.dot(color, color)
    return morphological_filter(img_arr, compare_function)


def edge(img_arr):
    h, w = img_arr.shape
    output = np.zeros([h, w], dtype=np.uint8)
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            dx = 0
            dy = 0
            # for n in range(-1, 2):
            #    for m in range(-1, 2):
            #         dx += img_arr[j + n][i + m] * DX_KERNEL[n + 1][m + 1]
            #         dy += img_arr[j + n][i + m] * DY_KERNEL[n + 1][m + 1]
            for m in range(3):
                dx += img_arr[j][i + m - 1] * DERIVATIVE[m]
                dy += img_arr[j + m - 1][i] * DERIVATIVE[::-1][m]
            color = 255 - (dx + dy) / 2
            output[j][i] = color
    return output
