import numpy as np
from progress.bar import Bar

# Local Modules
import utils


DEFAULT_SIZE = 3
DEFAULT_THETA = np.pi / 4
RGB_CHANNELS = 3


def box_blur_kernel(size=DEFAULT_SIZE):
    kernel = np.ones((size, size))
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel


def motion_blur_kernel(size=DEFAULT_SIZE, thickness=0.5, theta=DEFAULT_THETA):
    kernel = np.zeros((size, size))
    n = utils.normalize(np.array([np.cos(90 - theta), np.sin(90 - theta)]))
    for j in range(size):
        for i in range(size):
            x = i + 0.5
            y = size - (j + 0.5)
            p = np.array([x, y])
            dist = np.dot(p, n)
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
                        x = 0
                    if y > h - 1:
                        y = h - 1
                    color += np.round(img_arr[y][x] * kernel[n][m])
            output[j][i] = color
            counter += 1
            if counter % step_size == 0:
                bar.next()
    bar.finish()
    return output
