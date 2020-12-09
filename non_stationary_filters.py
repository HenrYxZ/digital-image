import numpy as np
from progress.bar import Bar

# Local Modules
from constants import MAX_COLOR
import utils


RGB_CHANNELS = 3
DEFAULT_SIZE = 9


def motion_blur_kernel_from(guide_color, size=DEFAULT_SIZE, thickness=1.5):
    thickness = max(thickness, thickness ** 2)
    kernel = np.zeros((size, size))
    n = utils.normalize(guide_color[:2])
    for j in range(size):
        for i in range(size):
            x = i + 0.5
            y = size - (j + 0.5)
            p = np.array([x, y])
            dif_vec = p - np.dot(p, n) * n
            dist_sq = np.dot(dif_vec, dif_vec)
            if dist_sq < thickness:
                kernel[j][i] = 1
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel


def motion_blur(img_arr, guide_arr, filter_size=DEFAULT_SIZE):
    h, w, _ = img_arr.shape
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    iterations = h * w
    step_size = np.ceil(iterations / 100).astype(int)
    counter = 0
    bar = Bar("Using convolution...", max=100, suffix='%(percent)d%%')
    bar.check_tty = False
    for j in range(h):
        for i in range(w):
            color = np.zeros(RGB_CHANNELS)
            guide_color = guide_arr[j][i]
            kernel = motion_blur_kernel_from(guide_color, filter_size)
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


def dilate(img_arr, dilation_arr):
    max_size = DEFAULT_SIZE
    h, w, _ = img_arr.shape
    output = np.copy(img_arr)
    for j in range(max_size, h - max_size):
        for i in range(max_size, w - max_size):
            value = img_arr[j][i]
            if len(dilation_arr.shape) > 2:
                channel = np.random.randint(0, 3)
                size = int(max_size * (dilation_arr[j][i][channel] / MAX_COLOR))
            else:
                size = int(max_size * (dilation_arr[j][i] / MAX_COLOR))
            shape = [
                (i, j) for j in range(-size, size + 1) for i in
                range(-size, size + 1)
            ]
            for point in shape:
                color = img_arr[j + point[1]][i + point[0]]
                if np.dot(value, value) > np.dot(color, color):
                    value = color
            output[j][i] = value
    return output
