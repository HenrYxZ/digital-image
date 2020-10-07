import numpy as np


DEFAULT_COLOR = np.array([240, 255, 200]) / 255


def draw(im_arr, p0, p1, color=DEFAULT_COLOR):
    h, w, _ = im_arr.shape
    x0, y0 = p0
    # a = np.abs(x1 - x0)
    # b = np.abs(y1 - y0)
    a, b = p1 - p0
    L = max(np.abs(a), np.abs(b))
    A = a / L
    B = b / L
    for i in range(L):
        x = int(x0 + i * A + 0.5)
        y = int(y0 + i * B + 0.5)
        im_arr[h - y][x] = color
