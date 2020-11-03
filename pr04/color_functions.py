import numpy as np


def rgb_to_hsv(color):
    """
    Change a normalized RGB color to HSV
    Args:
        color: normalized RGB color

    Returns:
        darray: color in HSV
    """
    r, g, b = color
    max_value = max(color)
    min_value = min(color)
    v = max_value
    if max_value == 0:
        s = 0
        h = 0
    else:
        #  saturation is color purity on scale 0 - 1
        s = (max_value - min_value) / max_value
        delta = max_value - min_value
        # hue doesn't matter if saturation is 0
        if delta == 0:
            h = 0
        else:
            if r == max_value:
                h = (g - b) / delta
            elif g == max_value:
                h = 2 + (b - r) / delta
            else:
                h = 4 + (r - g) / delta
            h *= 60
            if h < 0:
                h += 360
    hsv_color = np.array([h, s, v])
    return hsv_color


def hsv_to_rgb(color):
    return color


def linear_function(color):
    new_color = color.copy()
    return new_color


def change_hue(color, other_color):
    hsv_color = rgb_to_hsv(color)
    hsv_other = rgb_to_hsv(other_color)
    new_hsv = np.array([hsv_other[0], hsv_color[1], hsv_color[2]])
    new_color = hsv_to_rgb(new_hsv)
    return new_color
