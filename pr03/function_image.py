"""
Create an image from an implicit function F(x, y) < 0
"""
from progress.bar import Bar
import numpy as np

# M and N are samples for antialiasing
DEFAULT_M = 4
DEFAULT_N = 4
# Create a random generator
rng = np.random.default_rng()
MAX_COLOR = 255
COLOR_CHANNELS = 3


def create_from_function(
        implicit_function, w, h, c0, c1, m_samples=DEFAULT_M,
        n_samples=DEFAULT_N
):
    """
    Create an image array for an implicit function.

    Args:
        implicit_function(function): The function F(x,y)
        w(int): Width for the image
        h(int): Height for the image
        c0(ndarray): The color for inside the function
        c1(ndarray): The color for outside the function
        m_samples(int): Number of horizontal samples for antialiasing
        n_samples(int): Number of vertical samples for antialiasing

    Returns:
        array: A numpy array uint8 with the image
    """
    total_samples = m_samples * n_samples
    im_array = np.zeros([h, w, COLOR_CHANNELS])
    total_iterations = h * w
    bar = Bar('Processing', max=total_iterations, suffix='%(percent)d%%')
    for j in range(h):
        for i in range(w):
            for n in range(n_samples):
                for m in range(m_samples):
                    x = i + (rng.random() * m / m_samples)
                    y = j + (rng.random() * n / n_samples)
                    f = implicit_function(x, y)
                    if f < 0:
                        im_array[j][i] += c0
                    else:
                        im_array[j][i] += c1
            im_array[j][i] /= total_samples
            bar.next()
    bar.finish()
    # Turn to image
    im_array = im_array.round() * MAX_COLOR
    im_array = im_array.astype(np.uint8)
    return im_array
