import numpy as np
from PIL import Image
import random

# Local modules
import functions
from function_image import create_from_function

WIDTH = 256
HEIGHT = 256
c0 = np.array([0.8, 0.45, 0.87])
c1 = np.array([0.75, 0.75, 0.5])
MAX_QUALITY = 95

HALF_PLANE_FILENAME = "half_plane.jpg"
CIRCLE_FILENAME = "circle.jpg"
POLYNOMIAL_FILENAME = "polynomial.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for random half-plane\n"
            "[2] for random circle\n"
            "[3] for polynomial\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        if opt == '1':
            f = functions.create_half_plane(WIDTH, HEIGHT)
            img_filename = HALF_PLANE_FILENAME
        elif opt == '2':
            r = random.uniform(WIDTH * 0.05, WIDTH)
            center = np.array([WIDTH / 2, HEIGHT / 2])
            f = functions.create_circle(r, center)
            img_filename = CIRCLE_FILENAME
        else:
            center = np.array([WIDTH / 2, HEIGHT / 2])
            f = functions.create_polygon(center)
            img_filename = POLYNOMIAL_FILENAME
        im_arr = create_from_function(f, WIDTH, HEIGHT, c0, c1)
        img = Image.fromarray(im_arr)
        img.save(img_filename, quality=MAX_QUALITY)
        print(f"Image saved in {img_filename}")


if __name__ == '__main__':
    main()
