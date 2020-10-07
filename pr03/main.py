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

CONVEX_FILENAME = "convex.jpg"
STAR_FILENAME = "star.jpg"
LINE_FILENAME = "line.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for convex polygon\n"
            "[2] for star\n"
            "[3] for line\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        if opt == '1':
            f = functions.create_convex(WIDTH, HEIGHT)
            img_filename = CONVEX_FILENAME
        elif opt == '2':
            r = random.uniform(WIDTH * 0.05, WIDTH)
            center = np.array([WIDTH / 2, HEIGHT / 2])
            f = functions.create_star(r, center)
            img_filename = STAR_FILENAME
        else:
            center = np.array([WIDTH / 2, HEIGHT / 2])
            f = functions.create_line(center)
            img_filename = LINE_FILENAME
        im_arr = create_from_function(f, WIDTH, HEIGHT, c0, c1)
        img = Image.fromarray(im_arr)
        img.save(img_filename, quality=MAX_QUALITY)
        print(f"Image saved in {img_filename}")


if __name__ == '__main__':
    main()
