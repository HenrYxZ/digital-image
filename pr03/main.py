import numpy as np
from PIL import Image
import random

# Local modules
import functions
from function_image import create_from_function
import utils


WIDTH = 256
HEIGHT = 256
c0 = np.array([0.8, 0.45, 0.87])
c1 = np.array([0.75, 0.75, 0.5])
p0 = np.array([15, 20])
p1 = np.array([180, 243])
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
            f = functions.create_star(WIDTH, HEIGHT)
            img_filename = STAR_FILENAME
        else:
            f = functions.create_line(p0, p1)
            img_filename = LINE_FILENAME
        print("Creating image ...")
        timer = utils.Timer()
        timer.start()
        im_arr = create_from_function(f, WIDTH, HEIGHT, c0, c1)
        img = Image.fromarray(im_arr)
        img.save(img_filename, quality=MAX_QUALITY)
        print(f"Image saved in {img_filename}")
        timer.stop()
        print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
