import numpy as np
from PIL import Image
import random

# Local modules
import functions
from function_image import create_from_function
import line
import utils


WIDTH = 256
HEIGHT = 256
c0 = np.array([0.8, 0.45, 0.87])
c1 = np.array([0.75, 0.75, 0.5])
p0 = np.array([15, 20])
p1 = np.array([180, 243])
MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

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
        print("Creating image ...")
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            f = functions.create_convex(WIDTH, HEIGHT)
            im_arr = create_from_function(f, WIDTH, HEIGHT, c0, c1)
            img_filename = CONVEX_FILENAME
        elif opt == '2':
            f = functions.create_star(WIDTH, HEIGHT)
            im_arr = create_from_function(f, WIDTH, HEIGHT, c0, c1)
            img_filename = STAR_FILENAME
        else:
            im_arr = np.zeros([HEIGHT, WIDTH, COLOR_CHANNELS])
            line.draw(im_arr, p0, p1, c0)
            im_arr = im_arr.round() * MAX_COLOR
            im_arr = im_arr.astype(np.uint8)
            img_filename = LINE_FILENAME
        timer.stop()
        print(f"Total time spent: {timer}")
        img = Image.fromarray(im_arr)
        img.save(img_filename, quality=MAX_QUALITY)
        print(f"Image saved in {img_filename}")


if __name__ == '__main__':
    main()
