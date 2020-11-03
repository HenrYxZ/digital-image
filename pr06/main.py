import numpy as np
from PIL import Image
import random

# Local modules
import filters
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

IMG_FILENAME = "mickey.jpg"
OUTPUT_BOX_BLUR = "blur.jpg"
OUTPUT_MOTION_BLUR = "motion.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for box blur\n"
            "[2] for motion blur\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img = Image.open(IMG_FILENAME)
        img_arr = np.array(img)
        h, w, _ = img_arr.shape
        # output_arr = np.zeros((h, w, COLOR_CHANNELS))
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Creating box blur...")
            output_filename = OUTPUT_BOX_BLUR
            kernel = filters.box_blur_kernel(11)
            output_arr = filters.convolve(img_arr, kernel)
        else:
            print("Creating motion blur...")
            output_filename = OUTPUT_MOTION_BLUR
            kernel = filters.motion_blur_kernel(11)
            output_arr = filters.convolve(img_arr, kernel)
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save(output_filename, quality=MAX_QUALITY)
        print(f"Image saved in {output_filename}")


if __name__ == '__main__':
    main()
