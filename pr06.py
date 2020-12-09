import numpy as np
from PIL import Image

# Local modules
import non_stationary_filters
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

IMAGES_DIR = "images"
OUTPUT_DIR = "output"
IMG_FILENAME = f"{IMAGES_DIR}/mickey.jpg"
MOTION_BLUR_GUIDE = f"{IMAGES_DIR}/move.jpg"
DILATION_GUIDE = f"{IMAGES_DIR}/dilate.jpg"
OUTPUT_MOTION_BLUR = f"{OUTPUT_DIR}/non_stationary_motion.jpg"
OUTPUT_DILATION = f"{OUTPUT_DIR}/non_stationary_dilation.jpg"


def main():
    while True:
        print("Non-Stationary Filters")
        opt = input(
            "Enter an option:\n"
            "[1] for motion blur\n"
            "[2] for dilate\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img_arr = utils.open_image(IMG_FILENAME)
        h, w, _ = img_arr.shape
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Creating motion blur...")
            output_filename = OUTPUT_MOTION_BLUR
            guide_arr = utils.open_image(MOTION_BLUR_GUIDE)
            output_arr = non_stationary_filters.motion_blur(img_arr, guide_arr)
        else:
            print("Creating dilation...")
            output_filename = OUTPUT_DILATION
            guide_arr = utils.open_image(DILATION_GUIDE)
            output_arr = non_stationary_filters.dilate(img_arr, guide_arr)
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save(output_filename, quality=MAX_QUALITY)
        print(f"Image saved in {output_filename}")


if __name__ == '__main__':
    main()
