import numpy as np
from PIL import Image
import random

# Local modules
import color_functions
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

IMG_FILENAME = "night.jpg"
HUE_FILENAME = "mickey.jpg"
OUTPUT_LINEAR_FILENAME = "output-linear.jpg"
OUTPUT_HUE_FILENAME = "output-hue.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for using a piece-wise linear function\n"
            "[2] for changing hues using HSV color space\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img = Image.open(IMG_FILENAME)
        img_arr = img.asarray()
        h, w, _ = img_arr.shape
        output_arr = np.zeros((h, w, COLOR_CHANNELS))
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            for j in range(h):
                for i in range(w):
                    color = img_arr[j][i]
                    output_arr[j][i] = color_functions.linear_function(color)
            output_filename = OUTPUT_LINEAR_FILENAME
        else:
            hue_img = Image.open(HUE_FILENAME)
            hue_arr = np.array(hue_img)
            for j in range(h):
                for i in range(w):
                    output_arr[j][i] = color_functions.change_hue(
                        img_arr[j][i], hue_arr[j][i]
                    )
            output_filename = OUTPUT_HUE_FILENAME
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save(output_filename, quality=MAX_QUALITY)
        print(f"Image saved in {output_filename}")


if __name__ == '__main__':
    main()
