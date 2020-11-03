import numpy as np
from PIL import Image
import random

# Local modules
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

IMG_FILENAME = "mickey.jpg"


def prepare_image(im_arr, thickness=100):
    h, w, color_channels = im_arr.shape
    new_arr = np.zeros((h + 2 * thickness, w + 2 * thickness, color_channels))
    for j in range(h):
        for i in range(w):
            new_arr[j + thickness][i + thickness] = im_arr[j][i]
    im_arr = new_arr


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for scale\n"
            "[2] for translate\n"
            "[3] for rotate\n"
            "[4] for shear\n"
            "[5] for perspective\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img = Image.open(IMG_FILENAME)
        img_arr = np.array(img)
        h, w, color_channels = img_arr.shape
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Scaling image...")

            scale = 2
            inverse_matrix = np.array([
                [1 / scale, 0, 0],
                [0, 1 / scale, 0],
                [0, 0, 1]
            ])
        else:
            print("Translating image...")
            inverse_matrix = np.array([
                [1, 0, 10],
                [0, 1, 15],
                [0, 0, 1]
            ])

        output_arr = np.zeros((h, w, color_channels), dtype=np.uint8)
        for j in range(h):
            for i in range(w):
                x = i + 0.5
                y = j + 0.5
                u, v, _ = np.dot(inverse_matrix, np.array([x, y, 1]))
                output_arr[j][i] = img_arr[int(v)][int(u)]
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save("output.jpg", quality=MAX_QUALITY)
        print(f"Image saved in output.jpg")


if __name__ == '__main__':
    main()
