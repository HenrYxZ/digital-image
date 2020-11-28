import numpy as np
from PIL import Image

# Local modules
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 1
MAX_COLOR = 255
V_SAMPLES = 3
H_SAMPLES = 3
TOTAL_SAMPLES = V_SAMPLES * H_SAMPLES
IMAGES_DIR = "images"
IMG_FILENAME = f"{IMAGES_DIR}/mickey.jpg"

BLACK = np.zeros(COLOR_CHANNELS)
WHITE = np.ones(COLOR_CHANNELS)
DEFAULT_PALETTE = [BLACK, WHITE]
THRESHOLD_MAP = (1 / 16) * np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
])


def find_closest_color(color, palette):
    min_difference = np.inf
    closest_color = palette[0]
    for palette_color in palette:
        difference = np.abs(color - palette_color)
        if difference < min_difference:
            min_difference = difference
            closest_color = palette_color
    return closest_color


def floyd_steinberg_dithering(img_arr):
    h, w = img_arr.shape
    output = np.copy(img_arr) + np.random.random_sample([h, w])
    for j in range(h):
        for i in range(w):
            x = i if j % 2 == 0 else w - 1 - i
            original_pixel = output[j][x]
            # new_pixel = find_closest_color(original_pixel, palette)
            new_pixel = round(original_pixel)
            output[j][x] = new_pixel
            error = original_pixel - new_pixel
            if j < h - 1 and 0 < x < w - 1 and j % 2 == 0:
                output[j][x + 1] += error * 7 / 16
                output[j + 1][x - 1] += error * 3 / 16
                output[j + 1][x] += error * 5 / 16
                output[j + 1][x + 1] += error * 1 / 16
            if j < h - 1 and 0 < x < w - 1 and j % 2 == 1:
                output[j][x - 1] += error * 7 / 16
                output[j + 1][x - 1] += error * 3 / 16
                output[j + 1][x] += error * 5 / 16
                output[j + 1][x - 1] += error * 1 / 16
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def ordered_dithering(img_arr):
    h, w = img_arr.shape
    output = np.zeros([h, w], dtype=bool)
    for j in range(h):
        for i in range(w):
            if img_arr[j][i] < THRESHOLD_MAP[j % 4][i % 4]:
                output[j][i] = 0
            else:
                output[j][i] = 1
    return output


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for Floyd-Steinberg Error diffusion \n"
            "[2] for Ordered Dithering\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        img = Image.open(IMG_FILENAME)
        grayscale = img.convert('L')
        img_arr = np.array(grayscale, dtype=float) / MAX_COLOR
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Using Floyd-Steinberg Error Diffusion Dithering...")
            output = floyd_steinberg_dithering(img_arr)
        else:
            print("Using Ordered Dithering...")
            output = ordered_dithering(img_arr)
        output_img = Image.fromarray(output)
        output_img.save("output.jpg", quality=MAX_QUALITY)
        print("Image saved in output.jpg")
        timer.stop()
        print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
