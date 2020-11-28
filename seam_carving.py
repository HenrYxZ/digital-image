import numpy as np
from PIL import Image

# Local modules
from constants import MAX_QUALITY
import utils

CARVING_COUNT = 50
COLOR_CHANNELS = 1
V_SAMPLES = 3
H_SAMPLES = 3
TOTAL_SAMPLES = V_SAMPLES * H_SAMPLES
IMAGES_DIR = "images"
IMG_FILENAME = f"{IMAGES_DIR}/field.jpg"


def gradient(img, h, w):
    gradient = np.ones([h, w]) * np.inf
    for j in range(h):
        for i in range(1, w - 1):
            gradient[j][i] = np.linalg.norm(img[j][i - 1] - img[j][i + 1])
    return gradient


def get_min_seam(img, h, w):
    seam_array = np.ones([h, w]) * np.inf
    min_seam_col = 0
    min_total = np.inf
    grad_arr = gradient(img, h, w)
    for i in range(1, w - 1):
        total = 0
        parent_energy = grad_arr[0][i]
        min_i = i
        for j in range(1, h - 1):
            energy_0 = grad_arr[j][i - 1]
            energy_1 = grad_arr[j][i]
            energy_2 = grad_arr[j][i + 1]
            abs_diffs = np.abs(
                np.array([
                    energy_0 - parent_energy,
                    energy_1 - parent_energy,
                    energy_2 - parent_energy
                ])
            )
            min_idx = np.argmin(abs_diffs)
            total += abs_diffs[min_idx]
            min_i = min_i - 1 + min_idx
            seam_array[j][i] = min_i
        seam_array[h - 1][i] = total
        if total < min_total:
            min_total = total
            min_seam_col = i
    min_seam = [seam_array[j][min_seam_col] for j in range(1, h - 1)]
    min_seam.insert(0, min_seam_col)
    min_seam.append(min_seam_col)
    return min_seam


def carve(img, h, w, channels):
    new_img = np.zeros([h, w - 1, channels], dtype=np.uint8)
    min_seam = get_min_seam(img, h, w)
    for j in range(h):
        for i in range(w):
            if i < min_seam[j]:
                new_img[j][i] = img[j][i]
            elif i > min_seam[j]:
                new_img[j][i - 1] = img[j][i]
    return new_img


def seam_carve(img, count):
    new_img = np.copy(img)
    h, w, channels = img.shape
    for i in range(count):
        print(f"Carving {i + 1} of {count}")
        new_img = carve(new_img, h, w - i, channels)
    return new_img


def main():
    img = Image.open(IMG_FILENAME)
    img_arr = np.array(img)
    timer = utils.Timer()
    timer.start()
    output = seam_carve(img_arr, CARVING_COUNT)
    output_img = Image.fromarray(output)
    output_img.save("output.jpg", quality=MAX_QUALITY)
    print("Image saved in output.jpg")
    timer.stop()
    print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
