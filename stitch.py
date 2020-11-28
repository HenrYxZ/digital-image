import numpy as np
from PIL import Image

# Local modules
from constants import MAX_COLOR, MAX_QUALITY
import utils

COLOR_CHANNELS = 1
V_SAMPLES = 3
H_SAMPLES = 3
TOTAL_SAMPLES = V_SAMPLES * H_SAMPLES
IMAGES_DIR = "images"
IMG_1_FILENAME = f"{IMAGES_DIR}/img1.jpg"
IMG_2_FILENAME = f"{IMAGES_DIR}/img2.jpg"


def get_min_seam(img1, img2, h, w):
    seam_array = np.ones([h, w]) * np.inf
    min_seam_col = 0
    min_total = np.inf
    for i in range(1, w - 1):
        total = 0
        parent_energy = np.linalg.norm(img1[0][i] - img2[0][i])
        min_i = i
        for j in range(1, h - 1):
            energy_0 = np.linalg.norm(img1[j][i - 1] - img2[j][i - 1])
            energy_1 = np.linalg.norm(img1[j][i] - img2[j][i])
            energy_2 = np.linalg.norm(img1[j][i + 1] - img2[j][i + 1])
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
    return min_seam


def main():
    img1 = Image.open(IMG_1_FILENAME)
    img2 = Image.open(IMG_2_FILENAME)
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    h, w, channels = img1_arr.shape
    timer = utils.Timer()
    timer.start()
    output = np.copy(img1)
    print("Calculating min seam...")
    min_seam = get_min_seam(img1_arr, img2_arr, h, w)
    print("Stitching...")
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            if i > min_seam[j - 1]:
                output[j][i] = img2_arr[j][i]
    output_img = Image.fromarray(output)
    output_img.save("output.jpg", quality=MAX_QUALITY)
    print("Image saved in output.jpg")
    timer.stop()
    print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
