import errno
import numpy as np
from PIL import Image
import os
import os.path

# Local Modules
from ppm import PPMFile

FUNCTION_FILENAME = "function.ppm"
M = 3
N = 3
MAX_COLOR = 255


def func_img_arr():
    w, h = 512, 512
    scale = 33 / w
    img_arr = np.zeros([h, w, 3], dtype=np.uint8)
    center = np.array([w / 2.0, h / 2.0])
    # Create a random generator
    rng = np.random.default_rng()
    num_samples = M * N
    print("Calculating function...")
    for j in range(h):
        for i in range(w):
            color = np.zeros(3)
            for n in range(N):
                for m in range(M):
                    x = (i - center[0] + (rng.random() * m / M)) * scale
                    y = (j - center[1] + (rng.random() * n / N)) * scale
                    z = np.sin(x * y) / x * y
                    color += (
                        round((i / w) * MAX_COLOR),
                        round((j / h) * MAX_COLOR),
                        round(z * MAX_COLOR),
                    )
            img_arr[j][i] = (color / num_samples).round()
    return img_arr


def create_ppm_from_func():
    img_arr = func_img_arr()
    ppm_file = PPMFile(FUNCTION_FILENAME)
    ppm_file.from_array(img_arr)
    ppm_file.save()


def read_ppm(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename
        )
    ppm_file = PPMFile(filename)
    img = Image.fromarray(ppm_file.data)
    img.show()


def main():
    while(True):
        print(
            "Select an option:\n"
            "[1] Create an image from a function\n"
            "[2] Load the ppm file for function\n"
            "[3] Show mickey image from a ppm file\n"
            "[0] to quit"
        )
        opt = input()
        if opt == '1':
            create_ppm_from_func()
        elif opt == '2':
            read_ppm(FUNCTION_FILENAME)
        elif opt == '3':
            read_ppm("mickey.ppm")
        else:
            break


if __name__ == '__main__':
    main()
