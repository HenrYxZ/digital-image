import numpy as np
from PIL import Image

# Local modules
import filters
import utils


MAX_QUALITY = 95
COLOR_CHANNELS = 3
MAX_COLOR = 255

IMAGES_DIR = "images"
IMG_FILENAME = f"{IMAGES_DIR}/mickey.jpg"
OUTPUT_BOX_BLUR = "blur.jpg"
OUTPUT_MOTION_BLUR = "motion.jpg"


def main():
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for box blur\n"
            "[2] for motion blur\n"
            "[3] for dilate\n"
            "[4] for erode\n"
            "[5] for edge\n"
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
            kernel = filters.box_blur_kernel(5)
            output_arr = filters.convolve(img_arr, kernel)
        elif opt == '2':
            print("Creating motion blur...")
            output_filename = OUTPUT_MOTION_BLUR
            kernel = filters.motion_blur_kernel(5)
            output_arr = filters.convolve(img_arr, kernel)
        elif opt == '3':
            print("Creating dilation...")
            output_filename = "dilation.jpg"
            img_arr = utils.open_image(f"{IMAGES_DIR}/circle.jpg")
            output_arr = filters.dilate(img_arr)
        elif opt == '4':
            print("Creating erosion...")
            output_filename = "erosion.jpg"
            img_arr = utils.open_image(f"{IMAGES_DIR}/box.jpg")
            output_arr = filters.erode(img_arr)
        else:
            print("Creating edge...")
            output_filename = "edge.jpg"
            grayscale = np.array(img.convert('L'))
            output_arr = filters.edge(grayscale)
        timer.stop()
        print(f"Total time spent: {timer}")
        output_img = Image.fromarray(output_arr)
        output_img.save(output_filename, quality=MAX_QUALITY)
        print(f"Image saved in {output_filename}")


if __name__ == '__main__':
    main()
